import os
import numpy as np
import torch
from mmcv.runner import force_fp32

from mmdet.models import HEADS
from projects.mmdet3d_plugin.core.bbox.util import decode_points
from projects.mmdet3d_plugin.models.utils.misc import safe_sigmoid, get_rotation_matrix, DUMP
from projects.mmdet3d_plugin.ops import LocalAggregatorExt, local_aggregate_backtrace
from projects.mmdet3d_plugin.models.utils.difficulty_scorer_v2 import (
    per_voxel_difficulty, voxel_structural_difficulty, rank_normalize,
    scorer_ranking_loss, spearman_corr)
from projects.mmdet3d_plugin.models.utils.difficulty_modulation import rank01_tie_aware
from .streamocc_head import StreamOccHead


@HEADS.register_module()
class StreamOccHeadV7(StreamOccHead):
    def __init__(self, *args, lambda_score=0.1, use_difficulty=True, lam_lovasz=0.0,
                 lam_struct=1.0, teacher_use_camera_mask=False, den_thresh=1e-3,
                 memory_select='scale', memory_select_alpha=0.5,
                 **kwargs):
        super().__init__(*args, **kwargs)

        assert memory_select in ('scale', 'difficulty', 'mixed')
        self.memory_select = memory_select
        self.memory_select_alpha = memory_select_alpha
        self.lambda_score = lambda_score
        self.use_difficulty = use_difficulty
        self.lam_lovasz = lam_lovasz

        self.lam_struct = lam_struct
        self.teacher_use_camera_mask = teacher_use_camera_mask

        self.den_thresh = den_thresh
        self._last_bt_caches = None
        self._mem_s_query = None


        backend = str(getattr(self, 'aggregator_backend', 'standard')).lower()
        assert backend in ('standard', 'default', 'base', '_c'), (
            f"StreamOccHeadV7 requires aggregator_backend='standard' (got '{backend}'): "
            "the backtrace teacher replicates the standard kernel's responsibility.")


        self.aggregator = LocalAggregatorExt(
            scale_multiplier=3,
            H=self.voxel_num[0],
            W=self.voxel_num[1],
            D=self.voxel_num[2],
            pc_min=self.pc_range[:3],
            grid_size=self.voxel_size[0],
        )

    def sq2occ(self, cls_scores, refine_sqs):
        num_imgs = cls_scores.size(0)
        cls_scores = cls_scores.flatten(1, 2)
        refine_sqs = refine_sqs.flatten(1, 2)

        gs_mean = refine_sqs[..., :3]
        scales = refine_sqs[..., 3:6]
        rot = refine_sqs[..., 6:10]
        origi_opa = refine_sqs[..., 10:11]
        u = refine_sqs[..., 11:12]
        v = refine_sqs[..., 12:13]

        rots = get_rotation_matrix(rot)
        origi_opa = origi_opa.flatten(1, 2)
        u = u.flatten(1, 2)
        v = v.flatten(1, 2)

        opacities = cls_scores.softmax(dim=-1)
        opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])], dim=-1)

        gt_xyz = self.gt_xyz[None, ...].repeat([num_imgs, 1, 1, 1, 1])
        sampled_xyz = gt_xyz.flatten(1, 3).float()

        semantics = []
        bt_caches = []
        for i in range(num_imgs):
            semantic = self.aggregator(
                sampled_xyz[i:(i + 1)], gs_mean[i:(i + 1)], origi_opa[i:(i + 1)],
                u[i:(i + 1)], v[i:(i + 1)], opacities[i:(i + 1)],
                scales[i:(i + 1)], rots[i:(i + 1)])
            bt_caches.append(getattr(self.aggregator, '_bt_cache', None))

            sem = semantic[0][:, :-1] * semantic[1].unsqueeze(-1)
            geo = 1 - semantic[1].unsqueeze(-1)
            geosem = torch.cat([sem, geo], dim=-1)
            geosem = geosem.reshape(self.voxel_num[0], self.voxel_num[1], self.voxel_num[2], -1)
            semantics.append(geosem)

        occ_pred = torch.stack(semantics, dim=0)
        self._last_bt_caches = bt_caches
        return occ_pred

    # ---- override forward: base (densifier) pipeline + s_scores + bt caches ----
    def forward(self, img_metas, **data):
        if self.temp_fusion:
            self.pre_update_memory(data)

        mlvl_feats = data['img_feats']
        B, Q = mlvl_feats[0].shape[0], self.num_query
        init_points = self.init_points.weight[None, :, :].repeat(B, 1, 1)
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)

        if self.temp_fusion:
            query_feat, reference_points, temp_memory, temp_reference_point, memory_mask, rec_ego_pose = \
                self.temporal_alignment(query_feat, init_points, data)
        else:
            reference_points = init_points
            temp_memory = None
            temp_reference_point = None
            memory_mask = None


        query_feats, cls_scores, refine_sqs, s_scores = self.transformer(
            query_feat,
            reference_points.unsqueeze(dim=2),
            temp_memory,
            temp_reference_point,
            memory_mask,
            mlvl_feats,
            data,
            img_metas=img_metas,
        )


        s_last = s_scores[-1].detach()
        self._mem_s_query = 0.5 * (s_last.amax(dim=2) + s_last.mean(dim=2))   # (B, Q)

        cls_scores_list, pred_occ_list, layer_bt_caches = [], [], []
        for i in range(len(refine_sqs)):
            if not self.training and i < (len(refine_sqs) - 1):
                continue

            sq_mean = decode_points(refine_sqs[i][..., 0:3], self.pc_range)
            sq_scales = safe_sigmoid(refine_sqs[i][..., 3:6])
            sq_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * sq_scales
            rot = refine_sqs[i][..., 6:10]
            opa = safe_sigmoid(refine_sqs[i][..., 10:11])
            uv = safe_sigmoid(refine_sqs[i][..., 11:13])
            u = self.u_range[0] + (self.u_range[1] - self.u_range[0]) * uv[..., :1]
            v = self.v_range[0] + (self.v_range[1] - self.v_range[0]) * uv[..., 1:]
            sqs = torch.cat([sq_mean, sq_scales, rot, opa, u, v], dim=-1)

            voxel_logits, voxel_coors = self.densifiers[i](
                cls_scores[i],                    # (B, Q, R, C_pt) per-point features
                refine_sqs[i][..., :3]            # (B, Q, R, 3) encoded xyz
            )

            cls_score = self.scatter_voxel_to_points(
                self.densifiers[i], voxel_logits, voxel_coors, refine_sqs[i][..., :3]
            )   # (B, Q, R, num_classes)
            cls_scores_list.append(cls_score)

            occ_pred = self.sq2occ(cls_score, sqs)
            pred_occ_list.append(occ_pred)
            layer_bt_caches.append(self._last_bt_caches)

            if DUMP.enabled and i == (len(refine_sqs) - 1):
                scene_name = img_metas[0]['scene_name']
                sample_idx = img_metas[0]['sample_idx']
                save_path = '{}/{}/{}'.format(DUMP.out_dir, scene_name, sample_idx)
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)
                np.savez_compressed(f'{save_path}/pred_superquadrics.npz', sqs=sqs.detach().cpu().numpy())
                np.savez_compressed(f'{save_path}/pred_semantics.npz', cls_score=cls_score.detach().cpu().numpy())
                np.savez_compressed(f'{save_path}/pred_occupancy.npz', occ_pred=occ_pred.detach().cpu().numpy().astype(np.uint8))

        if self.temp_fusion:
            self.post_update_memory(data, rec_ego_pose, query_feats, cls_scores_list, refine_sqs)

        return dict(init_points=init_points,
                    all_cls_scores=cls_scores_list,
                    all_refine_sqs=refine_sqs,
                    all_pred_occ_list=pred_occ_list,
                    all_s_scores=s_scores,
                    layer_bt_caches=layer_bt_caches)

    @staticmethod
    def _rank01(x):
        return rank01_tie_aware(x)

    def post_update_memory(self, data, rec_ego_pose, all_query_feats, all_cls_scores, all_sq_preds):
 
        if self.memory_select == 'scale' or self._mem_s_query is None:
            return super().post_update_memory(
                data, rec_ego_pose, all_query_feats, all_cls_scores, all_sq_preds)

        from projects.mmdet3d_plugin.models.utils.misc import topk_gather, transform_reference_points

        sq_mean = decode_points(all_sq_preds[-1][..., 0:3], self.pc_range)
        rec_reference_points = torch.mean(sq_mean, dim=2)

        rec_memory = all_query_feats[-1]
        rec_timestamp = torch.zeros_like(rec_memory[..., 0:1], dtype=torch.float64)
        mem_mask = torch.ones_like(rec_memory[..., 0], dtype=torch.int)

        scales = all_sq_preds[-1][..., 3:6]
        scale_score = scales.max(dim=-1)[0].max(dim=-1)[0]          # (B, Q)
        if self.memory_select == 'difficulty':
            sel = self._mem_s_query
        else:  # 'mixed'
            a = self.memory_select_alpha
            sel = a * self._rank01(scale_score) + (1.0 - a) * self._rank01(self._mem_s_query)
        _, topk_indexes = torch.topk(sel, self.topk_proposals, dim=1)

        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose = torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_reference_point, _ = transform_reference_points(
            self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_mask = torch.cat([mem_mask, self.memory_mask], dim=1)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose

    def _primitive_difficulty(self, occ_pred, voxel_semantics, bt_caches,
                              h_loc=None, mask_camera=None):
        B = occ_pred.size(0)
        es, dens = [], []
        for i in range(B):
            occ_flat = occ_pred[i].reshape(-1, occ_pred.size(-1))     # (N, 18) same order as cache/pts
            gt_flat = voxel_semantics[i].reshape(-1).long()           # (N,)
            cam = mask_camera[i].reshape(-1) if mask_camera is not None else None
            hl = h_loc[i].reshape(-1) if h_loc is not None else None
            d, valid = per_voxel_difficulty(
                occ_flat, gt_flat, self.ignore_label,
                lovasz_fn=None, lam_lovasz=self.lam_lovasz,
                camera_mask=cam, h_loc=hl, lam_struct=self.lam_struct)
            num, den = local_aggregate_backtrace(bt_caches[i], d, valid)   # (P,), (P,)
            es.append(num / (den + 1e-6))
            dens.append(den)
        return torch.stack(es, 0), torch.stack(dens, 0)   # (B, P=Q*N_refine) x2


    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        loss_dict = super().loss(voxel_semantics, mask_camera, preds_dicts)
        if not self.use_difficulty:
            all_s = preds_dicts.get('all_s_scores', None)
            if all_s is not None:
                
                loss_dict['loss_score'] = sum(s.sum() for s in all_s) * 0.0
            return loss_dict

        occ_list = preds_dicts['all_pred_occ_list']
        all_s = preds_dicts['all_s_scores']
        lay_caches = preds_dicts['layer_bt_caches']

        h_loc = None
        if self.lam_struct > 0.0:
            h_loc = voxel_structural_difficulty(voxel_semantics.long(), self.ignore_label)
        cam = mask_camera if (self.teacher_use_camera_mask and mask_camera is not None) else None

        loss_score = voxel_semantics.new_zeros(()).float()
        diag_corr = voxel_semantics.new_zeros(()).float()
        L = len(occ_list)
        for li in range(L):
            e, den = self._primitive_difficulty(
                occ_list[li], voxel_semantics, lay_caches[li], h_loc=h_loc, mask_camera=cam)
            s = all_s[li].reshape(e.size(0), -1).float()  # (B,P), order q*N_refine+r matches e
            vmask = (den > self.den_thresh).float()       # support-based validity (NOT e > 0)
            loss_score = loss_score + scorer_ranking_loss(s, e, vmask)
            diag_corr = diag_corr + spearman_corr(s.detach(), e, vmask)
        loss_dict['loss_score'] = self.lambda_score * loss_score / max(1, L)

        loss_dict['diag_spearman'] = (diag_corr / max(1, L)).detach()
        decoder = getattr(self.transformer, 'decoder', None)
        adapters = getattr(decoder, 'adapters', None)
        if adapters is not None and len(adapters) > 0:
            gates = torch.stack([torch.tanh(a.gate.detach()).reshape(()) for a in adapters])
            loss_dict['diag_gate'] = gates.abs().mean()
        return loss_dict
