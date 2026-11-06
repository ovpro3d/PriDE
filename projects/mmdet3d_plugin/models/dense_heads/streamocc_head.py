import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from mmcv.runner import force_fp32, BaseModule
from mmcv.ops import knn
from mmcv.cnn import bias_init_with_prob
import torch_scatter
from mmdet.core import multi_apply
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from projects.mmdet3d_plugin.core.bbox.util import decode_points, encode_points
from projects.mmdet3d_plugin.models.utils.positional_encoding import nerf_positional_encoding, pos2posemb, NerfPositionalEncoder
from projects.mmdet3d_plugin.models.utils.misc import safe_sigmoid, get_rotation_matrix, transform_reference_points, \
    memory_refresh, topk_gather
from projects.mmdet3d_plugin.models.loss import lovasz_softmax
from projects.mmdet3d_plugin.ops import LocalAggregator
from projects.mmdet3d_plugin.models.utils.misc import DUMP
from spconv.pytorch import (SparseConvTensor, SparseConv3d, SubMConv3d,
                            SparseSequential)
class PFNLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 use_norm=True,
                 last_layer=False):
        super().__init__()
        
        self.last_vfe = last_layer
        self.use_norm = use_norm
        if not self.last_vfe:
            out_channels = out_channels // 2

        if self.use_norm:
            self.linear = nn.Linear(in_channels, out_channels, bias=False)
            self.norm = nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.01)
        else:
            self.linear = nn.Linear(in_channels, out_channels, bias=True)
        
        self.relu = nn.ReLU()

    def forward(self, inputs, unq_inv):

        x = self.linear(inputs)
        x = self.norm(x) if self.use_norm else x
        x = self.relu(x)
        x_max = torch_scatter.scatter_max(x, unq_inv, dim=0)[0]

        if self.last_vfe:
            return x_max
        else:
            return torch.cat([x, x_max[unq_inv, :]], dim=1)


class Densifier(BaseModule):

    def __init__(self, in_channels, pfn_channels, num_classes, pc_range, voxel_num):
        super().__init__()
        self.pfn_layers = nn.ModuleList()
        for i, pfn_channel in enumerate(pfn_channels):
            last_layer = (i == len(pfn_channels) - 1)
            self.pfn_layers.append(
                PFNLayer(
                    in_channels=in_channels + 6 if i == 0 else pfn_channel,
                    out_channels=pfn_channel,
                    use_norm=True,
                    last_layer=last_layer
                )
            )
        self.cls_branch = SparseSequential(
            SparseConv3d(
                pfn_channels[-1], pfn_channels[-1],
                kernel_size=3, stride=1, padding=1, indice_key='densifier1'
            ),
            nn.ReLU(inplace=True),
            SubMConv3d(
                pfn_channels[-1], num_classes,
                kernel_size=3, stride=1, padding=1, indice_key='densifier2'
            )
        )
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('voxel_num', voxel_num)
        self.scale_xyz = self.voxel_num[0] * self.voxel_num[1] * self.voxel_num[2]
        self.scale_yz = self.voxel_num[1] * self.voxel_num[2]
        self.scale_z = self.voxel_num[2]

    def init_weights(self):
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def discretize(self, pts, clip=True):
        encoded_pts = encode_points(pts, self.pc_range)
        voxel_num = self.voxel_num.to(encoded_pts.dtype)
        loc = torch.floor(encoded_pts * voxel_num)
        if clip:
            loc[..., 0] = loc[..., 0].clamp(0, self.voxel_num[0] - 1)
            loc[..., 1] = loc[..., 1].clamp(0, self.voxel_num[1] - 1)
            loc[..., 2] = loc[..., 2].clamp(0, self.voxel_num[2] - 1)
        
        centers = decode_points((loc + 0.5) / voxel_num, self.pc_range)
        return loc.long(), centers

    def forward(self, pts_feats, refine_pts):
        if pts_feats is None:
            return None, None

        B, Q, P, _ = refine_pts.shape
        refine_pts = decode_points(refine_pts, self.pc_range)

        # calculate voxel coordinates
        batch_index = torch.arange(B, device=pts_feats.device).view(B, 1, 1, 1).expand(B, Q, P, 1)
        coors, coor_centers = self.discretize(refine_pts.detach())
        coors = torch.cat([batch_index, coors], dim=-1)
        coors = coors[..., 0] * self.scale_xyz + coors[..., 1] * self.scale_yz + \
            coors[..., 2] * self.scale_z + coors[..., 3]
        
        # calculate voxel features
        centers = refine_pts.mean(dim=2, keepdim=True)
        pts_feats = torch.cat([refine_pts-coor_centers, refine_pts-centers, pts_feats], dim=-1)
        pts_feats = pts_feats.reshape(B * Q * P, -1)
        coors = coors.reshape(B * Q * P)
        
        unq_coors, unq_inv = coors.unique(return_inverse=True)
        for pfn_layer in self.pfn_layers:
            pts_feats = pfn_layer(pts_feats, unq_inv)
        voxel_coors = torch.stack(
            [unq_coors // self.scale_xyz,
             (unq_coors % self.scale_xyz) // self.scale_yz,
             (unq_coors % self.scale_yz) // self.scale_z,
             (unq_coors % self.scale_z)], dim=1
        )

        sparse_feats = SparseConvTensor(
            features=pts_feats,
            indices=voxel_coors.int(),
            spatial_shape=self.voxel_num.tolist(),
            batch_size=B)
        res = self.cls_branch(sparse_feats)
        densified_cls_scores = res.features
        densified_voxel_coors = res.indices.long() # from int32 to int64

        return densified_cls_scores, densified_voxel_coors


@HEADS.register_module()
class StreamOccHead(BaseModule):
    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query,
                 memory_len,
                 topk_proposals=500,
                 num_propagated=500,
                 prop_query=False,
                 temp_fusion=False,
                 with_ego_pos=False,
                 aggregator_backend='standard',
                 transformer=None,
                 empty_label=17,
                 ignore_label=255,
                 pfn_channels=[64,64],
                 pc_range=[],
                 voxel_size=[],
                 scale_range=[0.01, 3.2],
                 u_range=[0.1, 2],
                 v_range=[0.1, 2],
                 nusc_class_frequencies=[],
                 manual_class_weight=None,
                 score_thres=None,
                 use_geo_hardness=False,
                 geo_hardness_alpha=1.0,
                 geo_hardness_beta=1.0,
                 loss_occ=None,
                 loss_pts=None,
                 train_cfg=dict(),
                 test_cfg=dict(),
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg)
        self.num_query = num_query
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.prop_query = prop_query
        self.temp_fusion = temp_fusion
        self.with_ego_pos = with_ego_pos
        self.aggregator_backend = aggregator_backend
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.empty_label = empty_label
        self.transformer = build_transformer(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.score_thres = score_thres

        self.scale_range = scale_range
        self.u_range = u_range
        self.v_range = v_range
        self.ignore_label = ignore_label

        self.use_geo_hardness = use_geo_hardness
        self.geo_hardness_alpha = geo_hardness_alpha
        self.geo_hardness_beta = geo_hardness_beta

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()

        self.aggregator = LocalAggregator(
            scale_multiplier=3,
            H=voxel_num[0],
            W=voxel_num[1],
            D=voxel_num[2],
            pc_min=pc_range[:3],
            grid_size=voxel_size[0],
            backend=self.aggregator_backend,
        )
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)
        xyz = self.get_meshgrid(pc_range, voxel_num, voxel_size)
        self.register_buffer('gt_xyz', torch.tensor(xyz))

        self._init_layers()

        # build densifying network
        self.densifiers = nn.ModuleList()
        for _ in range (self.transformer.num_layers):
            self.densifiers.append(
                Densifier(
                    in_channels = self.transformer.num_pt_channels,
                    pfn_channels=pfn_channels,
                    num_classes=num_classes,
                    pc_range=self.pc_range,
                    voxel_num=self.voxel_num
                )
            )

        if manual_class_weight is not None:
            self.class_weights = torch.tensor(manual_class_weight, dtype=torch.float)
            self.cls_weights = (num_classes + 1) * F.normalize(self.class_weights, 1, -1)
        else:
            class_freqs = nusc_class_frequencies
            self.cls_weights = torch.from_numpy(1 / np.log(np.array(class_freqs[:num_classes+1]) + 0.001))

        loss_occ['class_weight'] = self.cls_weights
        loss_occ['ignore_label'] = self.ignore_label
        self.loss_occ = build_loss(loss_occ)
        self.loss_pts = build_loss(loss_pts)
        self.reset_memory()

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)

        # encoding ego pose
        if self.with_ego_pos:
            self.nerf_encoder = NerfPositionalEncoder(num_encoding_functions=6)
            self.ego_pose_memory = MLN(156)

    def init_weights(self):
        self.transformer.init_weights()
        for densifier in self.densifiers:
            densifier.init_weights()

    def get_meshgrid(self, ranges, grid, reso):
        xxx = torch.arange(grid[0], dtype=torch.float) * reso[0] + 0.5 * reso[0] + ranges[0]
        yyy = torch.arange(grid[1], dtype=torch.float) * reso[1] + 0.5 * reso[1] + ranges[1]
        zzz = torch.arange(grid[2], dtype=torch.float) * reso[2] + 0.5 * reso[2] + ranges[2]

        xxx = xxx[:, None, None].expand(*grid)
        yyy = yyy[None, :, None].expand(*grid)
        zzz = zzz[None, None, :].expand(*grid)

        xyz = torch.stack([
            xxx, yyy, zzz
        ], dim=-1).numpy()  # （Dx， Dy, Dz, 3)
        return xyz

    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None

    def pre_update_memory(self, data):
        x = data['prev_exists']
        B = x.size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims).float()
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3).float()
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1).float()
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4).float()
            self.memory_mask = x.new_zeros(B, self.memory_len).int()
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point, _ = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)

            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_mask = memory_refresh(self.memory_mask[:, :self.memory_len], x)

    def post_update_memory(self, data, rec_ego_pose, all_query_feats, all_cls_scores, all_sq_preds):
        sq_mean = decode_points(all_sq_preds[-1][..., 0:3], self.pc_range)  # (B, Q，N_refine, 3)
        rec_reference_points = torch.mean(sq_mean, dim=2)       # (B, Q, 3)

        rec_memory = all_query_feats[-1]    # (B, Q, C)
        rec_timestamp = torch.zeros_like(rec_memory[..., 0:1], dtype=torch.float64)
        mem_mask = torch.ones_like(rec_memory[..., 0], dtype=torch.int)

        scales = all_sq_preds[-1][..., 3:6]         # (B, Q，N_refine, 3)
        s = scales.max(dim=-1)[0].max(dim=-1)[0]    # (B, Q)
        _, topk_indexes = torch.topk(s, self.topk_proposals, dim=1)

        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose = torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_reference_point, _ = transform_reference_points(self.memory_reference_point, data['ego_pose'],
                                                                    reverse=False)
        self.memory_mask = torch.cat([mem_mask, self.memory_mask], dim=1)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose

    def prop_queries(self, query_feat, reference_points, temp_memory, temp_reference_point, data):
        """
        Args:
            query_feat: (B, Q, C)
            reference_points: (B, Q, 3)
            temp_memory: (B, Mem, C)
            temp_reference_point: (B, Mem, 3)

        Returns:
            new_query_feat: (B, Q, C)
            new_reference_points: (B, Q, 3)
        """
        B, Q, C = query_feat.shape
        num_prop = self.num_propagated
        num_keep = Q - num_prop

        prev_exists = data["prev_exists"].view(B, 1, 1)
        prop_feat = temp_memory[:, :num_prop]  # (B, num_prop, C)
        prop_reference_points = temp_reference_point[:, :num_prop]  # (B, num_prop, 3)

        real_prop_reference_points = decode_points(prop_reference_points, self.pc_range)  # (B, N_prop，3)
        real_reference_points = decode_points(reference_points, self.pc_range)  # (B, N_init，3)

        # (B, num_prop, Q, 3)
        diff = real_prop_reference_points[:, :, None, :] - real_reference_points[:, None, :, :]
        dist2 = (diff ** 2).sum(-1)   # (B, num_prop, Q)
        min_dist2 = dist2.min(dim=1).values     # (B, Q)
        far_mask = min_dist2 > 1.0      # (B, Q)

        rand = torch.rand_like(far_mask.float())
        score = rand.masked_fill(~far_mask, -1e8)
        _, idx = score.topk(num_keep, dim=1)  # (B, num_keep)
        idx_feat = idx[..., None].expand(-1, -1, C)
        idx_ref = idx[..., None].expand(-1, -1, 3)

        selected_feat = query_feat.gather(1, idx_feat)
        selected_points = reference_points.gather(1, idx_ref)
        merged_feat = torch.cat([prop_feat, selected_feat], dim=1)  # (B,Q,C)
        merged_ref = torch.cat([prop_reference_points, selected_points], dim=1)  # (B,Q,3)

        new_query_feat = torch.where(prev_exists, merged_feat, query_feat)
        new_reference_points = torch.where(prev_exists, merged_ref, reference_points)

        return new_query_feat, new_reference_points

    def temporal_alignment(self, query_feat, reference_points, data):
        B = query_feat.size(0)
        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / \
                               (self.pc_range[3:6] - self.pc_range[0:3])
        temp_memory = self.memory_embedding
        memory_mask = self.memory_mask
        rec_ego_pose = torch.eye(4, device=query_feat.device).unsqueeze(0).unsqueeze(0).repeat(B, query_feat.size(1), 1, 1)

        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[..., :1]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = self.nerf_encoder(rec_ego_motion)
            query_feat = self.ego_pose_memory(query_feat, rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_timestamp.float(), self.memory_egopose[..., :3, :].flatten(-2)], dim=-1)
            memory_ego_motion = self.nerf_encoder(memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        if self.prop_query:
            query_feat, reference_points = self.prop_queries(query_feat, reference_points, temp_memory, temp_reference_point, data)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_reference_point = temp_reference_point[:, self.num_propagated:]
            memory_mask = memory_mask[:, self.num_propagated:]

        return query_feat, reference_points, temp_memory, temp_reference_point, memory_mask, rec_ego_pose

    def forward(self, img_metas,  **data):
        if self.temp_fusion:
            self.pre_update_memory(data)

        mlvl_feats = data['img_feats']
        B, Q, = mlvl_feats[0].shape[0], self.num_query
        # (N_query, 3) --> (B, N_query, 3)
        init_points = self.init_points.weight[None, :, :].repeat(B, 1, 1)
        # (B, N_query, C)
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)

        if self.temp_fusion:
            query_feat, reference_points, temp_memory, temp_reference_point, memory_mask, rec_ego_pose = \
                self.temporal_alignment(query_feat, init_points, data)
        else:
            reference_points = init_points
            temp_memory = None
            temp_reference_point = None
            memory_mask = None

        # query_feats: List[(B, N_query, C), (B, N_query, C), ...]
        # cls_scores: List[(B, N_query, n_refine_1, n_cls), (B, N_query, n_refine_2, n_cls), ...]
        # refine_sqs: List[(B, N_query, n_refine_1, 13), (B, N_query, n_refine_2, 13), ...]
        query_feats, cls_scores, refine_sqs = self.transformer(
            query_feat,  # (B, N_query, C)
            reference_points.unsqueeze(dim=2),  # (B, N_query, 1, 3)
            temp_memory,  # (B, Mem, C)
            temp_reference_point,  # (B, N_query, 3)
            memory_mask,
            mlvl_feats,  # List[(B, N, C=256, H2, W2), ..., (B, N, C=256, H5, W5)]
            data,
            img_metas=img_metas,
        )

        cls_scores_list = []
        pred_occ_list = []
        for i in range(len(refine_sqs)):
            if not self.training:
                if i < (len(refine_sqs) - 1):
                    continue

           
            sq_mean = decode_points(refine_sqs[i][..., 0:3], self.pc_range)  # (B, Q，N_refine, 3)
            sq_scales = safe_sigmoid(refine_sqs[i][..., 3:6])  # (B, Q，N_refine, 3)
            sq_scales = self.scale_range[0] + (
                    self.scale_range[1] - self.scale_range[0]) * sq_scales  # (B, Q，N_refine, 3)
            rot = refine_sqs[i][..., 6:10]  # (B, Q，N_refine, 4)
            opa = safe_sigmoid(refine_sqs[i][..., 10:11])  # (B, Q，N_refine, 1)
            uv = safe_sigmoid(refine_sqs[i][..., 11:13])  # (B, Q，N_refine, 2)
            u = self.u_range[0] + (self.u_range[1] - self.u_range[0]) * uv[..., :1]  # (B, Q，N_refine, 1)
            v = self.v_range[0] + (self.v_range[1] - self.v_range[0]) * uv[..., 1:]  # (B, Q，N_refine, 1)
            sqs = torch.cat([sq_mean, sq_scales, rot, opa, u, v], dim=-1)

           
            voxel_logits, voxel_coors = self.densifiers[i](
                cls_scores[i],                    # (B, Q, R, 32) - per-point feature
                refine_sqs[i][..., :3]            # (B, Q, R, 3)  - encoded xyz only
            )             
            
            cls_score = self.scatter_voxel_to_points(
                self.densifiers[i], voxel_logits, voxel_coors, refine_sqs[i][..., :3]
            )   # (B, Q, R, num_classes)

            cls_scores_list.append(cls_score)

            occ_pred = self.sq2occ(cls_score, sqs)
            pred_occ_list.append(occ_pred)

            if DUMP.enabled and i == (len(refine_sqs) - 1):
                meta = img_metas[0]
                scene_name = meta.get('scene_name', meta.get('scene_token', 'scene_unknown'))
                sample_idx = meta.get('sample_idx', meta.get('token'))
                if sample_idx is None:
                    filenames = meta.get('filename')
                    if isinstance(filenames, (list, tuple)) and len(filenames) > 0:
                        sample_idx = os.path.splitext(os.path.basename(filenames[0]))[0]
                if sample_idx is None:
                    sample_idx = f"frame_{DUMP.frame_count:06d}"
                DUMP.frame_count += 1

                save_path = '{}/{}/{}'.format(DUMP.out_dir, scene_name, sample_idx)
                if not os.path.exists(save_path):
                    os.makedirs(save_path, exist_ok=True)

                import numpy as np
                np.savez_compressed(f'{save_path}/pred_superquadrics.npz', sqs=sqs.detach().cpu().numpy())
                np.savez_compressed(f'{save_path}/pred_semantics.npz', cls_score=cls_score.detach().cpu().numpy())
                np.savez_compressed(f'{save_path}/pred_occupancy.npz', occ_pred=occ_pred.detach().cpu().numpy().astype(np.uint8))

        if self.temp_fusion:
            self.post_update_memory(data, rec_ego_pose, query_feats, cls_scores_list, refine_sqs)

        return dict(init_points=init_points,
                    all_cls_scores=cls_scores_list,
                    all_refine_sqs=refine_sqs,
                    all_pred_occ_list=pred_occ_list)

    def sq2occ(self, cls_scores, refine_sqs):
        """

        :param cls_scores: (B, N_query, n_refine, n_cls)
        :param refine_sqs: (B, N_query, n_refine, 13)
        :return:
        """
        num_imgs = cls_scores.size(0)
        cls_scores = cls_scores.flatten(1, 2)  # (B, Q*N_refine, 17)
        refine_sqs = refine_sqs.flatten(1, 2)  # (B, Q*N_refine, 13)

        gs_mean = refine_sqs[..., :3]  # (B, Q*N_refine, 3)
        scales = refine_sqs[..., 3:6]  # (B, Q*N_refine, 3)
        rot = refine_sqs[..., 6:10]  # (B, Q*N_refine, 4)
        origi_opa = refine_sqs[..., 10:11]  # (B, Q*N_refine, 1)
        u = refine_sqs[..., 11:12]  # (B, Q*N_refine, 1)
        v = refine_sqs[..., 12:13]  # (B, Q*N_refine, 1)

        rots = get_rotation_matrix(rot)  # (B, Q*N_refine, 4, 4)
        origi_opa = origi_opa.flatten(1, 2)  # (B, Q*N_refine)
        u = u.flatten(1, 2)  # (B, Q*N_refine)
        v = v.flatten(1, 2)  # (B, Q*N_refine)

        opacities = cls_scores.softmax(dim=-1)  # (B, Q*N_refine, sem_dim=17)
        # opacities = torch.sigmoid(cls_scores)  # (B, Q*N_refine, sem_dim=17)
        opacities = torch.cat([opacities, torch.zeros_like(opacities[..., :1])],
                              dim=-1)  # (B, Q*N_refine, sem_dim=18),  18:(cls0, cls1, ..., free)

        gt_xyz = self.gt_xyz[None, ...].repeat([num_imgs, 1, 1, 1, 1])
        sampled_xyz = gt_xyz.flatten(1, 3).float()  # (B, Dx*Dy*Dz, 3)

        import time
        semantics = []
        for i in range(num_imgs):
            # logits: (Dx*Dy*Dz, sem_dim=18)
            # bin_logits: (Dx*Dy*Dz, )
            # density: (Dx*Dy*Dz, )
            semantic = self.aggregator(
                sampled_xyz[i:(i + 1)],
                gs_mean[i:(i + 1)],
                origi_opa[i:(i + 1)],
                u[i:(i + 1)],
                v[i:(i + 1)],
                opacities[i:(i + 1)],
                scales[i:(i + 1)],
                rots[i:(i + 1)])

            sem = semantic[0][:, :-1] * semantic[1].unsqueeze(-1)  # (Dx*Dy*Dz, 17)
            geo = 1 - semantic[1].unsqueeze(-1)  # (Dx*Dy*Dz, 1),  empty_prop
            geosem = torch.cat([sem, geo], dim=-1)  # (Dx*Dy*Dz, 18), 18: (cls0, cls1, ..., empty)
            geosem = geosem.reshape(self.voxel_num[0], self.voxel_num[1], self.voxel_num[2], -1)

            semantics.append(geosem)
        occ_pred = torch.stack(semantics, dim=0)  # (B, Dx*Dy*Dz, 18)

        return occ_pred

    def get_geo_hardness(self, target, alpha=1.0, beta=1.0):
        add_batch = False
        if target.dim() == 3:
            target = target.unsqueeze(0)
            add_batch = True

        
        valid = target != self.ignore_label

        
        occ = ((target != self.empty_label) & (target != self.ignore_label)).float()

        geo = torch.zeros_like(occ, dtype=torch.float32)

        
        diff = (occ[:, 1:, :, :] != occ[:, :-1, :, :]).float()
        valid_pair = valid[:, 1:, :, :] & valid[:, :-1, :, :]
        diff = diff * valid_pair.float()
        geo[:, 1:, :, :] += diff
        geo[:, :-1, :, :] += diff

        
        diff = (occ[:, :, 1:, :] != occ[:, :, :-1, :]).float()
        valid_pair = valid[:, :, 1:, :] & valid[:, :, :-1, :]
        diff = diff * valid_pair.float()
        geo[:, :, 1:, :] += diff
        geo[:, :, :-1, :] += diff

        
        diff = (occ[:, :, :, 1:] != occ[:, :, :, :-1]).float()
        valid_pair = valid[:, :, :, 1:] & valid[:, :, :, :-1]
        diff = diff * valid_pair.float()
        geo[:, :, :, 1:] += diff
        geo[:, :, :, :-1] += diff

        
        geo = geo / 6.0

        
        H_geo = alpha + beta * geo

        
        H_geo = H_geo * valid.float()

        
        if valid.any():
            H_geo = H_geo / (H_geo[valid].mean() + 1e-6)

        if add_batch:
            H_geo = H_geo.squeeze(0)

        return H_geo

    def loss_single(self,
                    pred_occ,
                    voxel_semantics,
                    geo_weight=None,
                    ):
        """
        Args:
            cls_scores: (B, Dx, Dy, Dz, n_cls)
            voxel_semantics: (B, Dx=200, Dy=200, Dz=16)
        """
        voxel_semantics = voxel_semantics.long()  # (B, Dx, Dy, Dz)
        preds = pred_occ.permute(0, 4, 1, 2, 3).contiguous()  # (B, n_cls, Dx, Dy, Dz)
        preds = torch.clamp(preds, 1e-6, 1. - 1e-6)

        num_total_samples = 0
        for i in range(self.num_classes + 1):
            if i == self.ignore_label:  
                continue
            num_total_samples += (voxel_semantics == i).sum() * self.cls_weights[i]
        loss_occ = self.loss_occ(
            preds,
            voxel_semantics,
            weight=geo_weight,
            avg_factor=num_total_samples
        )

        loss_voxel_lovasz = lovasz_softmax(preds, voxel_semantics, ignore=self.ignore_label)

        return loss_occ, loss_voxel_lovasz

    def loss_pts_single(self,
                        reference_points,
                        gt_points_list,
                        ):
        """
        Args:
            reference_points: (B, N_q, 3)
            gt_points_list: List[(N_occ0, 3), (N_occ1, 3), ...]
        """
        num_imgs = reference_points.size(0)
        reference_points = reference_points.reshape(num_imgs, -1, 3).contiguous()     # (B, N_q, 3)
        reference_points_list = [reference_points[i] for i in range(num_imgs)]   # List[(Q, 3), (Q, 3), ...]

        (gt_paired_pts, pred_paired_pts) = multi_apply(
            self._get_paired_pts, reference_points_list, gt_points_list)

        gt_pts = torch.cat(gt_points_list)  # (N_gt=N_occ0+N_occ1+..., 3)
        gt_paired_pts = torch.cat(gt_paired_pts)  # (N_gt=N_occ0+N_occ1+..., )
        pred_pts = torch.cat(reference_points_list)  # (N_pred=B*Q, 3)
        pred_paired_pts = torch.cat(pred_paired_pts)  # (N_pred=B*Q, 3)
        # calculate loss pts
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  avg_factor=gt_pts.shape[0]
                                  )

        loss_pts += self.loss_pts(pred_pts,
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])
        return (loss_pts, )

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        """
        Args:
            voxel_semantics: (B, Dx=200, Dy=200, Dz=16)
            mask_camera:
            preds_dicts: dict{
                'init_points': (B, N_query, 1, 3),
                'cls_scores': List[(B, N_query, n_cls), (B, N_query, n_cls), ...]
                'refine_sqs': List[(B, N_query, n_refine_1, 13), (B, N_query, n_refine_2, 13), ...]
                'all_pred_occ_list': List[(B, Dx, Dy, Dz, 18), (B, Dx, Dy, Dz, 18), ...]
            }
        """
        all_pred_occ_list = preds_dicts['all_pred_occ_list']  # List[(B, Dx, Dy, Dz, 18), (B, Dx, Dy, Dz, 18), ...]

        num_dec_layers = len(all_pred_occ_list)
        voxel_semantics_list = [voxel_semantics for _ in range(num_dec_layers)]

        if self.use_geo_hardness:
            geo_weight = self.get_geo_hardness(
                voxel_semantics,
                alpha=self.geo_hardness_alpha,
                beta=self.geo_hardness_beta,
            )  # (B, Dx, Dy, Dz)
        else:
            geo_weight = None
        geo_weight_list = [geo_weight for _ in range(num_dec_layers)]

        losses_occ, losses_voxel_lovasz = multi_apply(
            self.loss_single,
            all_pred_occ_list,  # List[(B, Dx, Dy, Dz, 18), (B, Dx, Dy, Dz, 18), ...]
            voxel_semantics_list,  # List[(B, Dx=200, Dy=200, Dz=16), (B, Dx=200, Dy=200, Dz=16), ...]
            geo_weight_list,
        )

        loss_dict = dict()

        loss_dict['loss_occ'] = losses_occ[-1]
        loss_dict['loss_voxel_lovasz'] = losses_voxel_lovasz[-1]


        num_dec_layer = 0
        for loss_occ_i, loss_voxel_lovasz_i in zip(losses_occ[:-1], losses_voxel_lovasz[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_occ'] = loss_occ_i
            loss_dict[f'd{num_dec_layer}.loss_voxel_lovasz'] = loss_voxel_lovasz_i
            num_dec_layer += 1

        gt_points_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics)
        init_points = preds_dicts['init_points']  # (B, Q, 3)
        init_points = decode_points(init_points, self.pc_range)  # (B, Q, 3)
        init_loss_pts = self.loss_pts_single(init_points, gt_points_list)[0]
        loss_dict['init_loss_pts'] = init_loss_pts

        return loss_dict

    def _get_paired_pts(self, pts, gt_points):
        gt_paired_idx = knn(1, pts[None, ...], gt_points[None, ...])  # (1, 1, N_gt)
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()  # (N_gt, )
        pred_paired_idx = knn(1, gt_points[None, ...], pts[None, ...])  # (1, 1, N_pred)
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()  # (N_pred, )
        gt_paired_pts = pts[gt_paired_idx]  # (N_gt, 3)
        pred_paired_pts = gt_points[pred_paired_idx]  # (N_pred, 3)

        return gt_paired_pts, pred_paired_pts

    def get_sparse_voxels(self, voxel_semantics):
        """
        Args:
            voxel_semantics: (B, Dx, Dy, Dz)
        Returns:
            gt_points: List[(N_occ0, 3), (N_occ1, 3), ...]
            gt_labels: List[(N_occ0, ), (N_occ1, ), ...]
        """
        coors = self.gt_xyz
        voxel_semantics = voxel_semantics.long()

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(len(voxel_semantics)):
            mask = (voxel_semantics[i] != self.empty_label) & (voxel_semantics[i] != self.ignore_label)  # (Dx, Dy, Dz)
            gt_points.append(coors[mask])  # (N_occ, 3)
            gt_labels.append(voxel_semantics[i][mask])  # (N_occ, )

        return gt_points, gt_labels

    def get_occ(self, pred_dicts, img_metas, rescale=False):
        occ_pred = pred_dicts['all_pred_occ_list'][-1]

        if self.score_thres is None:
            occ_res = occ_pred.argmax(-1).int()      # (B, Dx, Dy, Dz)
        else:
            fg_prob = occ_pred[..., :-1]  # (B, Dx, Dy, Dz, 17)
            score, occ_res = fg_prob.max(dim=-1)  # (B, Dx, Dy, Dz)
            score_mask = score < self.score_thres
            occ_res = occ_res.int()
            occ_res[score_mask] = 17

        return list(occ_res)

    def scatter_voxel_to_points(self, densifier, voxel_logits, voxel_coors, refine_pts_encoded):
        B, Q, R, _ = refine_pts_encoded.shape
        device = refine_pts_encoded.device
        
        if voxel_logits is None or voxel_coors is None or voxel_logits.numel() == 0:
            num_classes = getattr(self, 'num_classes', voxel_logits.shape[-1] if voxel_logits is not None else 0)
            return refine_pts_encoded.new_zeros(B, Q, R, num_classes)
        num_classes = voxel_logits.shape[-1]
        
        voxel_num = densifier.voxel_num.to(refine_pts_encoded.dtype)
        loc = torch.floor(refine_pts_encoded * voxel_num).long()
        loc[..., 0].clamp_(0, int(densifier.voxel_num[0]) - 1)
        loc[..., 1].clamp_(0, int(densifier.voxel_num[1]) - 1)
        loc[..., 2].clamp_(0, int(densifier.voxel_num[2]) - 1)
        batch_idx = torch.arange(B, device=device).view(B, 1, 1, 1).expand(B, Q, R, 1)
        pts_coor = torch.cat([batch_idx, loc], dim=-1)  # (B, Q, R, 4)
       
        scale_bxyz = int(densifier.scale_xyz)
        scale_yz = int(densifier.scale_yz)
        scale_z = int(densifier.scale_z)
        pts_hash = (pts_coor[..., 0] * scale_bxyz +
                    pts_coor[..., 1] * scale_yz +
                    pts_coor[..., 2] * scale_z +
                    pts_coor[..., 3]).reshape(-1)       # (B*Q*R,)
        vox_hash = (voxel_coors[:, 0] * scale_bxyz +
                    voxel_coors[:, 1] * scale_yz +
                    voxel_coors[:, 2] * scale_z +
                    voxel_coors[:, 3])                  # (N_out,)
  
        sorted_vox, sort_idx = torch.sort(vox_hash)
        search_idx = torch.searchsorted(sorted_vox, pts_hash)
        search_idx = search_idx.clamp(max=sorted_vox.numel() - 1)

        matched = sorted_vox[search_idx] == pts_hash     # (B*Q*R,)
        real_idx = sort_idx[search_idx]                  # (B*Q*R,)

        point_logits = voxel_logits[real_idx]            # (B*Q*R, num_classes)
        point_logits = point_logits * matched[:, None].to(point_logits.dtype)
        point_logits = point_logits.reshape(B, Q, R, num_classes)
        return point_logits

class MLN(nn.Module):
    '''
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out
