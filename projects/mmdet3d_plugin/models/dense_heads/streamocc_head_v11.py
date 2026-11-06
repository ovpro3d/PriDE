# StreamOccHeadV11: V10 without the densifier -- the transformer's per-point
# outputs are already num_classes-channel logits and feed sq2occ directly.

import os
import numpy as np
import torch
from torch import nn

from mmdet.models import HEADS
from projects.mmdet3d_plugin.core.bbox.util import decode_points
from projects.mmdet3d_plugin.models.utils.misc import safe_sigmoid, DUMP
from .streamocc_head_v10 import StreamOccHeadV10


@HEADS.register_module()
class StreamOccHeadV11(StreamOccHeadV10):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.densifiers = nn.ModuleList()

    def forward(self, img_metas, **data):
        logits_box = {}
        orig_forward = self.transformer.forward

        def forward4(*args, **kwargs):
            out = orig_forward(*args, **kwargs)
            if isinstance(out, tuple) and len(out) == 5:
                query_feats, cls_scores, refine_sqs, s_scores, s_logits = out
                logits_box['all_s_logits'] = s_logits
                return query_feats, cls_scores, refine_sqs, s_scores
            logits_box['all_s_logits'] = None
            return out

        self.transformer.forward = forward4
        try:
            preds = self._forward_no_densifier(img_metas, **data)
        finally:
            self.transformer.forward = orig_forward

        if logits_box.get('all_s_logits') is not None:
            preds['all_s_logits'] = logits_box['all_s_logits']
        return preds

    def _forward_no_densifier(self, img_metas, **data):
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

            cls_score = cls_scores[i]
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
