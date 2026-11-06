import torch
from mmcv.runner import force_fp32
from mmdet.models import HEADS

from projects.mmdet3d_plugin.ops.tile_localagg_prob_sq_bt2.tile_local_aggregate_prob_sq_bt2 import (
    LocalAggregatorExtV2, local_aggregate_backtrace, local_aggregate_backtrace_own)
from .streamocc_head_v9 import StreamOccHeadV9


@HEADS.register_module()
class StreamOccHeadV10(StreamOccHeadV9):
    def __init__(self, *args,
                 teacher_mode='own_cuda',
                 teacher_eps1=1e-6,
                 teacher_eps2=1e-6,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert teacher_mode in ('own_cuda', 'shared_cuda')
        self.teacher_mode = teacher_mode
        self.teacher_eps1 = float(teacher_eps1)
        self.teacher_eps2 = float(teacher_eps2)

        self.aggregator = LocalAggregatorExtV2(
            scale_multiplier=3,
            H=self.voxel_num[0],
            W=self.voxel_num[1],
            D=self.voxel_num[2],
            pc_min=self.pc_range[:3],
            grid_size=self.voxel_size[0],
        )

    def _primitive_difficulty(self, occ_pred, voxel_semantics, bt_caches,
                              h_loc=None, mask_camera=None):
        if self.teacher_mode == 'shared_cuda':
            return super()._primitive_difficulty(
                occ_pred, voxel_semantics, bt_caches, h_loc=h_loc, mask_camera=mask_camera)

        B = occ_pred.size(0)
        es, dens = [], []
        for i in range(B):
            gt_flat = voxel_semantics[i].reshape(-1).long()          # (N,) same order as cache/pts
            valid = (gt_flat != self.ignore_label) & (gt_flat != self.empty_label)
            if mask_camera is not None:
                valid = valid & (mask_camera[i].reshape(-1) > 0)

            wvox = None
            if h_loc is not None and self.lam_struct > 0.0:
                wvox = 1.0 + self.lam_struct * h_loc[i].reshape(-1).float()

            num, den = local_aggregate_backtrace_own(
                bt_caches[i], gt_flat, valid.float(), wvox=wvox, eps1=self.teacher_eps1)
            es.append(num / (den + self.teacher_eps2))
            dens.append(den)
        return torch.stack(es, 0), torch.stack(dens, 0)   # (B, P=Q*N_refine) x2
