import torch
import torch.nn.functional as F
from mmcv.runner import force_fp32
from mmdet.models import HEADS

from projects.mmdet3d_plugin.models.utils.difficulty_scorer_v3 import (
    paper_logistic_ranking_loss, spearman_corr)
from .streamocc_head import StreamOccHead
from .streamocc_head_v7 import StreamOccHeadV7


@HEADS.register_module()
class StreamOccHeadV9(StreamOccHeadV7):
    def __init__(self, *args, score_loss_type='paper_logistic',
                 score_gamma=1.0, score_pair_mode='sampled',
                 score_num_pairs=4096, score_exact_chunk=1024, **kwargs):
        super().__init__(*args, **kwargs)
        assert score_loss_type in ('paper_logistic', 'legacy_margin')
        assert score_pair_mode in ('sampled', 'exact')
        self.score_loss_type = score_loss_type
        self.score_gamma = float(score_gamma)
        self.score_pair_mode = score_pair_mode
        self.score_num_pairs = int(score_num_pairs)
        self.score_exact_chunk = int(score_exact_chunk)

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
            preds = super().forward(img_metas, **data)
        finally:
            self.transformer.forward = orig_forward

        if logits_box.get('all_s_logits') is not None:
            preds['all_s_logits'] = logits_box['all_s_logits']
        return preds

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self, voxel_semantics, mask_camera, preds_dicts):
        if self.score_loss_type == 'legacy_margin':
            return super().loss(voxel_semantics, mask_camera, preds_dicts)

        loss_dict = StreamOccHead.loss(self, voxel_semantics, mask_camera, preds_dicts)
        all_s = preds_dicts.get('all_s_scores', None)
        all_u = preds_dicts.get('all_s_logits', None)

        if not self.use_difficulty:
            zero = voxel_semantics.new_zeros(()).float()
            if all_s is not None:
                zero = zero + sum(s.sum() for s in all_s).float() * 0.0
            if all_u is not None:
                zero = zero + sum(u.sum() for u in all_u).float() * 0.0
            loss_dict['loss_score'] = zero
            return loss_dict

        if all_u is None:
            raise KeyError(
                "StreamOccHeadV9 paper_logistic requires preds_dicts['all_s_logits']; "
                "use StreamOccTransformerV6.")

        occ_list = preds_dicts['all_pred_occ_list']
        lay_caches = preds_dicts['layer_bt_caches']

        h_loc = None
        if self.lam_struct > 0.0:
            from projects.mmdet3d_plugin.models.utils.difficulty_scorer_v3 import (
                voxel_structural_difficulty)
            h_loc = voxel_structural_difficulty(voxel_semantics.long(), self.ignore_label)
        cam = mask_camera if (self.teacher_use_camera_mask and mask_camera is not None) else None

        loss_score = voxel_semantics.new_zeros(()).float()
        diag_corr = voxel_semantics.new_zeros(()).float()
        diag_den_valid = voxel_semantics.new_zeros(()).float()
        diag_den_mean = voxel_semantics.new_zeros(()).float()
        diag_den_max = voxel_semantics.new_zeros(()).float()
        L = len(occ_list)
        for li in range(L):
            e, den = self._primitive_difficulty(
                occ_list[li], voxel_semantics, lay_caches[li], h_loc=h_loc, mask_camera=cam)
            u = all_u[li].reshape(e.size(0), -1).float()
            vmask = (den > self.den_thresh).float()
            den_f = den.float()
            valid_ratio = vmask.mean()
            diag_den_valid = diag_den_valid + valid_ratio
            diag_den_mean = diag_den_mean + den_f.mean()
            diag_den_max = diag_den_max + den_f.max()
            loss_dict[f'diag_den_valid_l{li}'] = valid_ratio.detach()
            loss_score = loss_score + paper_logistic_ranking_loss(
                u, e, vmask, gamma=self.score_gamma,
                pair_mode=self.score_pair_mode,
                num_pairs=self.score_num_pairs,
                exact_chunk=self.score_exact_chunk)
            if all_s is not None:
                s = all_s[li].reshape(e.size(0), -1).float()
                diag_corr = diag_corr + spearman_corr(s.detach(), e, vmask)

        loss_dict['loss_score'] = self.lambda_score * loss_score / max(1, L)
        loss_dict['diag_spearman'] = (diag_corr / max(1, L)).detach()
        loss_dict['diag_den_valid'] = (diag_den_valid / max(1, L)).detach()
        loss_dict['diag_den_mean'] = (diag_den_mean / max(1, L)).detach()
        loss_dict['diag_den_max'] = (diag_den_max / max(1, L)).detach()

        decoder = getattr(self.transformer, 'decoder', None)
        adapters = getattr(decoder, 'adapters', None)
        if adapters is not None and len(adapters) > 0:
            gates = torch.stack([torch.tanh(a.gate.detach()).reshape(()) for a in adapters])
            loss_dict['diag_gate'] = gates.abs().mean()
        mod = getattr(self.transformer, 'zs_mod', None)
        if mod is not None:
            loss_dict['diag_kappa_sp'] = F.softplus(mod.kappa.detach()).reshape(())
            loss_dict['diag_beta_sp'] = F.softplus(mod.beta.detach()).reshape(())
            loss_dict['diag_beta_s_sp'] = F.softplus(mod.beta_s.detach()).reshape(())
            loss_dict['diag_b_lvl_norm'] = mod.b_lvl.detach().norm()
        return loss_dict
