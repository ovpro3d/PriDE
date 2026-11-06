# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES
from mmdet.models.losses.utils import weight_reduce_loss

@LOSSES.register_module()
class CELoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 class_weight=None,
                 activated=True,
                 ignore_label=255,
                 reduction='mean',
                 **kwargs):
        super(CELoss, self).__init__()

        self.activated = activated
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.reduction = reduction
        self.class_weight = class_weight

    def forward(self, ce_input, ce_label, weight=None, avg_factor=None):
        ce_input = ce_input.float()
        ce_label = ce_label.long()

        if not self.activated:
            ce_loss = F.cross_entropy(ce_input, ce_label, weight=self.class_weight.to(ce_input),
                                      ignore_index=self.ignore_label, reduction='none')
        else:
            ce_loss = F.nll_loss(torch.log(ce_input), ce_label, weight=self.class_weight.to(ce_input),
                                 ignore_index=self.ignore_label, reduction='none')

        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        ce_loss = weight_reduce_loss(
            ce_loss, weight=weight, reduction=self.reduction, avg_factor=avg_factor) * self.loss_weight

        return ce_loss


@LOSSES.register_module()
class FocalCELoss(nn.Module):
    def __init__(self,
                 loss_weight=1.0,
                 class_weight=None,
                 activated=True,
                 ignore_label=255,
                 gamma=2.0,
                 reduction='mean',
                 **kwargs):
        super(FocalCELoss, self).__init__()

        self.activated = activated
        self.loss_weight = loss_weight
        self.ignore_label = ignore_label
        self.reduction = reduction
        self.class_weight = class_weight
        self.gamma = gamma

    def forward(self, ce_input, ce_label, weight=None, avg_factor=None):
        ce_input = ce_input.float()
        ce_label = ce_label.long()

        if not self.activated:
            ce_loss = F.cross_entropy(ce_input, ce_label, ignore_index=self.ignore_label, reduction='none')
        else:
            ce_loss = F.nll_loss(torch.log(ce_input), ce_label, ignore_index=self.ignore_label, reduction='none')

        pt = torch.exp(-ce_loss)

        if self.class_weight is not None:
            class_weight = self.class_weight.to(ce_input)
            alpha_t = class_weight[ce_label]
        else:
            alpha_t = 1.0
        loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        # apply weights and do the reduction
        if weight is not None:
            weight = weight.float()
        loss = weight_reduce_loss(
            loss, weight=weight, reduction=self.reduction, avg_factor=avg_factor) * self.loss_weight

        return loss
