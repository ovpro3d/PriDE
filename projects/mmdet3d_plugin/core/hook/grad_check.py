# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner.hooks import HOOKS, Hook

@HOOKS.register_module()
class CheckGradHook(Hook):
    def __init__(self, interval=1):
        super(CheckGradHook, self).__init__()
        self.interval = interval

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, self.interval):
            model = runner.model
            exploding_params = []
            for name, param in model.named_parameters():
                if isinstance(param, torch.Tensor) and param.grad is not None:
                    norm = param.grad.data.norm(2)
                    if not torch.isfinite(norm):
                        exploding_params.append((name, norm.item()))

            if exploding_params:
                runner.logger.warning('⚠️ Exploding gradients detected:')
                for name, norm in exploding_params:
                    runner.logger.warning(f' - {name}: grad_norm = {norm:.4e}')