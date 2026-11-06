# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.runner.hooks import HOOKS, Hook
from .utils import is_parallel


@HOOKS.register_module()
class PropControlHook(Hook):
    """ """

    def __init__(self, temporal_start_iter=-1):
        super().__init__()
        self.prop_start_iter = temporal_start_iter
        self.prop = False

    def set_prop_flag(self, runner, flag):
        if is_parallel(runner.model.module):
            runner.model.module.module.pts_bbox_head.prop_query = flag
        else:
            runner.model.module.pts_bbox_head.prop_query = flag

    def before_run(self, runner):
        self.set_prop_flag(runner, False)

    def after_train_iter(self, runner):
        if runner.iter >= self.prop_start_iter and not self.prop:
            self.set_prop_flag(runner, True)