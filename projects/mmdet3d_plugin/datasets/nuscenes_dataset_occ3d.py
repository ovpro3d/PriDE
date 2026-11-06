import mmcv
import os
import os.path as osp
from tqdm import tqdm
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.datasets import DATASETS
import torch
import numpy as np
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
import math
from torch.utils.data import DataLoader
from projects.mmdet3d_plugin.core.evaluation.occ_metrics import Metric_mIoU


@DATASETS.register_module()
class NuScenesDatasetOcc3D(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def __init__(self, seq_mode=False, seq_split_num=1, adj_list=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data_infos = self.load_annotations(self.ann_file)
        self.adj_list = adj_list

        if seq_mode:
            self.seq_split_num = seq_split_num
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.

    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
        if self.seq_split_num == -1:
            self.flag = np.arange(len(self.data_infos))
            return

        res = []

        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and len(self.data_infos[idx]['cam_sweeps']) == 0:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.seq_split_num != 1:
            if self.seq_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    curr_sequence_length = np.array(
                        list(range(0, 
                                bin_counts[curr_flag], 
                                math.ceil(bin_counts[curr_flag] / self.seq_split_num)))
                        + [bin_counts[curr_flag]])

                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                assert len(np.bincount(new_flags)) == len(np.bincount(self.flag)) * self.seq_split_num
                self.flag = np.array(new_flags, dtype=np.int64)

    def collect_cam_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1

        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['cam_sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def prepare_train_data(self, index):
        """Training data preparation.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        """Prepare data for testing.

        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def get_data_info(self, index):
        info = self.data_infos[index]

        e2g_rotation = Quaternion(info['ego2global_rotation']).rotation_matrix
        e2g_translation = info['ego2global_translation']
        l2e_rotation = Quaternion(info['lidar2ego_rotation']).rotation_matrix
        l2e_translation = info['lidar2ego_translation']
        ego2global = convert_egopose_to_matrix_numpy(e2g_rotation, e2g_translation)
        lidar2ego = convert_egopose_to_matrix_numpy(l2e_rotation, l2e_translation)
        lidar2global = ego2global @ lidar2ego  # lidar2global

        ego2lidar = invert_matrix_egopose_numpy(lidar2ego)
        global2ego = invert_matrix_egopose_numpy(ego2global)
        global2lidar = invert_matrix_egopose_numpy(lidar2global)
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            scene_name=info['scene_name'],
            scene_token=info['scene_token'],
            timestamp=info['timestamp'] / 1e6,
            ego_pose=ego2global,
            ego_pose_inv=global2ego,
            lidar2global=lidar2global,
            global2lidar=global2lidar,
            ego2lidar=ego2lidar,
        )

        # if self.modality['use_lidar']:
        #     lidar_sweeps_prev, lidar_sweeps_next = self.collect_lidar_sweeps(index)
        #     input_dict.update(dict(
        #         pts_filename=info['lidar_path'],
        #         lidar_sweeps={'prev': lidar_sweeps_prev, 'next': lidar_sweeps_next},
        #     ))

        input_dict['occ_gt_path'] = info['occ_path']

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []
            intrinsics = []
            extrinsics = []
            cam_names = []

            for _, cam_info in info['cams'].items():
                cam_names.append(cam_info['type'])
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                cam2lidar_r = cam_info['sensor2lidar_rotation']
                cam2lidar_t = cam_info['sensor2lidar_translation']
                cam2lidar_rt = convert_egopose_to_matrix_numpy(cam2lidar_r, cam2lidar_t)
                lidar2cam_rt = invert_matrix_egopose_numpy(cam2lidar_rt)

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt)

                intrinsics.append(intrinsic)
                extrinsics.append(lidar2cam_rt)
                lidar2img_rts.append(lidar2img_rt)

            cam_sweeps_prev, cam_sweeps_next = self.collect_cam_sweeps(index)

            input_dict.update(dict(
                cam_names=cam_names,
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                cam_sweeps={'prev': cam_sweeps_prev, 'next': cam_sweeps_next},
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:

            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        results_dict = {}
        results_dict.update(
            self.eval_miou(occ_results, runner=runner, show_dir=show_dir, **eval_kwargs))
        results_dict.update(
            self.eval_binary_miou(occ_results, runner=runner, show_dir=show_dir, **eval_kwargs))
        results_dict.update(
            self.eval_riou(occ_results, runner=runner, show_dir=show_dir, **eval_kwargs))
        print(results_dict)
        return results_dict

    def eval_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        print('\nStarting Evaluation...')
        metric = Metric_mIoU(use_image_mask=True)

        from tqdm import tqdm
        for i in tqdm(range(len(occ_results))):
            info = self.data_infos[i]
            occ_file = info['occ_path']
            occ_infos = np.load(occ_file)

            occ_labels = occ_infos['semantics']
            mask_lidar = occ_infos['mask_lidar'].astype(np.bool_)
            mask_camera = occ_infos['mask_camera'].astype(np.bool_)

            occ_pred = occ_results[i].cpu().numpy()

            metric.add_batch(occ_pred, occ_labels, mask_lidar, mask_camera)

        return {'mIoU': metric.count_miou()}

    def eval_binary_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        print('\nStarting Evaluation...')
        metric = Metric_mIoU(use_image_mask=True, num_classes=2)

        from tqdm import tqdm
        for i in tqdm(range(len(occ_results))):
            info = self.data_infos[i]
            occ_file = info['occ_path']
            occ_infos = np.load(occ_file)

            occ_labels = occ_infos['semantics']
            mask_lidar = occ_infos['mask_lidar'].astype(np.bool_)
            mask_camera = occ_infos['mask_camera'].astype(np.bool_)

            occ_pred = occ_results[i].cpu().numpy()

            metric.add_batch(occ_pred, occ_labels, mask_lidar, mask_camera)

        return {'binary_mIoU': metric.count_miou()}

    def eval_riou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        occ_preds = []
        lidar_origins = []

        print('\nStarting Evaluation...')

        from projects.mmdet3d_plugin.core.evaluation.ray_metrics import main as calc_rayiou
        from .ego_pose_dataset import EgoPoseDataset

        data_loader = DataLoader(
            EgoPoseDataset(self.data_infos),
            batch_size=1,
            shuffle=False,
            num_workers=8
        )

        sample_tokens = [info['token'] for info in self.data_infos]

        for i, batch in enumerate(data_loader):
            token = batch[0][0]
            output_origin = batch[1]

            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            occ_file = info['occ_path']
            occ_infos = np.load(occ_file)
            gt_semantics = occ_infos['semantics']

            occ_pred = occ_results[data_id].cpu().numpy()
            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(occ_pred)

        return calc_rayiou(occ_preds, occ_gts, lidar_origins)

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            scene_name = info['scene_name']
            sample_idx = info['token']
            save_path = '{}/{}/{}'.format(submission_prefix, scene_name, sample_idx)
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)
            np.savez_compressed(f'{save_path}/pred_occupancy.npz', occ_pred=occ_pred.detach().cpu().numpy().astype(np.uint8))
        print('\nFinished.')


def invert_matrix_egopose_numpy(egopose):
    """ Compute the inverse transformation of a 4x4 egopose numpy matrix."""
    inverse_matrix = np.zeros((4, 4), dtype=np.float32)
    rotation = egopose[:3, :3]
    translation = egopose[:3, 3]
    inverse_matrix[:3, :3] = rotation.T
    inverse_matrix[:3, 3] = -np.dot(rotation.T, translation)
    inverse_matrix[3, 3] = 1.0
    return inverse_matrix

def convert_egopose_to_matrix_numpy(rotation, translation):
    transformation_matrix = np.zeros((4, 4), dtype=np.float32)
    transformation_matrix[:3, :3] = rotation
    transformation_matrix[:3, 3] = translation
    transformation_matrix[3, 3] = 1.0
    return transformation_matrix