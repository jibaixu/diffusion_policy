import numpy as np
import hydra
from typing import Optional
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer


class AllTasksMixedRobotDataset(BaseImageDataset):
    def __init__(
            self,
            dataset_cfg: Optional[dict] = None,
            datasets: Optional[list[BaseImageDataset]] = None,
            ):
        super().__init__()

        # 保存 datasets 或重新构造
        if datasets is not None:
            self.datasets = datasets
        else:
            zarr_paths = dataset_cfg['zarr_paths']
            self.datasets = []
            for zarr_path in zarr_paths:
                dataset_cfg['zarr_path'] = zarr_path
                dataset = hydra.utils.instantiate(dataset_cfg)
                self.datasets.append(dataset)

        # 记录长度
        self.lengths = []
        self.cumulative_lengths = []
        total = 0
        for dataset in self.datasets:
            dataset_len = len(dataset)
            self.lengths.append(dataset_len)
            total += dataset_len
            self.cumulative_lengths.append(total)
        self.total_length = total

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx: int):
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                sub_idx = idx if i == 0 else idx - self.cumulative_lengths[i - 1]
                return self.datasets[i][sub_idx]
        raise IndexError(f"Index {idx} out of range for dataset length {self.total_length}")
        
    def get_normalizer(self, mode='limits', **kwargs):
        data_action = []
        data_state = []

        for dataset in self.datasets:
            rb = dataset.replay_buffer
            data_action.append(rb['action'][:])
            data_state.append(rb['state'][:])

        data_action = np.concatenate(data_action, axis=0)
        data_state = np.concatenate(data_state, axis=0)

        data = {
            'action': data_action,
            'state': data_state,
        }

        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        normalizer['wrist_image'] = get_image_range_normalizer()
        return normalizer

    def get_validation_dataset(self) -> 'AllTasksMixedRobotDataset':
        # 关键：直接基于已有 dataset 创建其验证版本
        val_datasets = []
        for ds in self.datasets:
            if hasattr(ds, "get_validation_dataset"):
                val_ds = ds.get_validation_dataset()
                val_datasets.append(val_ds)
            else:
                raise ValueError(f"Dataset {ds} does not implement get_validation_dataset()")
        
        return AllTasksMixedRobotDataset(datasets=val_datasets)
