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
        self.lengths = []   #! 子数据集分为的训练集长度，并非正确数据集长度
        self.cumulative_lengths = []
        total = 0
        for dataset in self.datasets:
            #! 子数据集中的长度已经划分成训练集并加载到 sampler 中，所以值为训练集长度，保证索引映射正确
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
        #! 直接访问 replay_buffer，绕过内部 sampler，所以归一化是基于所有(训练集+验证集)数据
        data_action = [dataset.replay_buffer['action'][:] for dataset in self.datasets]
        data_state = [dataset.replay_buffer['state'][:] for dataset in self.datasets]

        data_action = np.vstack(data_action)
        data_state = np.vstack(data_state)

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
