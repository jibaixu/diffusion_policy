from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class AllTasksImageLanguageDataset(BaseImageDataset):
    def __init__(self,
            zarr_path, 
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            *args,
            **kwargs,
            ):
        
        super().__init__()
        #! 懒加载数据集到 replay_buffer 中
        self.zarr_path = zarr_path
        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['base_image', 'hand_image', 'state', 'action', 'lang_emb'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, 
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, 
            max_n=max_train_episodes, 
            seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=horizon,
            pad_before=pad_before, 
            pad_after=pad_after,
            episode_mask=train_mask,
            zarr_path=zarr_path)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, 
            sequence_length=self.horizon,
            pad_before=self.pad_before, 
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            zarr_path=self.zarr_path,
            )
        val_set.train_mask = ~self.train_mask
        return val_set

    #!############# 根据数据集构造归一化器 #############
    def get_normalizer(self, mode='limits', **kwargs):
        #! 定义需要归一化的数据字典
        data = {
            'action': self.replay_buffer['action'],
            'state': self.replay_buffer['state'],
        }
        normalizer = LinearNormalizer()
        #! 数据字典中 action 和 state 字段构造线性归一化器
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        #! 数据字典中图像字段使用 SingleFieldLinearNormalizer 构造
        normalizer['image'] = get_image_range_normalizer()
        normalizer['wrist_image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):
        state = sample['state'].astype(np.float32)
        image = np.moveaxis(sample['base_image'].astype(np.float32), -1, 1) / 255
        wrist_image = np.moveaxis(sample['hand_image'].astype(np.float32), -1, 1) / 255

        data = {
            'obs': {
                'image': image, # T, 3, 256, 256
                'wrist_image': wrist_image, # T, 3, 256, 256
                'state': state, # T, 8
            },
            'action': sample['action'].astype(np.float32), # T, 7
            'lang_emb': sample['lang_emb'].astype(np.float32), # T, 1, 768
        }
        return data
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data


def test():
    import os
    zarr_path = os.path.expanduser('~/dev/diffusion_policy/data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = AllTasksImageLanguageDataset(zarr_path, horizon=16)

    # from matplotlib import pyplot as plt
    # normalizer = dataset.get_normalizer()
    # nactions = normalizer['action'].normalize(dataset.replay_buffer['action'])
    # diff = np.diff(nactions, axis=0)
    # dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
