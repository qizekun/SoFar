import os
import json
import random
from .build import DATASETS
import torch.utils.data as data
from orientation.utils.logger import *
from orientation.datasets.utils import *


@DATASETS.register_module()
class SemanticOrientation(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.occlusion = config.occlusion
        self.rotation = config.rotation
        self.noise = config.noise
        self.num_points = config.POINTS

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.json')
        self.data_list = json.load(open(self.data_list_file))

        print_log(f'[DATASET] Open file {self.data_list_file}', logger='SemanticDirection')
        print_log(f'[DATASET] {len(self.data_list)} instances were loaded', logger='SemanticDirection')

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        pcs_path = os.path.join(self.pc_path, sample['point'])
        pcs = load_pts(pcs_path)
        pcs[:, :3] = pc_norm(pcs[:, :3])
        pcs[:, :3] = rotation(pcs[:, :3], sample['rotation'])
        instruction = sample['conversations'][0]['value'].replace('<point>\n', '')
        direction = np.array(sample['direction'])

        if self.rotation:
            if self.subset == 'train':
                rand_rot = np.random.uniform(-180, 180, 3).tolist()
                pcs[:, :3] = rotation(pcs[:, :3], rand_rot)
                direction = rotation(direction, rand_rot)
            else:
                rand_rot = sample['rand_rot']
                pcs[:, :3] = rotation(pcs[:, :3], rand_rot)
                direction = rotation(direction, rand_rot)

        if self.occlusion:
            pcs = occlusion(pcs, self.num_points, 90, fix=True if self.subset == 'test' else False)

        if self.noise:
            pcs += np.random.randn(*pcs.shape) * 0.01

        idx = random.sample(range(pcs.shape[0]), self.num_points)
        pcs = pcs[idx]
        pcs[:, :3] = pc_norm(pcs[:, :3])

        pcs = torch.from_numpy(pcs).float()
        direction = torch.from_numpy(direction).float()
        return pcs, direction, instruction

    def __len__(self):
        return len(self.data_list)


@DATASETS.register_module()
class SO_FT(data.Dataset):
    def __init__(self, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.occlusion = config.occlusion
        self.num_points = config.POINTS

        self.data_list_file = os.path.join(self.data_root, f'train.json')
        self.data_list = json.load(open(self.data_list_file))
        if self.subset == 'train':
            self.data_list = self.data_list[:int(len(self.data_list) * 0.9)]
        else:
            self.data_list = self.data_list[int(len(self.data_list) * 0.9):]

        print_log(f'[DATASET] Open file {self.data_list_file}', logger='SemanticDirection')
        print_log(f'[DATASET] {len(self.data_list)} instances were loaded', logger='SemanticDirection')

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        pcs_path = os.path.join(self.pc_path, sample['point'])
        pcs = load_pts(pcs_path)
        pcs[:, :3] = pc_norm(pcs[:, :3])
        instruction = sample['conversations'][0]['value'].replace('<point>\n', '')
        direction = np.array(sample['direction'])

        n = pcs.shape[0]
        if n >= self.num_points:
            idx = random.sample(range(n), self.num_points)
            pcs = pcs[idx]
        else:
            k = self.num_points // n + 1
            pcs = interpolate_point_cloud(pcs, k)
            n = pcs.shape[0]
            idx = random.sample(range(n), self.num_points)
            pcs = pcs[idx]

        pcs = torch.from_numpy(pcs).float()
        direction = torch.from_numpy(direction).float()

        direction = direction.repeat(10 // len(direction), 1)
        direction = direction[:10]

        return pcs, direction, instruction

    def __len__(self):
        return len(self.data_list)
