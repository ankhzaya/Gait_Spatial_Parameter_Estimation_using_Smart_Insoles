import sys
import os

import numpy as np
from torch.utils.data import Dataset

sys.path.append('../')

from data_processing.stride_data_utils import get_annos_infor
from data_processing.stride_data_augmentation import augment_sequence
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

class GaitSpatParamDataset(Dataset):
    def __init__(self, dataset_dir, configs, mode='train', augmentation=False):
        self.dataset_dir = dataset_dir
        self.mode = mode
        self.configs = configs
        self.augmentation = augmentation

        if self.mode == 'train' or self.mode == 'val':
            self.annos_infor = get_annos_infor(self.dataset_dir, self.mode, configs, augment=True)
        else:
            self.annos_infor = get_annos_infor(self.dataset_dir, self.mode, configs, augment=False)

        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.is_test = (self.mode == 'test')
        self.num_samples = len(self.annos_infor)

    def __getitem__(self, index):
        return self.load_input_with_targets(index)

    def __len__(self):
        return self.num_samples

    def load_input_with_targets(self, index):
        imu_sequence, target_long_spatial, target_short_spatial, sbj, speed, trial = self.annos_infor[index]
        len_sequence = imu_sequence.shape[0]

        # Pad the IMU sequence to the maximum length of 256
        if len_sequence < 256:
            pad_length = 256 - len_sequence
            imu_sequence = np.pad(imu_sequence, ((0, pad_length), (0, 0)), mode='constant', constant_values=0)
        else:
            imu_sequence = imu_sequence[:256]

        if self.augmentation:
            imu_sequence = augment_sequence(imu_sequence, visualize=False)

        # Integrate long and short spatial targets
        spat_target = np.concatenate((target_long_spatial, target_short_spatial), axis=0)

        return np.array(imu_sequence), np.array(spat_target), sbj, speed, trial   # test

if __name__ == '__main__':
    from configs.configs import parse_configs
    configs = parse_configs()

    configs.mode = 'test'

    augment = False
    print('dataset dir: {}'.format(configs.dataset_dir))

    gait_dataset = GaitSpatParamDataset(configs.dataset_dir, configs, mode=configs.mode, augmentation=augment)
    print('len gait_dataset: {}'.format(len(gait_dataset)))

    example_idx = 1
    # imu_sequence, spatial_target_long, spatial_target_short, sbj, speed, trial = gait_dataset.__getitem__(example_idx)
    imu_sequence, spatial_target, sbj, speed, trial = gait_dataset.__getitem__(example_idx)

    print('imu_sequence shape: {}'.format(imu_sequence.shape))
    # print('spatial_target_long shape: {}'.format(spatial_target_long.shape))
    # print('spatial_target_short shape: {}'.format(spatial_target_short.shape))

    print('spatial_target shape: {}'.format(spatial_target.shape))

    # Plot the IMU sequence
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 5))
    # plt.plot(imu_sequence[:, 0], label='L_ACC_X')
    # plt.plot(imu_sequence[:, 1], label='L_ACC_Y')
    # # plt.plot(imu_sequence[:, 2], label='L_ACC_Z')
    # plt.plot(imu_sequence[:, 3], label='R_ACC_X')
    # plt.plot(imu_sequence[:, 4], label='R_ACC_Y')
    # # plt.plot(imu_sequence[:, 5], label='R_ACC_Z')
    # plt.title('IMU Sequence')
    # plt.xlabel('Time Steps')
    # plt.ylabel('Acceleration')
    # plt.legend()
    # plt.show()

# [44.242 41.671 26.396 17.778 31.954 26.111 19.    15.   ]