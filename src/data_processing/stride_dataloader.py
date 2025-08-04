import torch
from torch.utils.data import DataLoader
from data_processing.stride_dataset import GaitSpatParamDataset



def create_test_dataloader(configs):
    test_dataset = GaitSpatParamDataset(configs.dataset_dir, configs, mode='test', augmentation=False)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=configs.test_batch_size,
        shuffle=False,
        num_workers=configs.num_workers,
        drop_last=True,
    )

    return test_dataloader


if __name__ == '__main__':
    from configs.configs import parse_configs
    configs = parse_configs()


    test = create_test_dataloader(configs)
    print('len test: {}'.format(len(test)))
