import time
import os
import sys
import warnings

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

sys.path.append('../')

from data_processing.stride_dataloader import create_test_dataloader

from models.estimation_models import create_model
from utils.misc import AverageMeter
from utils.logger import Logger

from models.model_utils import load_pretrained_model, make_data_parallel
from utils.test_utils import add_value_dict, add_value_spat_dict


def main(configs):
    if configs.gpu_idx is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if configs.dist_url == "env://" and configs.world_size == -1:
        configs.world_size = int(os.environ["WORLD_SIZE"])

    configs.distributed = configs.world_size > 1 or configs.multiprocessing_distributed

    if configs.multiprocessing_distributed:
        configs.world_size = configs.ngpus_per_node * configs.world_size
        mp.spawn(main_worker, nprocs=configs.ngpus_per_node, args=(configs,))
    else:
        main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx

    if configs.gpu_idx is not None:
        print("Use GPU: {} for training".format(configs.gpu_idx))
        configs.device = torch.device('cuda:{}'.format(configs.gpu_idx))

    if configs.distributed:
        if configs.dist_url == "env://" and configs.rank == -1:
            configs.rank = int(os.environ["RANK"])
        if configs.multiprocessing_distributed:
            configs.rank = configs.rank * configs.ngpus_per_node + gpu_idx

        dist.init_process_group(backend=configs.dist_backend, init_method=configs.dist_url,
                                world_size=configs.world_size, rank=configs.rank)

    logger = Logger(configs.logs_dir, configs.saved_fn)

    # model
    model = create_model(configs)
    model = model.to(configs.device)

    # Data Parallel
    model = make_data_parallel(model, configs)

    print('configs.pretrained_path: {}'.format(configs.pretrained_path))

    if configs.pretrained_path is not None:
        model = load_pretrained_model(model, configs.pretrained_path, gpu_idx)

    # Load dataset
    test_dataloader = create_test_dataloader(configs)
    print('len(test_loader): {}'.format(len(test_dataloader)))
    test(test_dataloader, model, configs, logger)


def test(test_loader, model, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        start_time = time.time()

    # targets
    RSTRL, LSTRL, RSTPL, LSTPL, RSTRW, LSTRW, RSTPW, LSTPW = [], [], [], [], [], [], [], []
    STRL, STPL, STRW, STPW = [], [], [], []

    # predictions
    pred_RSTRL, pred_LSTRL, pred_RSTPL, pred_LSTPL, pred_RSTRW, pred_LSTRW, pred_RSTPW, pred_LSTPW = [], [], [], [], [], [], [], []
    pred_STRL, pred_STPL, pred_STRW, pred_STPW = [], [], [], []

    subjects, speeds, trials = [], [], []

    for batch_idx, (imu_seq, target_spat, sbj, speed, trial) in enumerate(
        tqdm(test_loader)):
        data_time.update(time.time() - start_time)

        batch_size = imu_seq.size(0)
        target_spat = target_spat.to(configs.device, non_blocking=True)

        imu_seq = imu_seq.to(configs.device, non_blocking=True)
        b_pred_spat = model(imu_seq.float())

        for sample_idx in range(batch_size):
            # prepare targets
            target_spat = target_spat[sample_idx]

            # prepare predictions
            pred_spat = b_pred_spat[sample_idx]
            pred_spat = pred_spat[0]

            subjects.append(sbj[sample_idx])
            speeds.append(speed[sample_idx])
            trials.append(trial[sample_idx])

            # add to targets
            RSTRL.append(target_spat[0].item())
            LSTRL.append(target_spat[1].item())
            RSTPL.append(target_spat[2].item())
            LSTPL.append(target_spat[3].item())
            RSTPW.append(target_spat[4].item())
            LSTPW.append(target_spat[5].item())
            RSTRW.append(target_spat[6].item())
            LSTRW.append(target_spat[7].item())

            # Find average values of Left and Right for each parameter
            STRL.append((target_spat[0].item() + target_spat[1].item()) / 2)
            STPL.append((target_spat[2].item() + target_spat[3].item()) / 2)
            STPW.append((target_spat[4].item() + target_spat[5].item()) / 2)
            STRW.append((target_spat[6].item() + target_spat[7].item()) / 2)

            # add to predictions
            pred_RSTRL.append(pred_spat[0].item())
            pred_LSTRL.append(pred_spat[1].item())
            pred_RSTPL.append(pred_spat[2].item())
            pred_LSTPL.append(pred_spat[3].item())
            pred_RSTPW.append(pred_spat[4].item())
            pred_LSTPW.append(pred_spat[5].item())
            pred_RSTRW.append(pred_spat[6].item())
            pred_LSTRW.append(pred_spat[7].item())

            # Find average values of Left and Right for each parameter
            pred_STRL.append((pred_spat[0].item() + pred_spat[1].item()) / 2)
            pred_STPL.append((pred_spat[2].item() + pred_spat[3].item()) / 2)
            pred_STPW.append((pred_spat[4].item() + pred_spat[5].item()) / 2)
            pred_STRW.append((pred_spat[6].item() + pred_spat[7].item()) / 2)

            batch_time.update(time.time() - start_time)
            start_time = time.time()

    # Save the results
    save_dir = os.path.join(configs.results_dir, configs.saved_fn)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'stride_wise_{configs.test.speeds[0]}.csv')

    result_dict = {}
    result_dict['Subject'] = subjects
    result_dict['Speed'] = speeds
    result_dict['Trial'] = trials
    result_dict['Target_RSTRL'] = RSTRL
    result_dict['Pred_RSTRL'] = pred_RSTRL
    result_dict['Target_LSTRL'] = LSTRL
    result_dict['Pred_LSTRL'] = pred_LSTRL
    result_dict['Target_RSTPL'] = RSTPL
    result_dict['Pred_RSTPL'] = pred_RSTPL
    result_dict['Target_LSTPL'] = LSTPL
    result_dict['Pred_LSTPL'] = pred_LSTPL
    result_dict['Target_RSTRW'] = RSTRW
    result_dict['Pred_RSTRW'] = pred_RSTRW
    result_dict['Target_LSTRW'] = LSTRW
    result_dict['Pred_LSTRW'] = pred_LSTRW
    result_dict['Target_RSTPW'] = RSTPW
    result_dict['Pred_RSTPW'] = pred_RSTPW
    result_dict['Target_LSTPW'] = LSTPW
    result_dict['Pred_LSTPW'] = pred_LSTPW

    # Find average values of Left and Right for each parameter
    result_dict['Target_STRL'] = STRL
    result_dict['Pred_STRL'] = pred_STRL
    result_dict['Target_STPL'] = STPL
    result_dict['Pred_STPL'] = pred_STPL
    result_dict['Target_STRW'] = STRW
    result_dict['Pred_STRW'] = pred_STRW
    result_dict['Target_STPW'] = STPW
    result_dict['Pred_STPW'] = pred_STPW

    # Save the results to a CSV file
    import pandas as pd
    df = pd.DataFrame(result_dict)
    df.to_csv(save_path, index=False)


if __name__ == '__main__':
    from configs.configs import parse_configs
    configs = parse_configs()
    main(configs)
