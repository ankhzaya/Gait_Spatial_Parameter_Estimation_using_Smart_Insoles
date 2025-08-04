import torch
import os
import datetime
import argparse
from easydict import EasyDict as edict
import pandas as pd
import numpy as np
import sys

sys.path.append('../')

from utils.misc import make_folder

def parse_configs():
    parser = argparse.ArgumentParser(description='Gait Spatial Parameters Estimation')
    parser.add_argument('--seed', type=int, default=2025,
                        help='re-produce the results with seed random')
    parser.add_argument('--saved_fn', type=str, default='Conformer_Network', metavar='FN',
                        help='The name using for saving logs, models,...')
    ####################################################################
    ##############     Model configs            ###################
    ####################################################################
    parser.add_argument('-a', '--arch', type=str, default='conformer', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--pretrained_imagenet', action='store_true',
                        help='If true, use pre-trained weights from imagenet dataset')
    parser.add_argument('--dropout_p', type=float, default=0.2, metavar='P',
                        help='The dropout probability of the model')
    parser.add_argument('--pretrained_path', type=str,
                        default='../../checkpoints/Conformer_Network/Conformer_Network_best.pth',
                        metavar='PATH', help='the path of the pretrained checkpoint')
    # parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
    #                      help='the path of the pretrained checkpoint')

    ####################################################################
    ##############     Dataloader and Running configs            #######
    ####################################################################
    parser.add_argument('--no_all', action='store_true',
                        help='If true, use only Walking data for training and validation set')
    parser.add_argument('--no-test', action='store_true',
                        help='If true, dont evaluate the model on the test set')
    parser.add_argument('--val-size', type=float, default=0.2,
                        help='The size of validation set')
    parser.add_argument('--no_test', action='store_true',
                        help='If true, dont evaluate the model on the test set')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='mini-batch size (default: 16), this is the total'
                             'batch size of all GPUs on the c8urrent node when using'
                             'Data Parallel or Distributed Data Parallel')
    parser.add_argument('--test_batch_size', type=int, default=1,
                       help='None')
    parser.add_argument('--subdivisions', type=int, default=16,
                        help='subdivisions during training')
    parser.add_argument('--print_freq', type=int, default=100, metavar='N',
                        help='print frequency (default: 10)')
    parser.add_argument('--checkpoint_freq', type=int, default=1, metavar='N',
                        help='frequency of saving checkpoints (default: 3)')
    parser.add_argument('--tensorboard_freq', type=int, default=20, metavar='N',
                        help='frequency of saving tensorboard (default: 20)')

    ####################################################################
    ##############     Training strategy            ###################
    ####################################################################
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='the starting epoch')
    parser.add_argument('--num_epochs', type=int, default=100, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--warmup_epochs', type=int, default=8, metavar='N',)
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='initial learning rate')
    parser.add_argument('--minimum_lr', type=float, default=1e-5, metavar='MIN_LR',
                        help='minimum learning rate during training')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='momentum')
    parser.add_argument('-wd', '--weight_decay', type=float, default=1e-6, metavar='WD',
                        help='weight decay (default: 1e-6)')
    parser.add_argument('--optimizer_type', type=str, default='adam', metavar='OPTIMIZER',
                        help='the type of optimizer, it can be sgd or adam')
    parser.add_argument('--lr_type', type=str, default='plateau', metavar='SCHEDULER',
                        help='the type of the learning rate scheduler (steplr or plateau)')
    parser.add_argument('--lr_factor', type=float, default=0.5, metavar='FACTOR',
                        help='reduce the learning rate with this factor')
    parser.add_argument('--lr_step_size', type=int, default=5, metavar='STEP_SIZE',
                        help='step_size of the learning rate when using steplr scheduler')
    parser.add_argument('--lr_patience', type=int, default=3, metavar='N',
                        help='patience of the learning rate when using ReduceoPlateau scheduler')
    parser.add_argument('--earlystop_patience', type=int, default=12, metavar='N',
                        help='Early stopping the training process if performance is not improved within this value')
    ####################################################################
    ##############     Loss weight            ###################
    ####################################################################
    parser.add_argument('--losses_weight', type=list, default=[1., 4.],
                        help='The weight of losses of the event spotting module')
    parser.add_argument('--loss_type', type=str, default='huber',
                        help='The weight of loss of the event spotting module')
    parser.add_argument('--spat_loss_type', type=str, default='huber',
                        help='The weight of loss of the event spotting module')

    ####################################################################
    ##############     Distributed Data Parallel            ############
    ####################################################################
    parser.add_argument('--world-size', default=-1, type=int, metavar='N',
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, metavar='N',
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:29500', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')
    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--multiprocessing_distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    ####################################################################
    ##############     Evaluation configurations     ###################
    ####################################################################
    parser.add_argument('--evaluate', action='store_true',
                        help='only evaluate the model, not training')
    parser.add_argument('--resume_path', type=str, default=None, metavar='PATH',
                        help='the path of the resumed checkpoint')
    parser.add_argument('--use_best_checkpoint', action='store_true',
                        help='If true, choose the best model on val set, otherwise choose the last model')

    configs = edict(vars(parser.parse_args()))

    ####################################################################
    ############## Hardware configurations ############################
    ####################################################################
    configs.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    configs.ngpus_per_node = torch.cuda.device_count()

    configs.pin_memory = True

    ####################################################################
    ##############     Data configs            ###################
    ####################################################################
    configs.working_dir = '../../'
    configs.dataset_dir = '../../sample_dataset'

    configs.num_features = 12
    configs.num_strides = 256

    configs.model_utils = edict()

    configs.test = edict()
    configs.test.speeds = ['F']
    configs.mode = ['train', 'val', 'test']

    ####################################################################
    ############## logs, Checkpoints, and results dir ########################
    ####################################################################
    configs.checkpoints_dir = os.path.join(configs.working_dir, 'checkpoints', configs.saved_fn)
    configs.logs_dir = os.path.join(configs.working_dir, 'logs', configs.saved_fn)
    configs.use_best_checkpoint = True

    if configs.use_best_checkpoint:
        configs.saved_weight_name = os.path.join(configs.checkpoints_dir, '{}_best.pth'.format(configs.saved_fn))
    else:
        configs.saved_weight_name = os.path.join(configs.checkpoints_dir, '{}.pth'.format(configs.saved_fn))

    configs.results_dir = os.path.join(configs.working_dir, 'results')

    make_folder(configs.checkpoints_dir)
    make_folder(configs.logs_dir)
    make_folder(configs.results_dir)

    return configs

if __name__ == "__main__":
    configs = parse_configs()
    print(configs)

    print(datetime.date.today())
    print(datetime.datetime.now().year)
