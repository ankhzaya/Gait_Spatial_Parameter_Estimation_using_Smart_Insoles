import sys
import torch
import os
sys.path.append('../')

def load_pretrained_model(model, pretrained_path, gpu_idx):
    """Load weights from the pretrained model"""
    assert os.path.isfile(pretrained_path), "=> no checkpoint found at '{}'".format(pretrained_path)
    if gpu_idx is None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu_idx)
        checkpoint = torch.load(pretrained_path, map_location=loc)
    pretrained_dict = checkpoint['state_dict']
    if hasattr(model, 'module'):
        model_state_dict = model.module.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
        # 2. overwrite entries in the existing state dict
        model_state_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.module.load_state_dict(model_state_dict)
    else:
        model_state_dict = model.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_state_dict}
        # Load global to local stage
        # 2. overwrite entries in the existing state dict
        model_state_dict.update(pretrained_dict)
        # 3. load the new state dict
        model.load_state_dict(model_state_dict)
    return model

def make_data_parallel(model, configs):
    if configs.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if configs.gpu_idx is not None:
            torch.cuda.set_device(configs.gpu_idx)
            model.cuda(configs.gpu_idx)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            configs.batch_size = int(configs.batch_size / configs.ngpus_per_node)
            configs.num_workers = int((configs.num_workers + configs.ngpus_per_node - 1) / configs.ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[configs.gpu_idx],
                                                              find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif configs.gpu_idx is not None:
        torch.cuda.set_device(configs.gpu_idx)
        model = model.cuda(configs.gpu_idx)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    return model

def resume_model(resume_path, arch, gpu_idx):
    """Resume training model from the previous trained checkpoint"""
    assert os.path.isfile(resume_path), "=> no checkpoint found at '{}'".format(resume_path)
    if gpu_idx is None:
        checkpoint = torch.load(resume_path, map_location='cpu')
    else:
        # Map model to be loaded to specified single gpu.
        loc = 'cuda:{}'.format(gpu_idx)
        checkpoint = torch.load(resume_path, map_location=loc)
    assert arch == checkpoint['configs'].arch, "Load the different arch..."
    print("=> loaded checkpoint '{}' (epoch {})".format(resume_path, checkpoint['epoch']))

    return checkpoint

def get_num_parameters(model):
    """Count number of trained parameters of the model"""
    if hasattr(model, 'module'):
        num_parameters = sum(p.numel() for p in model.module.parameters() if p.requires_grad)
    else:
        num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return num_parameters