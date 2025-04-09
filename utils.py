import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from models.model import SSNet

def parser() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='S3Net semantic Stereo Matching Model Configuration')

    # Model & Dataset Configuration
    parser.add_argument('--model', type=str, default='S3Net', help='Model name'),  
    parser.add_argument('--dataset', type=str, default='DFC2019', help='Dataset name'), 
    parser.add_argument('--datapath', type=str, default="./dataset/US3D", help='Dataset path'), 

    # Disparity Configuration
    parser.add_argument('--maxdisp', type=int, default=48, help='Maximum disparity range'), 
    parser.add_argument('--mindisp', type=int, default=-48, help='Minimum disparity range'), 
    parser.add_argument('--classfication', default=6, help='class number')

    # Training Hyperparameters
    parser.add_argument('--lr', type=float, default=1e-4, help='Model learning rate'), 
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size'), 
    parser.add_argument('--num_workers', type=int, default=2, help='Number of workers'), 
    parser.add_argument('--epochs', type=int, default=120, help='Number of training epochs'), 

    # Checkpoint and Saving
    parser.add_argument('--ckpt', type=str, default='ckpt.tar', help='Pretrained model checkpoint path'), 
    parser.add_argument('--save_ckpt_path', type=str, default=None, help='Model checkpoint saving path'), 
    parser.add_argument('--save_csv_file_path', type=str, default=None, help='Model training log saving path'), 

    # Distributed Training
    parser.add_argument('--world_size', type=int, default=1, help='Total number of distributed training processes'), 
    parser.add_argument('--is_distributed', type=bool, default=False, help='Enable distributed training mode'), 
    parser.add_argument('--local_rank', type=int, default=None, help='Local process rank for distributed training'), 

    # Parse arguments
    args, _ = parser.parse_known_args()
    return args


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def adjust_learning_rate(optimizer, epoch):
    lr_model = 1e-3 * (0.5) ** (epoch // 40)

    for param_group in optimizer.param_groups:
        if param_group['name'] == 'model':
            param_group['lr'] = lr_model

def masked_cross_entropy_loss(y_pred, y_true):
    loss = F.cross_entropy(y_pred, y_true, reduction='none')
    return loss.mean()

def create_mask(disp, maxdisp, mindisp):
    disp = disp.unsqueeze(1)
    return disp, (disp != -999) & (~torch.isnan(disp)) & (disp >= mindisp) & (disp <= maxdisp)


def weights_init(m):
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose3d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def initialize_distributed(args):
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.world_size = num_gpus
    args.is_distributed = num_gpus > 1

    if args.is_distributed:
        dist.init_process_group(backend='nccl', init_method='env://')
        local_rank = dist.get_rank()
        torch.cuda.set_device(local_rank)

    else:
        if torch.cuda.is_available():
            local_rank = torch.cuda.current_device()
    
    args.local_rank = local_rank


def setup(args):
    rank = int(os.environ.get("RANK", 0))  
    world_size = int(os.environ.get("WORLD_SIZE", 1))  
    local_rank = int(os.environ.get("LOCAL_RANK", 0))  

    args.world_size = world_size
    args.is_distributed = world_size > 1
    num_gpus = torch.cuda.device_count()
    
    local_rank = min(local_rank, num_gpus - 1)

    args.local_rank = local_rank
    args.rank = rank
    
    torch.cuda.set_device(local_rank)
    
    if args.rank == 0:
        print(f"Total available GPUs: {world_size}")
    
    dist.init_process_group(
        backend='nccl', 
        init_method='env://',
        world_size=world_size, 
        rank=rank
    )
    
    torch.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)

def create_mask(disp, maxdisp, mindisp):
    return disp, (disp != -999) & (~torch.isnan(disp)) & (disp >= mindisp) & (disp <= maxdisp)

def initialize_model(args):

    model = SSNet(args.maxdisp, args.mindisp, args.classfication)
    model.apply(weights_init)

    model = model.to(args.local_rank)

    if args.rank == 0:
        print(f'Number of model parameters: {sum([p.data.nelement() for p in model.parameters()]) / 1e6:.2f}M')

    if args.is_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank,
            find_unused_parameters=True)
    else:
        if torch.cuda.is_available():
            model = nn.DataParallel(model)

    return model

