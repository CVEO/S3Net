import os
import csv

import torch
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

from utils import create_mask, adjust_learning_rate, masked_cross_entropy_loss

def save_checkpoint(model, epoch, args, local_rank):

    savefilename = os.path.join(args.save_ckpt_path, f'train_ckpt_{epoch}.tar')
    if args.is_distributed:
        if args.rank == 0:
            torch.save({'state_dict': model.module.state_dict()}, savefilename)
    else:
        torch.save({'state_dict': model.state_dict()}, savefilename)


def log_results(epoch, train_loss, valid_loss, args, fieldnames):
    with open(args.save_csv_file_path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({
            'epoch': epoch,
            'train_loss': train_loss,
            'valid_loss': valid_loss,
        })


def train_one_epoch(epoch, model, optimizer, train_loader, local_rank, args):

    model.train()
    total_train_loss = 0.0

    # Learning rate adjustment and distributed training setup
    adjust_learning_rate(optimizer, epoch)
    if args.is_distributed:
        train_loader.sampler.set_epoch(epoch)

    # Training loop
    for batch_idx, (left, right, disp, cls) in tqdm(enumerate(train_loader), total=len(train_loader)):

        left, right, disp = [
            tensor.to(local_rank).float() for tensor in 
            [left, right, disp]
        ]

        cls = cls.to(local_rank).long()

        optimizer.zero_grad()

        pred_disp1, pred_disp2, pred_disp3, pred_cls = model(left, right)

        # Create masks for different disparity scales
        disp, mask = create_mask(disp, args.maxdisp, args.mindisp)

        # Compute losses for different scales
        loss1 = 0.5*F.smooth_l1_loss(pred_disp1[mask], disp[mask], size_average=True) + \
                0.7*F.smooth_l1_loss(pred_disp2[mask], disp[mask], size_average=True) + \
                F.smooth_l1_loss(pred_disp3[mask], disp[mask], size_average=True)
        loss2 = masked_cross_entropy_loss(pred_cls, cls)
        loss = 0.15*loss1 + loss2

        if args.is_distributed:
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / args.world_size

        loss.backward()
        optimizer.step()

        if args.rank == 0:
            total_train_loss += loss.detach().cpu().numpy()

    return total_train_loss / len(train_loader)


def validate_one_epoch(epoch, model, valid_loader, local_rank, args):

    model.eval()
    total_valid_loss = 0.0

    if args.is_distributed:
        valid_loader.sampler.set_epoch(epoch)

    # Validation loop
    with torch.no_grad():
        for batch_idx, (left, right, disp, cls) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            
            left, right, disp = [
                tensor.to(local_rank).float() for tensor in 
                [left, right, disp]
            ]

            cls = cls.to(local_rank).long()
            
            pred_disp1, pred_disp2, pred_disp3, pred_cls = model(left, right)

            # Create masks for different disparity scales
            disp, mask = create_mask(disp, args.maxdisp, args.mindisp)

            # Compute losses for different scales
            loss1 = 0.5*F.smooth_l1_loss(pred_disp1[mask], disp[mask], size_average=True) + \
                    0.7*F.smooth_l1_loss(pred_disp2[mask], disp[mask], size_average=True) + \
                    F.smooth_l1_loss(pred_disp3[mask], disp[mask], size_average=True)
            loss2 = masked_cross_entropy_loss(pred_cls, cls)
            loss = 0.15*loss1 + loss2

            if args.is_distributed:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss = loss / args.world_size
            
            if args.rank == 0:
                total_valid_loss += loss.detach().cpu().numpy()

    return total_valid_loss / len(valid_loader)