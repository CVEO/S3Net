import os
import csv
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.optim as optim
from tqdm import tqdm
import torch.multiprocessing as mp

from utils import (
    parser, 
    initialize_model, 
    setup,
    )
from data import generate, initialize_dataloaders
from train import train_one_epoch, validate_one_epoch, save_checkpoint, log_results

def train():
    args = parser()
    setup(args)

    args.save_ckpt_path = args.save_ckpt_path or f'{args.model}_{args.dataset}'
    args.save_csv_file_path = args.save_csv_file_path or f'{args.model}_{args.dataset}.csv'
    os.makedirs(args.save_ckpt_path, exist_ok=True)
    
    # Distributed training initialization
    model = initialize_model(args)
    if args.rank == 0:
        print(args)

    # Dataset loading
    train_dataset, valid_dataset = generate(args.dataset, args.datapath)
    train_loader, valid_loader = initialize_dataloaders(args, train_dataset, valid_dataset)

    # Log file initialization
    fieldnames = ['epoch', 'train_loss', 'valid_loss']
    if not torch.cuda.is_available() and not os.path.exists(args.save_csv_file_path):
        with open(args.save_csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    # Loss function and optimizer configuration
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if args.rank == 0:
        print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")

    optimizer = optim.Adam([
        {'params': trainable_params, 'name': 'model', 'lr': args.lr},
    ], betas=(0.9, 0.999), weight_decay=1e-7)   

    # Training loop
    for epoch in range(args.epochs):
        # Training and validation
        train_loss = train_one_epoch(epoch, model, optimizer, train_loader, args.local_rank, args)
        save_checkpoint(model, epoch, args, args.local_rank)
        valid_loss = validate_one_epoch(epoch, model, valid_loader, args.local_rank, args)
        
        if args.rank == 0:
            log_results(epoch, train_loss, valid_loss, args, fieldnames)

if __name__ == '__main__':
    train()