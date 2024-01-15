import time
import argparse
# from typing import Any
# from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from osgeo import gdal, gdal_array
import torch
import csv
# import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall, F1Score
from model.model import SSNet
import torch.utils.data
# import lightning.pytorch as L
# import lightning.pytorch.callbacks as callbacks
# import lightning.pytorch.loggers.csv_logs as csv_logs
from data.data import *
import data.DFC2019Loader as DA



def adjust_learning_rate(optimizer, epoch):
    lr = 1e-4 * (0.5)**(epoch//40)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def masked_cross_entropy_loss(y_pred, y_true):
    loss = F.cross_entropy(y_pred, y_true, reduction='none')
    return loss.mean()


def convert_syncbn(model):
    return nn.SyncBatchNorm.convert_sync_batchnorm(model)

# class Module(L.LightningModule):
#     def __init__(self, model) -> None:
#         super().__init__()
#         self.model = model
#         accuracy_metric = Accuracy(task="multiclass", num_classes=6)
#         precision_metric = Precision(task="multiclass", num_classes=6)
#         recall_metric = Recall(task="multiclass", num_classes=6)
#         f1_metric = F1Score(task="multiclass", num_classes=6)

#         self.metrics = nn.ModuleDict(
#             accuracy = accuracy_metric,
#             precision = precision_metric,
#             recall = recall_metric,
#             f1 = f1_metric
#         )

#     def configure_optimizers(self) -> OptimizerLRScheduler:
#         optimizer = optim.Adam(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-7)
#         adjust_learning_rate(optimizer, epoch)
#         return optimizer

#     def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
#         left, right, disp, cls = batch

#         mask = (disp!=-32768)
#         mask.detach_()
        
#         disp1, disp2, disp3, cls1 = self.model(left, right)

#         loss1 = 0.5*F.smooth_l1_loss(disp1[mask], disp[mask], size_average=True) + \
#                 0.7*F.smooth_l1_loss(disp2[mask], disp[mask], size_average=True) + \
#                 F.smooth_l1_loss(disp3[mask], disp[mask], size_average=True)
#         loss2 = masked_cross_entropy_loss(cls1, cls)

#         loss = 0.1*loss1 + loss2

#         self.log('train_loss', loss)

#         return loss
    
#     def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
#         left, right, disp, cls = batch

#         disp1, disp2, disp3, cls1 = self.model(left, right)

#         mask = (disp!=-32768)
#         mask.detach_()

#         loss1 = 0.5*F.smooth_l1_loss(disp1[mask], disp[mask], size_average=True) + \
#                 0.7*F.smooth_l1_loss(disp2[mask], disp[mask], size_average=True) + \
#                 F.smooth_l1_loss(disp3[mask], disp[mask], size_average=True)
#         loss2 = masked_cross_entropy_loss(cls1, cls)

#         loss = 0.1*loss1 + loss2

#         for v in self.metrics.values():
#             v.update(cls1, cls)
#         values = {"test_loss": loss, **self.metrics}
#         self.log(values)
#         return loss, self.metrics
    
# def train_val(model, train_data, valid_data):

#     trainer = L.Trainer(devices=4,
#                         max_epochs=120,
#                         val_check_interval=1,
#                         enable_checkpointing=True,
#                         logger=csv_logs.CSVLogger(),
#                         default_root_dir="checkpoints/",
#                         callbacks=[
#                             callbacks.ModelCheckpoint(every_n_epochs=1)
#                         ])

#     model = Module(model)
#     imgL, imgR, imgDsp, imgCls = train_data.forward()
#     TrainDataLoader = torch.utils.data.DataLoader(
#         DA.myImageFloder(imgL, imgR, imgDsp, imgCls, True), 
#         batch_size=8, shuffle=True, drop_last=False, num_workers=4)

#     imgL, imgR, imgDsp, imgCls = valid_data.forward()
#     VaildDataLoader = torch.utils.data.DataLoader(
#         DA.myImageFloder(imgL, imgR, imgDsp, imgCls, False), 
#         batch_size=8, shuffle=True, drop_last=False, num_workers=4)


#     trainer.fit(model, TrainDataLoader, VaildDataLoader)


def train(left, right, disp, cls, model, device, optimizer, debug_save = False):
    
    model.train()
    left = left.to(device)
    right = right.to(device)
    disp = disp.to(device).float()
    cls = cls.to(device)
    cls = cls.long()

    mask = (disp!=-32768)
    mask.detach_().to(device)
    
    optimizer.zero_grad()
    disp1, disp2, disp3, cls1 = model(left, right)

    loss1 = 0.5*F.smooth_l1_loss(disp1[mask], disp[mask], size_average=True) + \
            0.7*F.smooth_l1_loss(disp2[mask], disp[mask], size_average=True) + \
            F.smooth_l1_loss(disp3[mask], disp[mask], size_average=True)
    loss2 = masked_cross_entropy_loss(cls1, cls)

    loss = 0.15*loss1 + loss2
    loss.backward()
    optimizer.step()

    return loss

def valid(metrics:dict, left, right, disp, cls, model, device, optimizer, epoch = 0, debug_save = False):
    
    model.eval()

    left = left.to(device).float()
    right = right.to(device).float()
    disp = disp.to(device).float()
    cls = cls.to(device)
    cls = cls.long()

    mask = (disp!=-32768)
    with torch.no_grad():

        disp1, disp2, disp3, cls1 = model(left, right)

        # if debug_save:
        #     gdal_array.SaveArray(left[0].detach().cpu().numpy(), f"val-left-{epoch}.tif")
        #     gdal_array.SaveArray(right[0].detach().cpu().numpy(), f"val-right-{epoch}.tif")
        #     gdal_array.SaveArray(cls[0].detach().cpu().numpy(), f"val-cls-{epoch}.tif")
        #     gdal_array.SaveArray(cls1[0].detach().cpu().numpy(), f"val-cls1-{epoch}.tif")

        loss1 = 0.5*F.smooth_l1_loss(disp1[mask], disp[mask], size_average=True) + \
                0.7*F.smooth_l1_loss(disp2[mask], disp[mask], size_average=True) + \
                F.smooth_l1_loss(disp3[mask], disp[mask], size_average=True)
        loss2 = masked_cross_entropy_loss(cls1, cls)

        loss = 0.15*loss1 + loss2
        for v in metrics.values():
            v.update(cls1, cls)
        # accuracy_metric(cls1, cls)
        # precision_metric(cls1, cls)
        # recall_metric(cls1, cls)
        # f1_metric(cls1, cls)

    # accuracy_value = accuracy_metric.compute().item()
    # precision_value = precision_metric.compute().item()
    # recall_value = recall_metric.compute().item()
    # f1_value = f1_metric.compute().item()

    return loss.data

def main():

    parser = argparse.ArgumentParser(description='S3Net')
    parser.add_argument('--DFC2019', default='2019', help='DFC2019')
    parser.add_argument('--gpus', default='0,1,2,3')
    parser.add_argument('--datapath', default="./data/dataset/", help='data')
    parser.add_argument('--epochs', default=120, type=int, help='train epoch')
    parser.add_argument('--maxdisp', default=48, help='maxium disparity')
    parser.add_argument('--model', default='SSNet', help='select model')
    parser.add_argument('--training_samples', default=700, help='train number')
    parser.add_argument('--classfication', default=6, help='class number')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--savepath', default="./output/", help='saveckpt')
    args, _ = parser.parse_known_args()

    

    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # device = torch.device("cuda" if args.cuda else "cpu")
    # print(device, flush=True)

    if args.DFC2019 == "2019":
        train_data, valid_data = Dataload(args.datapath, args.training_samples)

    else:
        raise ValueError("no data")

    if args.model == "SSNet":
        model = SSNet(args.maxdisp, args.classfication)
    else:
        raise ValueError("no model")

    # train_val(model, train_data, valid_data)

    init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    print("local_rank", local_rank, flush=True)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-7)

    
    # imgL, imgR, imgDsp, imgCls = train_data.forward()
    # TrainDataLoader = torch.utils.data.DataLoader(
    #     DA.myImageFloder(imgL, imgR, imgDsp, imgCls, True), 
    #     batch_size=8, shuffle=True, drop_last=False, num_workers=4)

    # imgL, imgR, imgDsp, imgCls = valid_data.forward()
    # VaildDataLoader = torch.utils.data.DataLoader(
    #     DA.myImageFloder(imgL, imgR, imgDsp, imgCls, False), 
    #     batch_size=8, shuffle=True, drop_last=False, num_workers=4)
    

    imgL, imgR, imgDsp, imgCls = train_data.forward()
    train_dataset = DA.myImageFloder(imgL, imgR, imgDsp, imgCls, True)
    train_sampler = DistributedSampler(train_dataset)
    TrainDataLoader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=False, drop_last=False, sampler=train_sampler)

    imgL, imgR, imgDsp, imgCls = valid_data.forward()
    valid_dataset = DA.myImageFloder(imgL, imgR, imgDsp, imgCls, False)
    valid_sampler = DistributedSampler(valid_dataset)
    VaildDataLoader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=1, shuffle=False, drop_last=False, sampler=valid_sampler)

    # imgL, imgR, imgDsp, imgCls = test_data.forward()
    # TestDataLoader = torch.utils.data.DataLoader(
    #     DA.myImageFloder(imgL, imgR, imgDsp, imgCls, False), 
    #     batch_size=8, shuffle=True, drop_last=False, num_workers=4)
    model.to(device)
    model = convert_syncbn(model)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = DDP(model,
                    device_ids=[local_rank],
                    output_device=local_rank,
                    find_unused_parameters=True)
   
    csv_file_path = 'training_logs.csv'
    fieldnames = ['epoch', 'train_loss', 'valid_loss', 'time', 'accuracy', 'precision', 'recall', 'f1']
    if not torch.cuda.is_available() and not os.path.exists(csv_file_path):
        with open(csv_file_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    start_full_time = time.time()
    #---------------------------------TRAIN--------------------------
    accuracy_metric = Accuracy(task="multiclass", num_classes=6).to(device)
    precision_metric = Precision(task="multiclass", num_classes=6).to(device)
    recall_metric = Recall(task="multiclass", num_classes=6).to(device)
    f1_metric = F1Score(task="multiclass", num_classes=6).to(device)

    metrics = dict(
        accuracy = accuracy_metric,
        precision = precision_metric,
        recall = recall_metric,
        f1 = f1_metric
    )

    for epoch in range(args.epochs):
        start_time = time.time()
        total_train_loss = 0

        for v in metrics.values():
            v.reset()

        adjust_learning_rate(optimizer, epoch)

        train_sampler.set_epoch(epoch)
        for batch_idx, (left, right, disp, cls) in enumerate(TrainDataLoader):
            # if batch_idx <=1:
            #     gdal_array.SaveArray(left[0].detach().cpu().numpy(), f"train-left-{epoch}.tif")
            #     gdal_array.SaveArray(right[0].detach().cpu().numpy(), f"train-right-{epoch}.tif")
            #     gdal_array.SaveArray(cls[0].detach().cpu().numpy(), f"train-cls-{epoch}.tif")
  
            loss = train(left, right, disp, cls, model, device, optimizer)
            loss_backup = loss.clone()
            loss = dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            total_train_loss += loss_backup
        
        if local_rank == 0:
            savefilename = args.savepath+'/train_checkpoint_'+str(epoch)+'.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.module.state_dict(),
                        'train_loss': total_train_loss.item()/len(TrainDataLoader),
            }, savefilename)

        
    #---------------------------------valid--------------------------

        total_valid_loss = 0
        valid_sampler.set_epoch(epoch)
        for batch_idx, (left, right, disp, cls) in enumerate(VaildDataLoader):
            
            
            loss = valid(metrics, left, right, disp, cls, model, device, optimizer, epoch, False)
            loss_backup = loss.clone()
            loss = dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            total_valid_loss += loss_backup
        
        metrics_dict = {}
        for k,v in metrics.items():
            metrics_dict[k] = v.compute().item()

        if local_rank == 0:
            savefilename = args.savepath+'/valid_checkpoint_'+str(epoch)+'.tar'
            torch.save({
                'valid_loss': total_valid_loss/len(VaildDataLoader),
                **metrics_dict
                }, savefilename)
        

        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'epoch': epoch,
                'train_loss': total_train_loss.item()/len(TrainDataLoader),
                'valid_loss': total_valid_loss/len(VaildDataLoader), 
                'time': time.time() - start_time,
                **metrics_dict
            })
        
        
        print(f"epoch: {epoch}, training loss: {total_train_loss/len(TrainDataLoader)}, validing loss: {total_valid_loss/len(VaildDataLoader)}, time: {time.time()-start_time}", flush=True)

    print('full time = %.2f HR' %((time.time() - start_full_time)/3600), flush=True)

if __name__ == "__main__":
    main()
