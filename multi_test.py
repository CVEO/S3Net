import time
import argparse
from osgeo import gdal, gdal_array
import torch
import torchvision.transforms as transforms
import random
from torch.multiprocessing import Process
from torch.nn.parallel import DistributedDataParallel
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torchmetrics import ConfusionMatrix
from model.model import SSNet
import torch.utils.data

from data.data import *
import data.DFC2019Loader as DA

import numpy as np


def get_transform(data):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]} 
    data = torch.from_numpy(data).float()
    transform = transforms.Compose([transforms.Normalize(**normal_mean_var)])
    return transform(data).float()

def test(left, right, model, device):
    model.eval()

    left = torch.tensor(left, device=device).float()
    right = torch.tensor(right, device=device).float()

    with torch.no_grad():

        disp1, disp2, disp3, cls = model(left, right)

    pred_disp = disp3.data.cpu().numpy()
    cls = torch.softmax(cls[0], dim=0)
    pred_label = torch.max(cls, dim=0)[1]

    pred_cls = pred_label.data.detach().cpu().numpy()
    
    return pred_disp, pred_cls

def calculate_tp_fp_fn(pre_cls, true_cls, pre_disp, ori_disp, tp, fp, fn, tp_3, correct_count, err_count, count):
    mask = (ori_disp != -999)
    error = torch.abs(pre_disp - ori_disp)
    correct = error < 3

    correct_count += torch.sum(correct[mask])
    err_count += torch.sum(error[mask])
    count += torch.sum(mask)
    valid_disparity = (ori_disp == -999) | (true_cls == 3) | correct

    for i in range(5):
        tp[i] += torch.sum((pre_cls==i) & (true_cls==i))
        fp[i] += torch.sum((pre_cls==i) & (true_cls!=i))
        fn[i] += torch.sum((pre_cls!=i) & (true_cls==i))
        # tn[i] += torch.sum((y_pred!=i) & (y_true!=i))
        tp_3[i] += torch.sum((pre_cls==i) & (true_cls==i) & valid_disparity)




def main(data, model, device, cls_output, disp_output):

    left_images, right_images, cls_images, dsp_images, ori_images = data
    correct_count = torch.zeros(1).to(device)
    err_count = torch.zeros(1).to(device)
    count = torch.zeros(1).to(device)
    tp = torch.zeros(5, dtype=int).to(device)
    fp = torch.zeros(5, dtype=int).to(device)
    fn = torch.zeros(5, dtype=int).to(device)
    # tn = torch.zeros(5, dtype=int).to(device)
    tp_3 = torch.zeros(5, dtype=int).to(device)
    # fp_3 = torch.zeros(5, dtype=int).to(device)
    # fn_3 = torch.zeros(5, dtype=int).to(device)

    for left, right, cls, disp, ori_disp in zip(left_images, right_images, cls_images, dsp_images, ori_images):
        imgL = gdal.Open(left).ReadAsArray() / 255.0
        imgR = gdal.Open(right).ReadAsArray() / 255.0
        imgL = get_transform(imgL)
        imgR = get_transform(imgR)
        imgL = imgL.unsqueeze(0)
        imgR = imgR.unsqueeze(0)

        pred_disp, pred_cls = test(imgL, imgR, model, device)

        gdal_array.SaveArray(pred_disp, os.path.join(disp_output, os.path.basename(disp)))
        gdal_array.SaveArray(pred_cls, os.path.join(cls_output, os.path.basename(cls)))

        true_cls = torch.from_numpy(gdal.Open(cls).ReadAsArray()).long().to(device)
        pre_cls = pre_cls.to(device)
        pre_disp = torch.from_numpy(pred_disp).squeeze(0).to(device)
        ori_disp = torch.from_numpy(gdal.Open(ori_disp).ReadAsArray()).to(device)

        calculate_tp_fp_fn(pre_cls, true_cls, pre_disp, ori_disp, tp, fp, fn, tp_3, correct_count, err_count, count)

    # np.savez(f'{str(a)}.npy',
    #         tp=tp[0:5].cpu().numpy(),
    #         fp=fp[0:5].cpu().numpy(),
    #         fn=fn[0:5].cpu().numpy(),
    #         tn=tn[0:5].cpu().numpy(),
    #         tp_3=tp_3[0:5].cpu().numpy(),
    #         fp_3=fp_3[0:5].cpu().numpy(),
    #         fn_3=fn_3[0:5].cpu().numpy(),
    #         correct_count=correct_count.cpu().numpy(),
    #         total_count=count.cpu().numpy(),
    #         error_count=err_count.cpu().numpy()) 
    return tp[0:5], fp[0:5], fn[0:5], tp_3[0:5], correct_count, err_count, count


if __name__ == "__main__":
    main()
