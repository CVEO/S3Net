import time
import argparse
from osgeo import gdal, gdal_array
import torch
import torchvision.transforms as transforms
import random
from torch.multiprocessing import Process
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
# from torchmetrics import Accuracy, Precision, Recall, F1Score
from model.model import SSNet
import torch.utils.data
from multi_test import main as mm
from data.data import *
import data.DFC2019Loader as DA

import numpy as np



def main():

    dir_name = "./"
    cls_output = "cls_test_output"
    disp_output = "disp_test_output"
    if not os.path.exists(cls_output):
        os.mkdir(cls_output)
    if not os.path.exists(disp_output):
        os.mkdir(disp_output)

    init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    print("local_rank", local_rank, flush=True)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)


    model = SSNet(48, 6)
    model.to(device)

    state_dict = torch.load("./model/ckpt.tar")
    model.load_state_dict(state_dict['state_dict'])
                          
    print("Let's use", torch.cuda.device_count(), "GPUs!", flush=True)
    model = DDP(model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True)
    

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


    left_images = [os.path.abspath(os.path.join(dir_name, "test_data", "test_left", img)) for img in os.listdir(os.path.join(dir_name, "test_data","test_left"))]
    right_images = [os.path.abspath(os.path.join(dir_name, "test_data","test_right", os.path.basename(img).replace("LEFT_RGB", "RIGHT_RGB"))) for img in left_images]
    cls_images = [os.path.abspath(os.path.join(dir_name, "test_data","test_cls", os.path.basename(img).replace("LEFT_RGB", "LEFT_CLS"))) for img in left_images]
    dsp_images = [os.path.abspath(os.path.join(dir_name, "test_data","test_dsp", os.path.basename(img).replace("LEFT_RGB", "LEFT_DSP"))) for img in left_images]
    ori_dsp_images = [os.path.abspath(os.path.join(dir_name, "test_data","test_ori_dsp", os.path.basename(img).replace("LEFT_RGB", "LEFT_DSP"))) for img in left_images]

    # all_iou = np.zeros(6)
    # all_num = np.zeros(6)

    if local_rank == 0:
        device = torch.cuda.current_device()
        a = local_rank
        data = (left_images[:113], right_images[:113], cls_images[:113], dsp_images[:113], ori_dsp_images[:113], a)
       
 
    elif local_rank == 1:    
        device = torch.cuda.current_device()
        a = local_rank
        data =(left_images[113:226], right_images[113:226], cls_images[113:226], dsp_images[113:226], ori_dsp_images[113:226], a)
        # iou_per_class, num = m(, model, device, output_name1, output_name2) 
      
    elif local_rank == 2:
        device = torch.cuda.current_device()
        a = local_rank
        data = (left_images[226:339], right_images[226:339], cls_images[226:339], dsp_images[226:339], ori_dsp_images[226:339], a)
        # iou_per_class, num = m( model, device, output_name1, output_name2)
        
    elif local_rank == 3:
        device = torch.cuda.current_device()
        a = local_rank
        data = (left_images[339:], right_images[339:], cls_images[339:], dsp_images[339:], ori_dsp_images[339:], a)
        # iou_per_class, num = m(, model, device, output_name1, output_name2)
       
    tp, fp, fn, tp_3, correct_count, err_count, count = mm(data, model, device, cls_output, disp_output)
    dist.all_reduce(tp, op=dist.ReduceOp.SUM)
    dist.all_reduce(fp, op=dist.ReduceOp.SUM)
    dist.all_reduce(fn, op=dist.ReduceOp.SUM)
    # dist.all_reduce(tn, op=dist.ReduceOp.SUM)
    dist.all_reduce(tp_3, op=dist.ReduceOp.SUM)
    dist.all_reduce(correct_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    dist.all_reduce(err_count, op=dist.ReduceOp.SUM)
    if local_rank == 0:
        iou = tp/(tp+fp+fn)
        miou = torch.mean(iou)
        iou_3 = tp_3/(tp_3+fp+fn)
        miou_3 = torch.mean(iou_3)
        D1 = (1-(correct_count/count))*100
        EPE = err_count/count
        print("iou, miou, iou_3, miou_3, D1, EPE", iou, miou, iou_3, miou_3, D1, EPE, flush=True)
    # dist.barrier()


    # all_iou = dist.reduce(cfm, op=dist.ReduceOp.SUM, dst=0)
    # all_num = dist.reduce(num, op=dist.ReduceOp.SUM, dst=0)


    # if local_rank == 0:
    #     ious = []
    #     for i in range(6):
    #         I = all_iou[i,i]
    #         U = np.sum(all_iou[i,:])  + np.sum(all_iou[:,i]) - I
    #         iou = I / (U + 1e-6)
    #         ious.append(iou)
    #     print(ious)
    #     print(np.mean(ious))
    # dist.destroy_process_group()
if __name__ == "__main__":
    main()
    # time.sleep(40)
    # m1 = np.load('0.npy.npz')
    # m2 = np.load('1.npy.npz')
    # m3 = np.load('2.npy.npz')
    # m4 = np.load('3.npy.npz')
    # print("===========================", flush=True)
    # for key, value in m1.items():
    #     print(f"{key}: {value}", flush=True)
    # for key, value in m2.items():
    #     print(f"{key}: {value}", flush=True)
    # for key, value in m3.items():
    #     print(f"{key}: {value}", flush=True)
    # for key, value in m4.items():
    #     print(f"{key}: {value}", flush=True)
    # print("----------------------------", flush=True)
    # m = sum([m1["iou"],m2["iou"],m3["iou"],m4["iou"]])

    # ious = []
    # for i in range(5):
    #     I = m[i,i]
    #     U = np.sum(m[i,:])  + np.sum(m[:,i]) - I
    #     iou = I / (U + 1e-6)
    #     ious.append(iou)

    # print(ious, flush=True)
    # print(np.mean(ious), flush=True)


    # D1 = (1-sum([m1["correct_count"],m2["correct_count"],m3["correct_count"],m4["correct_count"]])/
    #     sum([m1["total_count"],m2["total_count"],m3["total_count"],m4["total_count"]]))*100
    # EPE = (sum([m1["error_count"],m2["error_count"],m3["error_count"],m4["error_count"]])/
    #         sum([m1["total_count"],m2["total_count"],m3["total_count"],m4["total_count"]]))
    #    m = sum([m1["iou"],m2["iou"],m3["iou"],m4["iou"]])

    # iou = sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])/(sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])+\
    #                                              sum([m1["fp"],m2["fp"],m3["fp"],m4["fp"]])+\
    #                                              sum([m1["fn"],m2["fn"],m3["fn"],m4["fn"]]))
    # iou_3 = sum([m1["tp_3"],m2["tp_3"],m3["tp_3"],m4["tp_3"]])/(sum([m1["tp_3"],m2["tp_3"],m3["tp_3"],m4["tp_3"]])+\
    #                                             sum([m1["fp"],m2["fp"],m3["fp"],m4["fp"]])+\
    #                                             sum([m1["fn"],m2["fn"],m3["fn"],m4["fn"]]))
    
    # # iou_3_1 = sum([m1["tp_3"],m2["tp_3"],m3["tp_3"],m4["tp_3"]])/(sum([m1["tp_3"],m2["tp_3"],m3["tp_3"],m4["tp_3"]])+\
    # #                                         sum([m1["fp"],m2["fp"],m3["fp"],m4["fp"]])+\
    # #                                         sum([m1["fn"],m2["fn"],m3["fn"],m4["fn"]])+1)

    # print("iou", iou, flush=True)
    # print(np.mean(iou), flush=True)

    # print("iou_3", iou_3, flush=True)
    # print(np.mean(iou_3), flush=True)

    # # print("iou_3_1", iou_3_1, flush=True)
    # # print(np.mean(iou_3_1), flush=True)

    # accuracy = (sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])+sum([m1["tn"],m2["tn"],m3["tn"],m4["tn"]]))/(sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])+\
    #                                              sum([m1["fp"],m2["fp"],m3["fp"],m4["fp"]])+\
    #                                              sum([m1["fn"],m2["fn"],m3["fn"],m4["fn"]])+\
    #                                              sum([m1["tn"],m2["tn"],m3["tn"],m4["tn"]]))
    # recall = sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])/(sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])+\
    #                                                      sum([m1["fn"],m2["fn"],m3["fn"],m4["fn"]]))

    # precision = sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])/(sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])+\
    #                                                     sum([m1["fp"],m2["fp"],m3["fp"],m4["fp"]]))

    # f1 = 2*precision*recall/(precision+recall)
    # print("cls", accuracy, recall, precision, f1, flush=True)


    # all_accuracy = np.sum((sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])+sum([m1["tn"],m2["tn"],m3["tn"],m4["tn"]])))/np.sum((sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])+\
    #                                              sum([m1["fp"],m2["fp"],m3["fp"],m4["fp"]])+\
    #                                              sum([m1["fn"],m2["fn"],m3["fn"],m4["fn"]])+\
    #                                              sum([m1["tn"],m2["tn"],m3["tn"],m4["tn"]])))
    # all_recall = np.sum(sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]]))/np.sum((sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])+\
    #                                                      sum([m1["fn"],m2["fn"],m3["fn"],m4["fn"]])))

    # all_precision = np.sum(sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]]))/np.sum((sum([m1["tp"],m2["tp"],m3["tp"],m4["tp"]])+\
    #                                                     sum([m1["fp"],m2["fp"],m3["fp"],m4["fp"]])))

    # all_f1 = 2*all_precision*all_recall/(all_precision+all_recall)
    # print("all", all_accuracy, all_recall, all_precision, all_f1, flush=True)

    # D1 = (1-sum([m1["correct_count"],m2["correct_count"],m3["correct_count"],m4["correct_count"]])/
    #     sum([m1["total_count"],m2["total_count"],m3["total_count"],m4["total_count"]]))*100
    # EPE = (sum([m1["error_count"],m2["error_count"],m3["error_count"],m4["error_count"]])/
    #         sum([m1["total_count"],m2["total_count"],m3["total_count"],m4["total_count"]]))
    # print(D1, EPE, flush=True)
