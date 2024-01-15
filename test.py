import time
import argparse
from osgeo import gdal, gdal_array
import torch
import torchvision.transforms as transforms
import random
from torch.multiprocessing import Process
from pathlib import Path
from torch import nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchmetrics import Accuracy, Precision, Recall, F1Score
from model.model import SSNet
import torch.utils.data

from data.data import *
import data.DFC2019Loader as DA

def get_transform(data):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]} 
    data = torch.from_numpy(data).float()
    transform = transforms.Compose([transforms.Normalize(**normal_mean_var)])
    return transform(data).float()

def test(left, right, model, device):
    start = time.time()
    model.train()

    left_tensor = torch.tensor(left, device=device).float()
    right_tensor = torch.tensor(right, device=device).float()

    with torch.no_grad():

        disp1, disp2, disp3, cls1 = model(left_tensor, right_tensor)
    print("all", time.time()-start, disp3.shape, cls1.shape)
    pred_disp = disp3.detach().cpu().numpy()
    cls1 = torch.softmax(cls1[0], dim=0)
    pred_label = torch.max(cls1, dim=0)[1]
    print(pred_label.shape)
    pred_label = pred_label.data.detach().cpu().numpy()
    
    return pred_disp, pred_label

def transform(img):
    img = img.GetRasterBand(1)
    nodata_value = img.GetNoDataValue()
    img = img.ReadAsArray(200, 200, 1024, 1024)

    valid = img != nodata_value
    avg = np.mean(img[valid])
    std = np.std(img[valid])
    maxEle = avg + 3 * std
    minEle = avg - 3 * std

    img1 = img[:]
    img1 = np.clip((img1.astype(np.float32) - minEle) /
            (maxEle - minEle + 0.00001), 0, 1) * 255. 
    
    img = img.astype(np.uint8)
    if nodata_value is not None:
        img1[img == nodata_value] = 255
    
    img = np.expand_dims(img, axis=0)
    img = np.concatenate([img, img, img], axis=0)
    print(img.shape)
    return img
    
def main():
    start_time = time.time()
    parser = argparse.ArgumentParser(description='S3Net')
    parser.add_argument('--DFC2019', default='2019', help='DFC2019')
    parser.add_argument('--ImgL', default="./data/dataset/", help='ImgL')
    parser.add_argument('--ImgR', default="./data/dataset/", help='ImgR')
    parser.add_argument('--epochs', default=120, type=int, help='train epoch')
    parser.add_argument('--maxdisp', default=48, help='maxium disparity')
    parser.add_argument('--model', default='SSNet', help='select model')
    parser.add_argument('--train_num', default=700, help='train number')
    parser.add_argument('--classfication', default=6, help='class number')
    parser.add_argument('--no_cuda', action='store_true', default=False, help='enables CUDA training')
    parser.add_argument('--savepath', default="./ckpt/", help='saveckpt')
    parser.add_argument('--output', default="./output/", help='save_output')
    parser.add_argument('--ckpt', default="./", help='ckpt')
    args = parser.parse_args()


    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    print(device, flush=True)


    if args.model == "SSNet":
        model = SSNet(args.maxdisp, args.classfication)
    else:
        raise ValueError("no model")


    # model = nn.DataParallel(model)
    model.to(device)
    
    state_dict = torch.load(args.ckpt)
    model.load_state_dict(state_dict['state_dict'])


    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    
    # w, h = 1024, 1024
    # th, tw = 256, 256
    # off_x = random.randint(0, w - tw)
    # off_y = random.randint(0, h - th)
    imgL = gdal.Open(args.ImgL).ReadAsArray()/255.0
    imgR = gdal.Open(args.ImgR).ReadAsArray()/255.0
    # imgL = gdal.Open(r"Z:\临时存放与文件传输\hongshan\GF701_004549_E114.5_N30.4_20200828111926_MUX_01_SC0_0004_2009012377.tif").ReadAsArray(2000, 2000, 1024, 1024)[:3]/2048.0
    # imgR = gdal.Open(r"Z:\临时存放与文件传输\hongshan\GF701_004549_E114.5_N30.4_20200828111926_MUX_01_SC0_0004_2009012377.tif").ReadAsArray(1998, 2000, 1024, 1024)[:3]/2048.0
    # imgL = imgL[[2,0,1]].copy()
    # imgR = imgR[[2,0,1]].copy()
    # gdal_array.SaveArray(imgL, "ori_left.tif")
    # gdal_array.SaveArray(imgR, "ori_right.tif")
    # imgL = gdal.Open(r"D:\浏览器\download\data_scene_flow\testing\image_2\000000_10.png").ReadAsArray(0, 0, 350, 350)/255.0
    # imgR = gdal.Open(r"D:\浏览器\download\data_scene_flow\testing\image_3\000000_10.png").ReadAsArray(0, 0, 350, 350)/255.0
    # imgL = transform(imgL)
    # imgR = transform(imgR)
    # gdal_array.SaveArray(imgL, "ori_left.tif")

    imgL = get_transform(imgL)
    imgR = get_transform(imgR)
    imgL = imgL.unsqueeze(0)
    imgR = imgR.unsqueeze(0)



    # from thop import profile
    # flops, params = profile(model, inputs =(imgL.cuda(), imgR.cuda()))
    # print('FLOPs = ' + str(flops/1000**3) + 'G')
    # print('Params = ' + str(params/1000**2) + 'M')

    pred_disp, pre_cls = test(imgL,imgR, model, device)
    print("time", time.time()-start_time)

    if not os.path.exists(args.output):
        os.makedirs(args.output)
    gdal_array.SaveArray(pred_disp, str(Path(args.output)/"disp.tif"))
    gdal_array.SaveArray(pre_cls, str(Path(args.output)/"cls.tif"))


if __name__ == "__main__":
    main()
