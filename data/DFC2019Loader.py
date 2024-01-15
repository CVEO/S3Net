from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
import torch
from osgeo import gdal
import numpy as np

def get_transform(data):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]} 
    data = torch.from_numpy(data).float()
    transform = transforms.Compose([transforms.Normalize(**normal_mean_var)])
    return transform(data).float()

def dataloader(path):
    return gdal.Open(path, gdal.GA_ReadOnly)

class myImageFloder(Dataset):
    def __init__(self, left, right, disp, cls, training, dataloader=dataloader):
 
        self.left = left
        self.right = right
        self.disp = disp
        self.cls = cls
        self.training = training
        self.dataloader = dataloader

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp = self.disp[index]
        cls = self.cls[index]


        if self.training:

            w = self.dataloader(left).RasterXSize
            h = self.dataloader(left).RasterYSize
            th, tw = 512, 512
            off_x = random.randint(0, w - tw)
            off_y = random.randint(0, h - th)
            left = self.dataloader(left).ReadAsArray(off_x, off_y, tw, th) / 255.0
            right = self.dataloader(right).ReadAsArray(off_x, off_y, tw, th) / 255.0
            disp = self.dataloader(disp).ReadAsArray(off_x, off_y, tw, th)
            cls = self.dataloader(cls).ReadAsArray(off_x, off_y, tw, th)


            if random.random() > 0.5:
                left = np.flip(left, axis=1).copy()
                right = np.flip(right, axis=1).copy()
                disp = np.flip(disp, axis=0).copy()
                cls = np.flip(cls, axis=0).copy()

            if random.random() > 0.5:
                left = np.flip(left, axis=2).copy()
                right = np.flip(right, axis=2).copy()
                disp = -np.flip(disp, axis=1).copy()
                cls = np.flip(cls, axis=1).copy()


            # random gamma correction: gamma = base + channel, base in [0.5, 1.5], channel for [-0.2, 0.2]
            gamma_base = np.random.random() + 0.5
            gamma = np.random.random([left.shape[0],]) * 0.4 - 0.2 + gamma_base
            left = left ** gamma[:,None,None]
            right = right ** gamma[:,None,None]

        else:
            left = self.dataloader(left).ReadAsArray() / 255.0
            right = self.dataloader(right).ReadAsArray() / 255.0
            disp = self.dataloader(disp).ReadAsArray()
            cls = self.dataloader(cls).ReadAsArray()


        left = get_transform(left)
        right = get_transform(right)


        return left, right, disp, cls
        
    def __len__(self):
        return len(self.left)
    

class testdata(Dataset):
    def __init__(self, left, right, disp, cls, training, dataloader=dataloader):
 
        self.left = left
        self.right = right
        self.disp = disp
        self.cls = cls
        self.training = training
        self.dataloader = dataloader

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp = self.disp[index]
        cls = self.cls[index]




        left = self.dataloader(left).ReadAsArray() / 255.0
        right = self.dataloader(right).ReadAsArray() / 255.0
        disp = self.dataloader(disp).ReadAsArray()
        cls = self.dataloader(cls).ReadAsArray()

        left = get_transform(left)
        right = get_transform(right)
        
        return left, right, disp, cls
        
    def __len__(self):
        return len(self.left)
