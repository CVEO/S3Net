import os
import cv2
import torch
import random
import torchvision.transforms as transforms
import torch.distributed as dist
import numpy as np
from torch.utils.data import Dataset

class DataLoader:
    @staticmethod
    def load(datapath: str, dataset: str) -> tuple:
        def _get_image_paths(subdir):
            base_path = os.path.join(datapath, subdir)
            return [os.path.join(base_path, img) for img in os.listdir(base_path)]

        left_train = _get_image_paths("left")
        right_train = [path.replace("LEFT_RGB", "RIGHT_RGB").replace("left", "right") for path in left_train]
        disp_train = [path.replace("LEFT_RGB", "LEFT_DSP").replace("left", "disp") for path in left_train]
        cls_train = [path.replace("LEFT_RGB", "LEFT_CLS").replace("left", "cls") for path in left_train]

        left_valid = _get_image_paths("valid_left")
        right_valid = [path.replace("LEFT_RGB", "RIGHT_RGB").replace("left", "right") for path in left_valid]
        disp_valid = [path.replace("LEFT_RGB", "LEFT_DSP").replace("left", "disp") for path in left_valid]
        cls_valid = [path.replace("LEFT_RGB", "LEFT_CLS").replace("left", "cls") for path in left_valid]


        train_data = (left_train, right_train, disp_train, cls_train)
        valid_data = (left_valid, right_valid, disp_valid, cls_valid)

        return train_data, valid_data

class StereoDataset(Dataset):
    def __init__(self, left_images, right_images, disp_images, cls_images, training=True):
        self.left = left_images
        self.right = right_images
        self.disp = disp_images
        self.cls = cls_images
        self.training = training

    def __len__(self):
        return len(self.left)

    def get_transform(self, data):
        normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                            'std': [0.229, 0.224, 0.225]} 
        data = torch.from_numpy(data).float()
        transform = transforms.Compose([transforms.Normalize(**normal_mean_var)])
        return transform(data).float()

    def _augment(self, left, right, disp, cls):

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

        _, h, w = left.shape
        x = random.randint(0, w - 512)
        y = random.randint(0, h - 512)
        left = left[:, y:y+512, x:x+512].copy()
        right = right[:, y:y+512, x:x+512].copy()
        cls = cls[y:y+512, x:x+512].copy()
        disp = disp[y:y+512, x:x+512].copy()

        return left, right, disp, cls

    def __getitem__(self, index):
        left = self._read_image(self.left[index])
        right = self._read_image(self.right[index])
        disp = self._read_image(self.disp[index], is_disp_cls=True)
        cls = self._read_image(self.cls[index], is_disp_cls=True)

        if self.training:
            left, right, disp, cls = self._augment(left, right, disp, cls)

        left = self.get_transform(left)
        right = self.get_transform(right)

        return left, right, disp, cls

    def _read_image(self, path, is_disp_cls=False):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype('float32')

        if len(img.shape) == 3:
            img = np.moveaxis(img, -1, 0) / 255.0
            return img

        if is_disp_cls:
            return img


def generate(dataset, datapath):
    
    train_data, valid_data = DataLoader.load(datapath, dataset)
    
    train_dataset = StereoDataset(
        left_images=train_data[0], 
        right_images=train_data[1], 
        disp_images=train_data[2], 
        cls_images=train_data[3],
        training=True
    )
    
    valid_dataset = StereoDataset(
        left_images=valid_data[0], 
        right_images=valid_data[1], 
        disp_images=valid_data[2], 
        cls_images=valid_data[3],
        training=False
    )
    
    return train_dataset, valid_dataset


def initialize_dataloaders(args, train_dataset, valid_dataset):
    if args.is_distributed:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        valid_sampler = torch.utils.data.DistributedSampler(valid_dataset, num_replicas=dist.get_world_size(),
                                                            rank=dist.get_rank())
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            sampler=train_sampler, pin_memory=True)

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
            sampler=valid_sampler, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                                   shuffle=True, num_workers=args.num_workers, drop_last=False)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size,
                                                   shuffle=False, num_workers=args.num_workers, drop_last=False)
    return train_loader, valid_loader