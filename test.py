import cv2
import torch
import numpy as np
import torchvision.transforms as transforms

from utils import parser
from models.model import SSNet


def get_transform(data):
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                        'std': [0.229, 0.224, 0.225]} 
    data = torch.from_numpy(data).float()
    transform = transforms.Compose([transforms.Normalize(**normal_mean_var)])
    return transform(data).float()


def eval():
    args = parser()

    model = SSNet(args.maxdisp, args.mindisp, args.classfication)
    model.load_state_dict(torch.load(args.ckpt)['state_dict'])
    model.eval().cuda()

    # Dataset loading
    left_path = 'xxxx'
    right_path = 'xxxx'

    left = cv2.imread(left_path, cv2.IMREAD_UNCHANGED).astype('float32')
    right = cv2.imread(right_path, cv2.IMREAD_UNCHANGED).astype('float32')

    left = np.moveaxis(left, -1, 0) / 255.0
    right = np.moveaxis(right, -1, 0) / 255.0

    left = get_transform(left).unsqueeze(0).float().cuda()
    right = get_transform(right).unsqueeze(0).float().cuda()

    # Inference
    with torch.no_grad():
        _, _, pred_disp, pred_cls = model(left, right)
        cv2.imwrite('pred_disp.tif', pred_disp.squeeze().cpu().numpy().astype(np.float32))
        pred_cls_np = torch.argmax(pred_cls, dim=1).squeeze().cpu().numpy().astype(np.uint8)
        cv2.imwrite('pred_cls.tif', pred_cls_np)

if __name__ == "__main__":
    eval()
