import os
import shutil
from osgeo import gdal_array, gdal

class Datalist:
    def __init__(self, data):
        self.left_images, self.right_images, self.dsp_images, self.cls_images = data

    def forward(self):
        return self.left_images, self.right_images, self.dsp_images, self.cls_images

def Dataload(datapath: str, num: int) -> Datalist:
    train = num * 5
    # valid = num * 5

    left_images = [os.path.abspath(os.path.join(datapath, "left", img)) for img in os.listdir(os.path.join(datapath, "left"))]
    right_images = [os.path.abspath(os.path.join(datapath, "right", os.path.basename(img).replace("LEFT_RGB", "RIGHT_RGB"))) for img in left_images]
    dsp_images = [os.path.abspath(os.path.join(datapath, "new_dsq", os.path.basename(img).replace("LEFT_RGB", "LEFT_DSP"))) for img in left_images]
    cls_images = [os.path.abspath(os.path.join(datapath, "cls", os.path.basename(img).replace("LEFT_RGB", "LEFT_CLS"))) for img in left_images]

    train_data = Datalist((left_images[:train], right_images[:train], dsp_images[:train], cls_images[:train]))
    valid_data = Datalist((left_images[train:], right_images[train:], dsp_images[train:], cls_images[train:]))
    # test_data = Datalist((left_images[valid:], right_images[valid:], dsp_images[valid:], cls_images[valid:]))

    # destination_folder = os.path.join(datapath, "test_data")

    # subfolders = ['left', 'right', 'new_dsq', 'cls']
    # for folder in subfolders:
    #     os.makedirs(os.path.join(destination_folder, folder), exist_ok=True)

    # for left_path, right_path, dsp_path, cls_path in zip(test_data.left_images, test_data.right_images, test_data.dsp_images, test_data.cls_images):
    #     shutil.move(left_path, os.path.join(destination_folder, 'left', os.path.basename(left_path)))
    #     shutil.move(right_path, os.path.join(destination_folder, 'right', os.path.basename(right_path)))
    #     shutil.move(dsp_path, os.path.join(destination_folder, 'new_dsq', os.path.basename(dsp_path)))
    #     shutil.move(cls_path, os.path.join(destination_folder, 'cls', os.path.basename(cls_path)))

    return train_data, valid_data

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    Dataload(path, 700)
