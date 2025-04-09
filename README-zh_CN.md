# S3Net

[English](./README.md) | 简体中文 

CVEO小组在IGARSS 2024学术研讨会上提交的论文"在卫星极线图像中使用创新的单分支语义立体网络(S3Net)进行立体匹配和语义分割"的开源代码

## 模型概述
### 框架
![model](./example/model.png)

### 实验结果
#### US3D测试集上的立体匹配结果
![cls](./example/table_disp.png)
![disp](./example/disp.png)

#### US3D测试集上的语义分割结果
![cls](./example/table_cls.png)
![cls](./example/cls.png)

## 使用说明
### 安装
```bash
git clone https://github.com/CVEO/S3Net.git
cd S3Net
conda env create -f environment.yml
conda activate s3net
```
### 数据集
本实验使用的数据集是[2019数据融合竞赛](https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019)中的US3D赛道2数据集。
### 预训练权重
[百度网盘](https://pan.baidu.com/s/1EHYTq4eBKVJXgeFTq8SYFQ?pwd=1111) : 1111 

[谷歌云盘](https://drive.google.com/file/d/1QrbsIir5FmKkZ2xlNL57AQKeQ7-vMubh/view?usp=drive_link)

## 训练启动方法

### 1. 单节点单GPU训练
```bash
python main.py
```

### 2. 单节点多GPU训练
```bash
torchrun --nproc_per_node=N main.py
```

### 3. 多节点多GPU训练

#### 启动命令
在主节点上：
```bash
torchrun --nproc_per_node=4 --nnodes=N --node_rank=0 --master_addr=MASTER_IP --master_port=PORT main.py
```

在其他节点上：
```bash
torchrun --nproc_per_node=4 --nnodes=N --node_rank=R --master_addr=MASTER_IP --master_port=PORT main.py
```

## 推理启动方法

使用test.py进行模型推理：
```bash
python test.py
```
## 文件目录说明
```
S3Net 
├── example
│   ├── cls.png
│   ├── disp.png
│   ├── model.png
│   ├── table_cls.png
│   └── table_disp.png
├── models
│   └── model.py
├── README-zh_CN.md
├── README.md
├── environment.yml
├── utils.py
├── train.py
├── test.py
├── main.py
└── data.py
```

## 最新工作
如果您对我们的最新工作感兴趣，欢迎查看我们的新项目 [TriGeoNet](https://github.com/CVEO/TriGeoNet)！

## 许可证
代码仅供非商业和研究目的使用。如需商业用途，请联系作者。

## 引用本工作
如果您觉得S3Net对您的研究有帮助，请考虑给个star ⭐ 并引用：
```
@inproceedings{yang2024s,
  title={S3Net: Innovating Stereo Matching and Semantic Segmentation with a Single-Branch Semantic Stereo Network in Satellite Epipolar Imagery},
  author={Yang, Qingyuan and Chen, Guanzhou and Tan, Xiaoliang and Wang, Tong and Wang, Jiaqi and Zhang, Xiaodong},
  booktitle={IGARSS 2024-2024 IEEE International Geoscience and Remote Sensing Symposium},
  pages={8737--8740},
  year={2024},
  organization={IEEE}
}
```

或引用旧版本S2Net：

```
@article{liao2023s,
  title={S2Net: A Multitask Learning Network for Semantic Stereo of Satellite Image Pairs},
  author={Liao, Puyun and Zhang, Xiaodong and Chen, Guanzhou and Wang, Tong and Li, Xianwei and Yang, Haobo and Zhou, Wenlin and He, Chanjuan and Wang, Qing},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  volume={62},
  pages={1--13},
  year={2023},
  publisher={IEEE}
}
```