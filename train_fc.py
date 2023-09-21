"""
训练新添加的全连接层，设置feature_extract = True
使用源模型的特征提取参数
"""

import warnings

import cv2
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler, random_split, Subset
from torchvision import datasets

from utils import *

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 是否在GPU上训练
if torch.cuda.is_available():
    print('use GPU.')
else:
    print('use CPU.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # 加载训练模型框架
    resnet_model = initialize_model()
    resnet_model = resnet_model.to(device)

    # 使用双卡训练
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        resnet_model = nn.DataParallel(resnet_model)
        resnet_model.to(device)

    # 优化器与损失函数设置（取默认值：betas=[0.9, 0.999], eps=1e-8）
    optimizer = torch.optim.Adam(resnet_model.parameters(), lr=args.lr)
    # 学习率衰减:每step_size个epoch之后，学习率衰减为原来的gamma倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    # 由于最后一层已经使用了LogSoftmax()，故交叉熵就等价于这样计算
    criterion = nn.NLLLoss()

    # 标准的预处理输入图像
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(args.resize_size),
                                         transforms.CenterCrop(args.input_size),
                                         # 这里是自行计算的GTSRB训练集的均值和方差
                                         transforms.Normalize(mean=[0.3402, 0.3121, 0.3214],
                                                              std=[0.1681, 0.1683, 0.1785])
                                         ])
    # Resize, normalize and jitter image brightness
    data_jitter_brightness = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize(args.resize_size),
                                                 transforms.CenterCrop(args.input_size),
                                                 transforms.ColorJitter(brightness=5),
                                                 # 这里是自行计算的GTSRB训练集的均值和方差
                                                 transforms.Normalize(mean=[0.3402, 0.3121, 0.3214],
                                                                      std=[0.1681, 0.1683, 0.1785])
                                                 ])

    # Resize, normalize and jitter image saturation
    data_jitter_saturation = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Resize(args.resize_size),
                                                 transforms.CenterCrop(args.input_size),
                                                 transforms.ColorJitter(saturation=5),
                                                 # 这里是自行计算的GTSRB训练集的均值和方差
                                                 transforms.Normalize(mean=[0.3402, 0.3121, 0.3214],
                                                                      std=[0.1681, 0.1683, 0.1785])
                                                 ])

    # Resize, normalize and jitter image contrast
    data_jitter_contrast = transforms.Compose([transforms.ToTensor(),
                                               transforms.Resize(args.resize_size),
                                               transforms.CenterCrop(args.input_size),
                                               transforms.ColorJitter(contrast=5),
                                               # 这里是自行计算的GTSRB训练集的均值和方差
                                               transforms.Normalize(mean=[0.3402, 0.3121, 0.3214],
                                                                    std=[0.1681, 0.1683, 0.1785])
                                               ])

    # Resize, normalize and jitter image hues
    data_jitter_hue = transforms.Compose([transforms.ToTensor(),
                                          transforms.Resize(args.resize_size),
                                          transforms.CenterCrop(args.input_size),
                                          transforms.ColorJitter(hue=0.4),
                                          # 这里是自行计算的GTSRB训练集的均值和方差
                                          transforms.Normalize(mean=[0.3402, 0.3121, 0.3214],
                                                               std=[0.1681, 0.1683, 0.1785])
                                          ])

    # Resize, normalize and rotate image
    data_rotate = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(args.resize_size),
                                      transforms.CenterCrop(args.input_size),
                                      transforms.RandomRotation(15),
                                      # 这里是自行计算的GTSRB训练集的均值和方差
                                      transforms.Normalize(mean=[0.3402, 0.3121, 0.3214],
                                                           std=[0.1681, 0.1683, 0.1785])
                                      ])

    # Resize, normalize and flip image horizontally and vertically
    data_hvflip = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize(args.resize_size),
                                      transforms.CenterCrop(args.input_size),
                                      transforms.RandomHorizontalFlip(1),
                                      transforms.RandomVerticalFlip(1),
                                      # 这里是自行计算的GTSRB训练集的均值和方差
                                      transforms.Normalize(mean=[0.3402, 0.3121, 0.3214],
                                                           std=[0.1681, 0.1683, 0.1785])
                                      ])

    # Resize, normalize and shear image
    data_shear = transforms.Compose([transforms.ToTensor(),
                                     transforms.Resize(args.resize_size),
                                     transforms.CenterCrop(args.input_size),
                                     transforms.RandomAffine(degrees=15, shear=2),
                                     # 这里是自行计算的GTSRB训练集的均值和方差
                                     transforms.Normalize(mean=[0.3402, 0.3121, 0.3214],
                                                          std=[0.1681, 0.1683, 0.1785])
                                     ])

    # Resize, normalize and translate image
    data_translate = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize(args.resize_size),
                                         transforms.CenterCrop(args.input_size),
                                         transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                                         # 这里是自行计算的GTSRB训练集的均值和方差
                                         transforms.Normalize(mean=[0.3402, 0.3121, 0.3214],
                                                              std=[0.1681, 0.1683, 0.1785])
                                         ])

    # 训练数据集与验证数据集
    dataset = torch.utils.data.ConcatDataset([datasets.ImageFolder(args.train_path,
                                                                   transform=data_transform),
                                              datasets.ImageFolder(args.train_path,
                                                                   transform=data_jitter_brightness),
                                              datasets.ImageFolder(args.train_path,
                                                                   transform=data_jitter_hue),
                                              datasets.ImageFolder(args.train_path,
                                                                   transform=data_jitter_contrast),
                                              datasets.ImageFolder(args.train_path,
                                                                   transform=data_jitter_saturation),
                                              datasets.ImageFolder(args.train_path,
                                                                   transform=data_translate),
                                              datasets.ImageFolder(args.train_path,
                                                                   transform=data_rotate),
                                              datasets.ImageFolder(args.train_path,
                                                                   transform=data_hvflip),
                                              datasets.ImageFolder(args.train_path,
                                                                   transform=data_shear)])
    # 制定数据集比例
    train_ratio = 0.8
    valid_ratio = 1 - train_ratio

    # 计算划分的索引边界
    num_samples = len(dataset)
    indices = list(range(num_samples))
    split_train = int(np.floor(train_ratio * num_samples))
    split_valid = int(np.floor(valid_ratio * num_samples))

    # 随机打乱索引顺序
    np.random.shuffle(indices)

    # 划分训练集和验证集的索引
    train_indices, valid_indices = indices[:split_train], indices[split_train:split_train + split_valid]

    # 创建两个SubsetRandomSampler对象，分别用于训练集和验证集
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)

    print(f"train_dataset_len = {len(train_indices)}, valid_dataset_len = {len(valid_indices)}")

    # 使用 DataLoader 加载训练集和验证集
    train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size)
    valid_loader = DataLoader(dataset, sampler=valid_sampler, batch_size=args.batch_size)

    # 开始训练，参数设置
    train(net=resnet_model,
          train_data=train_loader,
          valid_data=valid_loader,
          optimizer=optimizer,
          scheduler=scheduler,
          criterion=criterion)


if __name__ == '__main__':
    main()
