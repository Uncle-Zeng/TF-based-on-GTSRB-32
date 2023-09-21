import os

import torch
import time
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models import resnet18, ResNet18_Weights

from configs import args

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset.
    """
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:, i, :, :].mean()
            std[i] += inputs[:, i, :, :].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def set_parameter_requires_grad(model):
    """
    用于设置需要进行参数更新的参数
    :param model: 使用的模型
    :return:
    """
    if args.feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model():
    """
    初始化resnet模型设置
    :return:
    """
    # 选择合适的模型，不同的模型初始化存在些许差异
    # 此处就只写初始化resnet34的相关代码
    # 加载训练模型
    model = resnet34(weights=ResNet34_Weights.DEFAULT)
    # 设置需要进行更新的参数
    set_parameter_requires_grad(model)
    # 获取全连接层前的输入个数
    num_ftrs = model.fc.in_features
    # 修改全连接层的输出神经元个数并使用HE初始化
    model.fc = nn.Sequential(nn.Linear(num_ftrs, args.num_classes), nn.LogSoftmax(dim=1))
    nn.init.kaiming_uniform_(model.fc[0].weight, mode='fan_in', nonlinearity='relu')
    nn.init.zeros_(model.fc[0].bias)

    # 打印需要更新的参数
    print("Prams to learn:")
    if args.feature_extract:
        # 使用预训练模型的特征提取
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print("\t", name)

    return model


def get_acc(output, label):
    # output 的形状为 (batch_size, num_classes)
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()

    return num_correct / total


def save_model(epoch, net):
    # Save checkpoint.
    print('Saving...')

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    # 保存模型
    checkpoint_path = 'checkpoints/model_{}.pth'.format(epoch + 1)
    checkpoint_param_path = 'checkpoints/model_param_{}.pth'.format(epoch + 1)

    # 保存模型时只保存主模型部分的状态字典
    if torch.cuda.device_count() > 1:
        torch.save(net.module.state_dict(), checkpoint_param_path)
    else:
        torch.save(net.state_dict(), checkpoint_path)



# 定义训练函数
def train(net, train_data, valid_data, optimizer, scheduler, criterion):
    start_time = time.time()  # 记录整个训练过程的起始时间

    # 设置早停（Early Stopping）的相关参数（当前最小损失、容忍度值、计数变量）
    best_loss = float('inf')
    best_acc = 0
    counter = 0
    tolerance = args.tolerance

    # 开始训练
    for epoch in range(args.epochs):
        epoch_start_time = time.time()  # 记录每一轮训练的起始时间
        step = 0
        net = net.train()
        net.to(device)

        # 输出当前学习率值
        for param_group in optimizer.param_groups:
            print(f"Current LR = {param_group['lr']}")
        # 学习率衰减
        scheduler.step()

        for im, label in train_data:
            # 将数据转至GPU进行计算
            im = im.to(device)
            label = label.to(device)

            # 前向传播
            output = net(im)
            loss = criterion(output, label)

            # 梯度清零、反向传播、参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算训练集损失和准确率
            train_loss = loss.item()
            train_acc = get_acc(output, label)
            step += 1

            # 每10个step进行一次输出
            if step % 10 == 0:
                print(f"epoch = {epoch + 1}, "
                      f"step = {step}, "
                      f"train_loss = {train_loss:.4f}, "
                      f"train_accuracy = {train_acc:.4f}  ")

        # 每一个epoch训练结束之后进行测试
        valid_loss = 0
        valid_acc = 0
        net = net.eval()
        for im, label in valid_data:
            # 将数据转至GPU进行计算
            im = im.to(device)
            label = label.to(device)

            # 前向传播、计算损失
            output = net(im)
            loss = criterion(output, label)

            # 这里计算的是损失加和 与 准确率加和，后面输出时计算准确率
            valid_loss += loss.item()
            valid_acc += get_acc(output, label)

        # 输出测试集的测试结果
        print(f"epoch:{epoch + 1}  "
              f"valid_loss:{valid_loss / len(valid_data):.4f}  "
              f"valid_accuracy:{valid_acc / len(valid_data):.4f}\n")
        
        # 创建txt文件并保存（训练过程记录）
        with open('checkpoints/train_info.txt', 'a') as file:
            file.write(f"params: epochs = {args.epochs}, batch_size:{args.batch_size} \n")
            file.write(f"epoch:{epoch + 1}  ")
            file.write(f"valid loss:{valid_loss / len(valid_data):.4f}  ")
            file.write(f"valid_accuracy:{valid_acc / len(valid_data):.4f}\n")

        epoch_end_time = time.time()  # 记录每一轮训练的结束时间
        epoch_time = epoch_end_time - epoch_start_time  # 计算每一轮训练的时间差
        print(f"Epoch {epoch+1} time: {epoch_time:.2f} seconds")

        # 判断验证集上的性能是否有改善：早停（early stopping）
        if valid_loss / len(valid_data) <= best_loss and valid_acc / len(valid_data) >= best_acc:
            best_loss = valid_loss / len(valid_data)
            best_acc = valid_acc / len(valid_data)
            counter = 0

            # Save checkpoint.
            save_model(epoch=epoch, net=net)

        elif valid_loss / len(valid_data) <= best_loss and valid_acc / len(valid_data) < best_acc:
            best_loss = valid_loss / len(valid_data)
            counter = 0
            print("模型性能存在下降可能性。\n")

            # Save checkpoint.
            save_model(epoch=epoch, net=net)

        elif valid_loss / len(valid_data) > best_loss and valid_acc / len(valid_data) >= best_acc:
            best_acc = valid_acc / len(valid_data)
            counter = 0
            print("模型性能存在下降可能性。\n")

            # Save checkpoint.
            save_model(epoch=epoch, net=net)

        else:
            counter = counter + 1
            print(f"counter = {counter}, 模型性能下降。\n")

        # 大于最大容忍次数时跳出训练循环
        if counter > tolerance:
            print(f"epoch {epoch + 1 - tolerance} may be the best one.")
            break
    end_time = time.time()  # 记录整个训练过程的结束时间
    total_time = end_time - start_time  # 计算整个训练过程的总时间
    print(f"Total training time: {total_time:.2f} seconds") 
