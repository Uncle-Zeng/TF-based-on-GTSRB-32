import warnings

import cv2
import pandas as pd
import torch.nn.functional as F
from torchvision import transforms

from utils import *

# 忽略特定类型的警告
warnings.filterwarnings("ignore", category=UserWarning)

# 分别加载结构和参数
if torch.cuda.is_available():
    resnet_model = torch.load(args.model_structure_path)
    resnet_model_weights = torch.load(args.model_param_path)
    resnet_model.load_state_dict(resnet_model_weights)
    print("GPU加载模型")
else:
    resnet_model = torch.load(args.model_structure_path, map_location=torch.device('cpu'))
    resnet_model.load_state_dict(torch.load(args.model_param_path))
    print("CPU加载模型")

# 使用双卡
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    resnet_model = nn.DataParallel(resnet_model)
    resnet_model.to(device)

# 将模型切换到评估模式进行预测
resnet_model.eval()

# 预处理输入图像
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize(args.resize_size),
                                transforms.CenterCrop(args.input_size),
                                # 这里是GTSRB数据集的均值和方差
                                transforms.Normalize(mean=[0.3402, 0.3121, 0.3214],
                                                     std=[0.1681, 0.1683, 0.1785])
                                ])


def model(input):
    """
    用于分类的model，返回对抗样本输入后的模型预测的向量
    :param input: 需要分类的图像
    :return: 置信度向量 和 对应标签
    """

    # 输入图像为单张图像，增加一个batch维度
    input_img = transform(input).unsqueeze(0).to(device)

    output = resnet_model(input_img)
    output = torch.squeeze(output, dim=0)
    output = F.softmax(output, dim=0)
    label = output.argmax(dim=0, keepdim=True)

    return output, label


# 用于测试对训练数据进行分类的效果
if __name__ == "__main__":
    # 获取路径
    img_dir = args.test_path
    # 用于保存模型输出的标签、真实的标签列表、用于计算准确率
    label_list = []
    real_label_list = []
    counts = 0

    # 读取 Excel 表格文件
    df = pd.read_csv(args.test_label_path)
    # 指定要读取的列名
    columns_to_read = ['ClassId', 'Path']

    # 顺序读取每一行，取出其中指定的列
    for index, row in df.iterrows():
        cols_toread = row[columns_to_read].tolist()

        # 获取图像路径并读取图像
        img_path = "../GTSRB/" + cols_toread[1]
        img = cv2.imread(img_path)

        # 输入模型中
        output, label = model(img)
        # 输出当前的置信度和标签，并加入label_list中
        print(f"第{index + 1}行：{cols_toread}  output:{output.max():.4f}, label:{label.item()}")
        label_list.append(label.item())

        real_label_list.append(cols_toread[0])

        # 如果输出结果与标注结果一致，则+1
        if label_list[index] == real_label_list[index]:
            counts = counts + 1
        # 输出当前的准确率
        print(f"curr_pred_accuracy : {counts / (index + 1):.4f}")

    # 计算预测准确率
    # 似乎最后的index+1也能表示测试集图像的数量
    print(f"pred_accuracy : {counts / (index + 1):.4f}")
