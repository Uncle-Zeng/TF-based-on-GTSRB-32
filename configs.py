import argparse

# 使用python内置的argparse来进行参数设置，也方便调参.
# 使用ArgumentParser对象进行初始化
parser = argparse.ArgumentParser()

parser.add_argument('--num_classes', type=int, default=43,
                    help='迁移学习时最终的全连接层神经元个数（输出个数）')
parser.add_argument('--resize_size', type=int, default=40,
                    help='先进行resize，然后进行中心裁剪，去除部分背景')
parser.add_argument('--input_size', type=int, default=32,
                    help='输入模型的数据尺寸')

# 加载数据、优化器、模型等
parser.add_argument('--epochs', type=int, default=50,
                    help='训练迭代次数')
parser.add_argument('--batch_size', type=int, default=256,
                    help='训练时选用的batch_size')

parser.add_argument('--feature_extract', type=bool, default=True,
                    help='是否使用原有的特征提取步骤')

# 优化器与学习率衰减
parser.add_argument('--lr', type=float, default=0.01,
                    help='优化器学习率参数')
parser.add_argument('--step_size', type=int, default=1,
                    help='学习率衰减频率')
parser.add_argument('--gamma', type=float, default=0.8,
                    help='学习率衰减系数')

parser.add_argument('--tolerance', type=int, default=5,
                    help='用于早停的容忍度值')

# 路径相关
parser.add_argument('--train_path', type=str, default='../GTSRB/Train_backup',
                    help='训练数据集路径')
parser.add_argument('--valid_path', type=str, default='../GTSRB/Valid',
                    help='验证集路径')
parser.add_argument('--test_path', type=str, default='../GTSRB/Test',
                    help='测试数据集路径')
parser.add_argument('--train_label_path', type=str, default='../GTSRB/Train.csv',
                    help='训练集的标签数据csv文件路径')
parser.add_argument('--test_label_path', type=str, default='../GTSRB/Test.csv',
                    help='测试集的标签数据csv文件路径')

parser.add_argument('--model_structure_path', type=str,
                    default="resnet34_model.pth",
                    help='模型结构的保存位置')

parser.add_argument('--model_path', type=str,
                    default='checkpoints/model_28.pth',
                    help='结构+参数路径')
parser.add_argument('--model_param_path', type=str,
                    default='checkpoints/model_param_28.pth',
                    help='用于分类的模型路径')

# 使用parse_args()解析函数
args = parser.parse_args()
