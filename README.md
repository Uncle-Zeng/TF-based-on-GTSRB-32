与上一[迁移学习项目](https://github.com/Uncle-Zeng/Transfer-Learning-based-on-Resnet34-GTSRB-PyTorch.git)的不同：

 - 这次是在自己的账户下配置好环境，然后进行训练的；
 - 简单优化了`utils.py`的逻辑，并新增了`time()`函数用于计算`epoch`时间；
 - 将模型的输入尺寸建议改为32*32，并重新训练，在测试集上获得了`94.08%`的准确率；