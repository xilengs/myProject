## 实现NIN网络

本项目主要实现NIN网络架构。其中网络结构中的参数来源于[cuda-convnet/NIN/cifar-10_def at master · mavenlin/cuda-convnet](https://github.com/mavenlin/cuda-convnet/blob/master/NIN/cifar-10_def)

NIN的网络结构相对简单，主要的创新在于提出了$mlpconv$和使用全局平均池化对传统的卷积神经网络进行改善。在不同的数据集上也取得了不错的效果。

本项目在实现NIN网络的基础上，使用CIFAR-10数据集进行训练。

第一次训练在学习率为0.001下训练50轮。在20轮左右开始验证集的Loss和Acc开始出现震荡，训练效果较差。Loss维持在【1.2，1.4]之间；Acc维持在[0.55, 0.60]之间。

第二次训练在学习率为0.0005下训练20轮。验证集在整个训练过程中Loss/Acc没有明显下降/上升，但相较于第一轮都有改善，Loss在[1.20, 1.25]之间，Acc在[0.62, 0.64]之间。