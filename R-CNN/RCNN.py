import torch
import torch.nn as nn

# VGG16 Net
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # Conv Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Conv Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


class RCNN(nn.Module):
    """
    R-CNN 模型: 基于 VGG16 骨干网络，并在其上添加分类头和回归头。
    通过复用 VGG16 的 fc6 和 fc7 层来简化代码。

    注意：在实际训练中，需要确保输入 x 是经过 RoI Pooling 后的固定尺寸特征
    （例如 7x7x512），或者在 forward 中包含 RoI Pooling 逻辑。
    这里假设输入 x 已经是 512x7x7 的特征图。
    """

    def __init__(self, num_classes, pretrained=None):
        super().__init__()

        # 1. 实例化 VGG16 (num_classes 设为 1000 只是为了初始化 VGG16 的结构)
        vgg16 = VGG16(num_classes=1000)

        if pretrained:
            # 实际加载预训练权重，这里仅示意加载流程，您需要在外部实现权重加载
            print("加载 VGG16 预训练权重...")
            state_dict = torch.load(pretrained)
            vgg16.load_state_dict(state_dict)

            # 2. 特征提取层：保留 VGG16 的所有卷积层
        self.features = vgg16.features

        # 3. R-CNN 头部：从 VGG16 的 classifier 中取出 fc6 和 fc7
        # VGG16.classifier = [FC6, ReLU, Dropout, FC7, ReLU, Dropout, FC8(1000)]

        # FC6 (index 0)
        self.fc6 = list(vgg16.classifier.children())[0]  # nn.Linear(512 * 7 * 7, 4096)
        self.relu6 = list(vgg16.classifier.children())[1]
        self.dropout6 = list(vgg16.classifier.children())[2]

        # FC7 (index 3)
        self.fc7 = list(vgg16.classifier.children())[3]  # nn.Linear(4096, 4096)
        self.relu7 = list(vgg16.classifier.children())[4]
        self.dropout7 = list(vgg16.classifier.children())[5]

        # 4. R-CNN 任务分支 (替换 VGG16 原有的 FC8)
        # 分类头 (K+1 类: K个物体类 + 1个背景类)
        self.cls_score = nn.Linear(4096, num_classes + 1)

        # 边界框回归头 (K 类，每类 4 个回归参数: K*4)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

        # 初始新增的分类和回归头的权重
        self._init_rcnn_heads()

    def _init_rcnn_heads(self):
        # 初始化分类和回归层的权重
        nn.init.normal_(self.cls_score.weight, 0, 0.01)
        nn.init.constant_(self.cls_score.bias, 0)
        nn.init.normal_(self.bbox_pred.weight, 0, 0.001)
        nn.init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        # 假设：输入 x 是经过 RoI Pooling 后的特征 (N_rois, 512, 7, 7)

        # 1. 展平 (Flatten)
        # 从维度 1 开始展平：(N_rois, 512, 7, 7) -> (N_rois, 512*7*7)
        x = torch.flatten(x, 1)

        # 2. 全连接层 (fc6, fc7)
        x = self.dropout6(self.relu6(self.fc6(x)))
        x = self.dropout7(self.relu7(self.fc7(x)))

        # 3. 分支：分类和回归
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred