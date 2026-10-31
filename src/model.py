
import torch
from torch import nn


# ==========================================
# 1. 定义 Fashion-MNIST CNN 网络
# ==========================================

class FashionCNN(nn.Module):

    def __init__(
        self,
        conv1_channels=16,
        conv2_channels=32
    ):
        super().__init__()

        # ==================================
        # 第一层卷积
        # ==================================

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1_channels,
            kernel_size=3,
            padding=1
        )

        # ==================================
        # 第二层卷积
        # ==================================

        self.conv2 = nn.Conv2d(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=3,
            padding=1
        )

        # ReLU 激活层
        self.relu = nn.ReLU()

        # 最大池化层
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        # 展平层
        self.flatten = nn.Flatten()

        # ==================================
        # 第一个全连接层
        # ==================================

        # 28×28 输入经过两次池化：
        # 28 -> 14 -> 7
        # 因此输入特征数为：
        # conv2_channels × 7 × 7

        self.fc1 = nn.Linear(
            in_features=conv2_channels * 7 * 7,
            out_features=128
        )

        # ==================================
        # 第二个全连接层
        # ==================================

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )

    # ======================================
    # 2. 定义前向传播
    # ======================================

    def forward(self, x):

        # 第一个卷积模块
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 第二个卷积模块
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        # 展平卷积特征
        x = self.flatten(x)

        # 第一个全连接层
        x = self.fc1(x)
        x = self.relu(x)

        # 第二个全连接层
        x = self.fc2(x)

        return x


# ==========================================
# 3. 测试三种 CNN 结构
# ==========================================

if __name__ == "__main__":

    # 定义三种模型
    models = {
        "原始 CNN": FashionCNN(
            conv1_channels=16,
            conv2_channels=32
        ),

        "第一次剪枝模型": FashionCNN(
            conv1_channels=12,
            conv2_channels=32
        ),

        "第二次剪枝模型": FashionCNN(
            conv1_channels=12,
            conv2_channels=24
        )
    }

    # 创建一批模拟图片
    x = torch.randn(1, 1, 28, 28)

    print("=" * 55)
    print("Fashion-MNIST CNN 模型结构测试")
    print("=" * 55)

    for name, model in models.items():

        model.eval()

        with torch.inference_mode():
            output = model(x)

        # 统计模型参数量
        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        print(f"\n模型：{name}")

        print(
            "Conv1 输出通道：",
            model.conv1.out_channels
        )

        print(
            "Conv2 输出通道：",
            model.conv2.out_channels
        )

        print(
            "FC1 输入特征数：",
            model.fc1.in_features
        )

        print("输入形状：", x.shape)
        print("输出形状：", output.shape)
        print("模型参数量：", f"{total_params:,}")

    print("\n全部模型结构测试完成！")
