
import os
import torch
from model import FashionCNN
from torch import nn


# =====================================
# 1. 定义第一阶段剪枝后的模型
# =====================================

class PrunedFashionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 32, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =====================================
# 2. 加载第一阶段微调后的模型
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

checkpoint_path = os.path.join(
    project_root,
    "fashion_cnn_pruned_12ch_finetuned.pth"
)

model = PrunedFashionCNN().to(device)

state_dict = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("第一阶段微调模型加载成功")


# =====================================
# 3. 计算 conv2 每个输出通道的重要性
# =====================================

scores = (
    model.conv2.weight.detach()
    .abs()
    .sum(dim=(1, 2, 3))
)

print("\n========== conv2 通道重要性 ==========")

for channel_idx, score in enumerate(scores):
    print(
        f"通道 {channel_idx:2d} 的重要性: "
        f"{score.item():.4f}"
    )


# =====================================
# 4. 找出评分最低的 8 个通道
# =====================================

remove_channels = torch.argsort(scores)[:8].tolist()

keep_channels = [
    i for i in range(32)
    if i not in remove_channels
]

print("\n准备删除的 8 个通道:")
print(sorted(remove_channels))

print("\n准备保留的 24 个通道:")
print(keep_channels)

print("\nconv2 通道重要性分析完成")
