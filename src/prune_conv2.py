
import os
import torch
from torch import nn


# =====================================
# 1. 定义第一阶段模型
# =====================================

class FirstPrunedCNN(nn.Module):

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
        return self.fc2(x)


# =====================================
# 2. 定义第二阶段剪枝后的模型
# =====================================

class SecondPrunedCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # 24 个通道，每个通道有 7×7 个特征
        self.fc1 = nn.Linear(24 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# =====================================
# 3. 加载第一阶段微调模型
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

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"没有找到模型文件: {checkpoint_path}"
    )

source_model = FirstPrunedCNN().to(device)

state_dict = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

source_model.load_state_dict(state_dict)
source_model.eval()

print("第一阶段微调模型加载成功")
print("当前设备:", device)


# =====================================
# 4. 计算 conv2 通道重要性
# =====================================

scores = (
    source_model.conv2.weight.detach()
    .abs()
    .sum(dim=(1, 2, 3))
)

# 删除评分最低的 8 个通道
remove_channels = torch.argsort(scores)[:8].tolist()

keep_channels_list = [
    i for i in range(32)
    if i not in remove_channels
]

keep_channels = torch.tensor(
    keep_channels_list,
    dtype=torch.long,
    device=device
)

print("删除的通道:", sorted(remove_channels))
print("保留的通道:", keep_channels_list)


# =====================================
# 5. 找到 fc1 对应的输入特征
# =====================================

# 每个 conv2 通道对应连续的 49 个特征
keep_features_list = [
    channel * 49 + offset
    for channel in keep_channels_list
    for offset in range(49)
]

keep_features = torch.tensor(
    keep_features_list,
    dtype=torch.long,
    device=device
)

print("保留的 fc1 输入特征数量:", len(keep_features_list))


# =====================================
# 6. 创建剪枝后的模型并复制权重
# =====================================

pruned_model = SecondPrunedCNN().to(device)

with torch.no_grad():

    # conv1 不变，完整复制
    pruned_model.conv1.weight.copy_(
        source_model.conv1.weight
    )
    pruned_model.conv1.bias.copy_(
        source_model.conv1.bias
    )

    # conv2：选择保留的输出通道
    pruned_model.conv2.weight.copy_(
        source_model.conv2.weight[keep_channels]
    )
    pruned_model.conv2.bias.copy_(
        source_model.conv2.bias[keep_channels]
    )

    # fc1：同步选择对应的输入特征
    pruned_model.fc1.weight.copy_(
        source_model.fc1.weight[:, keep_features]
    )
    pruned_model.fc1.bias.copy_(
        source_model.fc1.bias
    )

    # fc2 保持不变
    pruned_model.fc2.weight.copy_(
        source_model.fc2.weight
    )
    pruned_model.fc2.bias.copy_(
        source_model.fc2.bias
    )

pruned_model.eval()


# =====================================
# 7. 检查网络结构和前向传播
# =====================================

print("\n========== 剪枝后的网络 ==========")
print(pruned_model)

test_images = torch.randn(
    2, 1, 28, 28,
    device=device
)

with torch.inference_mode():
    outputs = pruned_model(test_images)

print("输入形状:", test_images.shape)
print("输出形状:", outputs.shape)

assert outputs.shape == (2, 10)

print("前向传播检查通过")


# =====================================
# 8. 比较参数量
# =====================================

original_params = sum(
    p.numel() for p in source_model.parameters()
)

pruned_params = sum(
    p.numel() for p in pruned_model.parameters()
)

print("\n========== 参数量对比 ==========")
print(f"剪枝前参数量: {original_params:,}")
print(f"剪枝后参数量: {pruned_params:,}")
print(
    f"参数量减少比例: "
    f"{(original_params - pruned_params) / original_params * 100:.2f}%"
)


# =====================================
# 9. 保存第二阶段剪枝模型
# =====================================

save_path = os.path.join(
    project_root,
    "fashion_cnn_pruned_12_24ch.pth"
)

torch.save(pruned_model.state_dict(), save_path)

print("\n模型保存成功:", save_path)
print("第二阶段结构化剪枝完成")
