
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import FashionCNN


# =====================================
# 1. 定义结构化剪枝后的 CNN
# =====================================

class StructuredFashionCNN(nn.Module):

    def __init__(self, conv1_channels=12, conv2_channels=24):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=conv1_channels,
            kernel_size=3,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=conv1_channels,
            out_channels=conv2_channels,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # 输入特征数随第二个卷积层的通道数变化
        self.fc1 = nn.Linear(
            in_features=conv2_channels * 7 * 7,
            out_features=128
        )

        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# =====================================
# 2. 根据卷积核重要性选择通道
# =====================================

def structured_prune(original_model, device):

    # 原始卷积核的 L1 重要性：
    # 将一个输出通道的所有权重绝对值相加

    conv1_scores = (
        original_model.conv1.weight.detach()
        .abs()
        .sum(dim=(1, 2, 3))
    )

    # 保留 conv1 中最重要的 12 个输出通道
    keep_conv1 = torch.topk(
        conv1_scores,
        k=12
    ).indices.sort().values

    # 先只考虑保留的 conv1 输入通道，
    # 再计算 conv2 每个输出通道的重要性
    conv2_weights = (
        original_model.conv2.weight.detach()[:, keep_conv1, :, :]
    )

    conv2_scores = conv2_weights.abs().sum(dim=(1, 2, 3))

    # 保留 conv2 中最重要的 24 个输出通道
    keep_conv2 = torch.topk(
        conv2_scores,
        k=24
    ).indices.sort().values

    # 创建实际尺寸更小的新模型
    pruned_model = StructuredFashionCNN(
        conv1_channels=12,
        conv2_channels=24
    ).to(device)

    # =====================================
    # 3. 复制保留通道对应的权重
    # =====================================

    with torch.no_grad():

        # conv1：选择保留的输出通道
        pruned_model.conv1.weight.copy_(
            original_model.conv1.weight[keep_conv1]
        )

        pruned_model.conv1.bias.copy_(
            original_model.conv1.bias[keep_conv1]
        )

        # conv2：同时选择保留的输入和输出通道
        pruned_model.conv2.weight.copy_(
            original_model.conv2.weight[
                keep_conv2
            ][:, keep_conv1, :, :]
        )

        pruned_model.conv2.bias.copy_(
            original_model.conv2.bias[keep_conv2]
        )

        # 展平后，每个通道对应 7*7=49 个特征
        fc1_indices = torch.cat([
            torch.arange(
                int(channel.item()) * 49,
                (int(channel.item()) + 1) * 49,
                device=device
            )
            for channel in keep_conv2
        ])

        # fc1：删除被剪掉的卷积通道对应的输入列
        pruned_model.fc1.weight.copy_(
            original_model.fc1.weight[:, fc1_indices]
        )

        # fc1 的输出神经元全部保留
        pruned_model.fc1.bias.copy_(
            original_model.fc1.bias
        )

        # fc2 的结构没有改变，直接复制
        pruned_model.fc2.weight.copy_(
            original_model.fc2.weight
        )

        pruned_model.fc2.bias.copy_(
            original_model.fc2.bias
        )

    return pruned_model


# =====================================
# 4. 定义测试集评估函数
# =====================================

def evaluate(model, test_loader, criterion, device):

    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total * 100

    return average_loss, accuracy


# =====================================
# 5. 初始化环境并加载原始模型
# =====================================

print("开始进行 CNN 结构化剪枝实验")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)

# 以项目根目录作为文件路径基准
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

baseline_path = os.path.join(
    project_root,
    "fashion_cnn_baseline.pth"
)

pruned_path = os.path.join(
    project_root,
    "fashion_cnn_structured_pruned.pth"
)

if not os.path.exists(baseline_path):
    raise FileNotFoundError(
        f"没有找到原始模型权重：{baseline_path}"
    )

original_model = FashionCNN().to(device)

state_dict = torch.load(
    baseline_path,
    map_location=device,
    weights_only=True
)

original_model.load_state_dict(state_dict)
original_model.eval()

print("原始模型权重加载成功")


# =====================================
# 6. 加载官方测试集
# =====================================

transform = transforms.ToTensor()

test_dataset = datasets.FashionMNIST(
    root=os.path.join(project_root, "data"),
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("测试集图片数量:", len(test_dataset))


# =====================================
# 7. 评估剪枝前的模型
# =====================================

criterion = nn.CrossEntropyLoss()

baseline_loss, baseline_accuracy = evaluate(
    original_model,
    test_loader,
    criterion,
    device
)

print("\n========== 结构化剪枝前 ==========")
print(f"测试集 Loss: {baseline_loss:.4f}")
print(f"测试集 Accuracy: {baseline_accuracy:.2f}%")


# =====================================
# 8. 执行结构化剪枝
# =====================================

pruned_model = structured_prune(
    original_model,
    device
)

pruned_model.eval()

print("\n结构化剪枝完成")
print("原始 conv1 输出通道数: 16")
print("剪枝后 conv1 输出通道数: 12")
print("原始 conv2 输出通道数: 32")
print("剪枝后 conv2 输出通道数: 24")

# 检查模型输出尺寸
with torch.inference_mode():
    sample = torch.randn(2, 1, 28, 28, device=device)
    output = pruned_model(sample)

print("剪枝后模型输出形状:", output.shape)


# =====================================
# 9. 评估剪枝后的模型
# =====================================

pruned_loss, pruned_accuracy = evaluate(
    pruned_model,
    test_loader,
    criterion,
    device
)

print("\n========== 结构化剪枝后 ==========")
print(f"测试集 Loss: {pruned_loss:.4f}")
print(f"测试集 Accuracy: {pruned_accuracy:.2f}%")

print(
    f"准确率变化: "
    f"{pruned_accuracy - baseline_accuracy:+.2f} 个百分点"
)


# =====================================
# 10. 比较模型参数量
# =====================================

baseline_params = sum(
    p.numel() for p in original_model.parameters()
)

pruned_params = sum(
    p.numel() for p in pruned_model.parameters()
)

reduction = (
    (baseline_params - pruned_params)
    / baseline_params * 100
)

print("\n========== 参数量对比 ==========")
print(f"原始模型参数量: {baseline_params:,}")
print(f"剪枝后参数量: {pruned_params:,}")
print(f"参数量减少: {reduction:.2f}%")


# =====================================
# 11. 保存剪枝后的模型
# =====================================

torch.save(
    pruned_model.state_dict(),
    pruned_path
)

baseline_size = os.path.getsize(baseline_path)
pruned_size = os.path.getsize(pruned_path)

print("\n========== 模型文件大小对比 ==========")
print(f"原始模型: {baseline_size:,} bytes")
print(f"剪枝后模型: {pruned_size:,} bytes")

print(
    f"文件大小变化: "
    f"{(pruned_size - baseline_size) / baseline_size * 100:+.2f}%"
)

print("\n剪枝后模型保存位置:", pruned_path)
print("结构化剪枝实验完成")
