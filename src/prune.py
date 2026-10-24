
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.nn.utils import prune

from model import FashionCNN


# =====================================
# 1. 初始化环境
# =====================================

print("开始进行 CNN 模型剪枝实验")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)


# =====================================
# 2. 加载原始模型
# =====================================

model = FashionCNN().to(device)

baseline_path = "./fashion_cnn_baseline.pth"

if not os.path.exists(baseline_path):
    raise FileNotFoundError(
        f"找不到原始模型文件: {baseline_path}"
    )

state_dict = torch.load(
    baseline_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

print("原始模型加载成功")


# =====================================
# 3. 加载 Fashion-MNIST 测试集
# =====================================

transform = transforms.ToTensor()

test_dataset = datasets.FashionMNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("测试集数量:", len(test_dataset))


# =====================================
# 4. 定义模型评估函数
# =====================================

def evaluate(model, test_loader, device):

    model.eval()

    correct = 0
    total = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss()

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
# 5. 评估剪枝前的模型
# =====================================

baseline_loss, baseline_accuracy = evaluate(
    model,
    test_loader,
    device
)

print("\n========== 剪枝前 ==========")
print(f"测试集 Loss: {baseline_loss:.4f}")
print(f"测试集 Accuracy: {baseline_accuracy:.2f}%")


# =====================================
# 6. 设置需要剪枝的网络层
# =====================================

parameters_to_prune = (
    (model.conv1, "weight"),
    (model.conv2, "weight"),
    (model.fc1, "weight"),
    (model.fc2, "weight"),
)

print("\n参与剪枝的层:")
print("conv1、conv2、fc1、fc2")


# =====================================
# 7. 执行全局非结构化剪枝
# =====================================

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3
)

print("\n30% 全局非结构化剪枝完成")


# =====================================
# 8. 移除剪枝重参数化
# =====================================

# 将剪枝结果固定到普通 weight 参数中
# 被剪掉的权重仍然保持为 0

for module, parameter_name in parameters_to_prune:
    prune.remove(module, parameter_name)

print("剪枝重参数化已移除")


# =====================================
# 9. 统计剪枝后的权重稀疏率
# =====================================

total_weights = 0
zero_weights = 0

for module, _ in parameters_to_prune:

    weights = module.weight.detach()

    total_weights += weights.numel()

    zero_weights += (weights == 0).sum().item()

sparsity = zero_weights / total_weights * 100

print("\n========== 权重稀疏率 ==========")
print(f"参与剪枝的权重总数: {total_weights:,}")
print(f"其中零权重数量: {zero_weights:,}")
print(f"实际权重稀疏率: {sparsity:.2f}%")


# =====================================
# 10. 评估剪枝后的模型
# =====================================

pruned_loss, pruned_accuracy = evaluate(
    model,
    test_loader,
    device
)

print("\n========== 剪枝后 ==========")
print(f"测试集 Loss: {pruned_loss:.4f}")
print(f"测试集 Accuracy: {pruned_accuracy:.2f}%")


# =====================================
# 11. 计算准确率变化
# =====================================

accuracy_change = pruned_accuracy - baseline_accuracy

print("\n========== 剪枝效果 ==========")
print(f"剪枝前 Accuracy: {baseline_accuracy:.2f}%")
print(f"剪枝后 Accuracy: {pruned_accuracy:.2f}%")
print(f"准确率变化: {accuracy_change:+.2f} 个百分点")


# =====================================
# 12. 保存剪枝后的模型
# =====================================

pruned_path = "./fashion_cnn_pruned_30.pth"

torch.save(
    model.state_dict(),
    pruned_path
)

print("\n剪枝后的模型保存成功:", pruned_path)


# =====================================
# 13. 比较权重文件大小
# =====================================

baseline_size = os.path.getsize(baseline_path)
pruned_size = os.path.getsize(pruned_path)

print("\n========== 文件大小对比 ==========")
print(f"原始模型: {baseline_size:,} bytes")
print(f"剪枝模型: {pruned_size:,} bytes")
print(
    f"文件大小变化: "
    f"{(pruned_size - baseline_size) / baseline_size * 100:+.2f}%"
)

print("\n第一次模型剪枝实验完成")
