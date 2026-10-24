
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import FashionCNN


# =====================================
# 1. 初始化测试环境
# =====================================

print("开始评估原始 CNN 模型")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)


# =====================================
# 2. 加载已经训练好的模型
# =====================================

model = FashionCNN().to(device)

# 模型权重保存在项目根目录
checkpoint_path = "./fashion_cnn_baseline.pth"

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"没有找到模型文件: {checkpoint_path}"
    )

state_dict = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

print("模型权重加载成功")


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

print("测试集图片数量:", len(test_dataset))
print("测试 Batch 数量:", len(test_loader))


# =====================================
# 4. 在测试集上评估模型
# =====================================

criterion = nn.CrossEntropyLoss()

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

test_loss = total_loss / total
test_accuracy = correct / total * 100

print("\n========== 测试集评估结果 ==========")
print(f"测试集 Loss: {test_loss:.4f}")
print(f"测试集正确数量: {correct}/{total}")
print(f"测试集 Accuracy: {test_accuracy:.2f}%")


# =====================================
# 5. 统计模型参数量
# =====================================

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n========== 模型参数统计 ==========")
print(f"模型总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")


# =====================================
# 6. 统计模型权重文件大小
# =====================================

file_size_bytes = os.path.getsize(checkpoint_path)

file_size_mb = file_size_bytes / (1024 ** 2)

print("\n========== 模型文件统计 ==========")
print(f"权重文件大小: {file_size_bytes:,} bytes")
print(f"权重文件大小: {file_size_mb:.4f} MB")

print("\n原始模型评估完成")
