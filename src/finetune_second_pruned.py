
import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


# =====================================
# 1. 定义第二阶段剪枝后的模型
# =====================================

class SecondPrunedCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(1176, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# =====================================
# 2. 定义一个 Epoch 的训练函数
# =====================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        # 清空上一批次的梯度
        optimizer.zero_grad()

        # 前向传播
        outputs = model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        optimizer.step()

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += batch_size

        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            print(
                f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / total
    accuracy = correct / total * 100

    return average_loss, accuracy


# =====================================
# 3. 定义验证函数
# =====================================

def evaluate(
    model,
    val_loader,
    criterion,
    device
):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.inference_mode():

        for images, labels in val_loader:

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
# 4. 初始化环境
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

print("当前设备:", device)


# =====================================
# 5. 加载第二阶段剪枝模型
# =====================================

checkpoint_path = os.path.join(
    project_root,
    "fashion_cnn_pruned_12_24ch.pth"
)

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"没有找到模型文件: {checkpoint_path}"
    )

model = SecondPrunedCNN().to(device)

state_dict = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)

print("第二阶段剪枝模型加载成功")


# =====================================
# 6. 加载训练数据并划分验证集
# =====================================

dataset = datasets.FashionMNIST(
    root=os.path.join(project_root, "data"),
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

train_dataset, val_dataset = random_split(
    dataset,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False
)

print("训练集数量:", len(train_dataset))
print("验证集数量:", len(val_dataset))


# =====================================
# 7. 设置微调参数
# =====================================

criterion = nn.CrossEntropyLoss()

# 微调时使用较小的学习率
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0005
)

num_epochs = 5

best_val_accuracy = 0.0

save_path = os.path.join(
    project_root,
    "fashion_cnn_pruned_12_24ch_finetuned.pth"
)


# =====================================
# 8. 开始微调
# =====================================

print("\n========== 开始第二阶段微调 ==========")

for epoch in range(num_epochs):

    print(
        f"\n========== Epoch "
        f"[{epoch + 1}/{num_epochs}] =========="
    )

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    print(f"训练集 Loss: {train_loss:.4f}")
    print(f"训练集 Accuracy: {train_accuracy:.2f}%")
    print(f"验证集 Loss: {val_loss:.4f}")
    print(f"验证集 Accuracy: {val_accuracy:.2f}%")

    # 保存验证准确率最高的模型
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy

        torch.save(
            model.state_dict(),
            save_path
        )

        print("已保存当前最佳模型")

print("\n========== 第二阶段微调完成 ==========")
print(f"最佳验证集 Accuracy: {best_val_accuracy:.2f}%")
print("模型保存路径:", save_path)
