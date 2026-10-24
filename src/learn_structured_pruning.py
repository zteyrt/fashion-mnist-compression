
import os
import torch
from torch import nn
from model import FashionCNN
from torchvision import datasets,transforms
from torch.utils.data import DataLoader, random_split


# =====================================
# 1. 定义结构化剪枝后的 CNN
# =====================================

class PrunedFashionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # 第一层：输出通道由 16 减少到 12
        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=12,
            kernel_size=3,
            padding=1
        )

        # 第二层：输入通道由 16 减少到 12
        # 输出通道仍然保持 32
        self.conv2 = nn.Conv2d(
            in_channels=12,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # 全连接层保持不变
        self.fc1 = nn.Linear(1568, 128)
        self.fc2 = nn.Linear(128, 10)

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
# 2. 初始化环境并加载原始模型
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)

# 假设本文件位于项目的 src 文件夹中
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

checkpoint_path = os.path.join(
    project_root,
    "fashion_cnn_baseline.pth"
)

if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(
        f"没有找到原始模型权重文件: {checkpoint_path}"
    )

model = FashionCNN().to(device)

state_dict = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(state_dict)
model.eval()

print("原始模型权重加载成功")


# =====================================
# 3. 计算 conv1 各通道的重要性
# =====================================

scores = (
    model.conv1.weight.detach()
    .abs()
    .sum(dim=(1, 2, 3))
)

print("\n========== conv1 通道重要性 ==========")

for channel_idx, score in enumerate(scores):
    print(
        f"通道 {channel_idx:2d} 的重要性: "
        f"{score.item():.4f}"
    )


# =====================================
# 4. 找出需要删除的 4 个通道
# =====================================

least_important = torch.argsort(scores)[:4]

remove_channels = least_important.tolist()

print("\n需要删除的通道索引:")
print(remove_channels)

# 找出需要保留的 12 个通道
keep_channels_list = [
    i for i in range(16)
    if i not in remove_channels
]

keep_channels = torch.tensor(
    keep_channels_list,
    dtype=torch.long,
    device=device
)

print("\n保留的通道索引:")
print(keep_channels.tolist())


# =====================================
# 5. 提取需要保留的权重
# =====================================

# conv1：选择保留通道对应的输出权重和偏置
new_conv1_weight = model.conv1.weight[keep_channels]
new_conv1_bias = model.conv1.bias[keep_channels]

# conv2：选择对应的输入通道权重
new_conv2_weight = model.conv2.weight[:, keep_channels, :, :]

print("\n========== 切片后的权重形状 ==========")
print("conv1.weight:", new_conv1_weight.shape)
print("conv1.bias:", new_conv1_bias.shape)
print("conv2.weight:", new_conv2_weight.shape)


# =====================================
# 6. 创建真正剪枝后的模型
# =====================================

pruned_model = PrunedFashionCNN().to(device)

# 将保留的权重复制到新模型
with torch.no_grad():

    # 第一层卷积
    pruned_model.conv1.weight.copy_(new_conv1_weight)
    pruned_model.conv1.bias.copy_(new_conv1_bias)

    # 第二层卷积
    pruned_model.conv2.weight.copy_(new_conv2_weight)
    pruned_model.conv2.bias.copy_(model.conv2.bias)

    # 全连接层：完整复制原始权重
    pruned_model.fc1.weight.copy_(model.fc1.weight)
    pruned_model.fc1.bias.copy_(model.fc1.bias)

    pruned_model.fc2.weight.copy_(model.fc2.weight)
    pruned_model.fc2.bias.copy_(model.fc2.bias)

pruned_model.eval()

print("\n========== 剪枝后的模型 ==========")
print(pruned_model)


# =====================================
# 7. 检查前向传播是否正常
# =====================================

test_images = torch.randn(
    2, 1, 28, 28,
    device=device
)

with torch.inference_mode():
    test_outputs = pruned_model(test_images)

print("\n========== 前向传播测试 ==========")
print("测试输入形状:", test_images.shape)
print("模型输出形状:", test_outputs.shape)

assert test_outputs.shape == (2, 10), \
    "模型输出形状不正确"

print("前向传播测试通过")


# =====================================
# 8. 比较剪枝前后的参数量
# =====================================

original_params = sum(
    p.numel() for p in model.parameters()
)

pruned_params = sum(
    p.numel() for p in pruned_model.parameters()
)

print("\n========== 参数量对比 ==========")
print(f"原始模型参数量: {original_params:,}")
print(f"剪枝后参数量: {pruned_params:,}")
print(
    f"参数量减少: "
    f"{(original_params - pruned_params) / original_params * 100:.2f}%"
)

print("\n结构化剪枝基础实验完成")


# =====================================
# 9. 在测试集上比较剪枝前后的模型
# =====================================

# 1. 加载官方测试集
test_dataset = datasets.FashionMNIST(
    root=os.path.join(project_root, "data"),
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

print("\n测试集图片数量:", len(test_dataset))


# 2. 定义损失函数
criterion = nn.CrossEntropyLoss()


# 3. 定义统一的模型评估函数
def evaluate_model(model, test_loader, criterion, device):

    # 切换到评估模式
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    # 评估时不需要计算梯度
    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)

            batch_size = images.size(0)

            total_loss += loss.item() * batch_size

            # 统计预测正确的图片数量
            predicted = outputs.argmax(dim=1)
            correct += (predicted == labels).sum().item()
            total += batch_size

    average_loss = total_loss / total
    accuracy = correct / total * 100

    return average_loss, accuracy


# 4. 评估原始模型
print("\n========== 评估原始模型 ==========")

original_loss, original_accuracy = evaluate_model(
    model,
    test_loader,
    criterion,
    device
)


# 5. 评估剪枝后的模型
print("\n========== 评估剪枝后的模型 ==========")

pruned_loss, pruned_accuracy = evaluate_model(
    pruned_model,
    test_loader,
    criterion,
    device
)


# 6. 对比评估结果
print("\n========== 剪枝前后测试结果 ==========")

print(f"原始模型 Loss: {original_loss:.4f}")
print(f"原始模型 Accuracy: {original_accuracy:.2f}%")

print(f"剪枝模型 Loss: {pruned_loss:.4f}")
print(f"剪枝模型 Accuracy: {pruned_accuracy:.2f}%")

accuracy_change = pruned_accuracy - original_accuracy

print(f"准确率变化: {accuracy_change:+.2f} 个百分点")

print("\n剪枝前后模型评估完成")


# =====================================
# 10. 准备微调训练集
# =====================================

print("\n========== 准备微调数据 ==========")

full_train_dataset = datasets.FashionMNIST(
    root=os.path.join(project_root, "data"),
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

# 使用与原训练阶段相同的随机划分
train_dataset, val_dataset = random_split(
    full_train_dataset,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

print("微调训练图片数量:", len(train_dataset))


# =====================================
# 11. 微调剪枝后的模型
# =====================================

print("\n========== 开始微调 ==========")

criterion = nn.CrossEntropyLoss()

# 使用较小的学习率，调整剪枝后的模型参数
finetune_optimizer = torch.optim.Adam(
    pruned_model.parameters(),
    lr=0.0001
)

finetune_epochs = 5

for epoch in range(finetune_epochs):

    pruned_model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # 清空梯度
        finetune_optimizer.zero_grad()

        # 前向传播
        outputs = pruned_model(images)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()

        # 更新参数
        finetune_optimizer.step()

        # 统计训练指标
        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        predicted = outputs.argmax(dim=1)

        correct += (predicted == labels).sum().item()
        total += batch_size

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total * 100

    print(
        f"微调 Epoch [{epoch + 1}/{finetune_epochs}] | "
        f"Loss: {epoch_loss:.4f} | "
        f"Accuracy: {epoch_accuracy:.2f}%"
    )


# =====================================
# 12. 在测试集上重新评估
# =====================================

print("\n========== 微调后测试 ==========")

finetuned_loss, finetuned_accuracy = evaluate_model(
    pruned_model,
    test_loader,
    criterion,
    device
)

print(f"原始模型 Accuracy: {original_accuracy:.2f}%")
print(f"微调前剪枝模型 Accuracy: {pruned_accuracy:.2f}%")
print(f"微调后剪枝模型 Accuracy: {finetuned_accuracy:.2f}%")

print(
    f"相对原始模型的准确率变化: "
    f"{finetuned_accuracy - original_accuracy:+.2f} 个百分点"
)

print(f"微调后测试集 Loss: {finetuned_loss:.4f}")


# =====================================
# 13. 保存微调后的剪枝模型
# =====================================

save_path = os.path.join(
    project_root,
    "fashion_cnn_pruned_12ch_finetuned.pth"
)

torch.save(pruned_model.state_dict(), save_path)

print("\n微调后的模型权重已保存:", save_path)
