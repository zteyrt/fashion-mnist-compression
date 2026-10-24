import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from model import FashionCNN


# =====================================
# 1. 定义训练函数
# =====================================

def train_one_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    device
):
    # 切换到训练模式
    model.train()

    # 初始化统计变量
    total_loss = 0.0
    correct = 0
    total = 0

    # 遍历训练集中的每个 Batch
    for batch_idx, (images, labels) in enumerate(train_loader):

        # 将图片和标签移动到 GPU 或 CPU
        images = images.to(device)
        labels = labels.to(device)

        # 第一步：清空梯度
        optimizer.zero_grad()

        # 第二步：前向传播
        outputs = model(images)

        # 第三步：计算损失
        loss = criterion(outputs, labels)

        # 第四步：反向传播
        loss.backward()

        # 第五步：更新模型参数
        optimizer.step()

        # 统计当前 Batch 的数据
        batch_size = images.size(0)

        # 累计损失（转换成该 Batch 的损失总和）
        total_loss += loss.item() * batch_size

        # 获取预测类别
        predicted = outputs.argmax(dim=1)

        # 累计预测正确的图片数量
        correct += (predicted == labels).sum().item()

        # 累计处理的图片数量
        total += batch_size

        # 每 100 个 Batch 打印一次进度
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            print(
                f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Loss: {loss.item():.4f}"
            )

    # 计算整个训练集的平均 Loss 和 Accuracy
    average_loss = total_loss / total
    accuracy = correct / total * 100

    print("训练集累计正确数量:", correct)
    print("训练集累计图片数量:", total)
    print(f"训练集平均 Loss: {average_loss:.4f}")
    print(f"训练集 Accuracy: {accuracy:.2f}%")

    return average_loss, accuracy


# =====================================
# 2. 定义验证函数
# =====================================

def evaluate(
    model,
    val_loader,
    criterion,
    device
):
    # 切换到评估模式
    model.eval()

    # 初始化统计变量
    total_loss = 0.0
    correct = 0
    total = 0

    # 验证阶段不需要计算梯度
    with torch.inference_mode():

        # 遍历验证集中的每个 Batch
        for images, labels in val_loader:

            # 将图片和标签移动到 GPU 或 CPU
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)

            # 计算损失
            loss = criterion(outputs, labels)

            # 当前 Batch 的图片数量
            batch_size = images.size(0)

            # 累计损失
            total_loss += loss.item() * batch_size

            # 获取预测类别
            predicted = outputs.argmax(dim=1)

            # 累计预测正确的数量
            correct += (predicted == labels).sum().item()

            # 累计处理的图片数量
            total += batch_size

    # 计算整个验证集的平均 Loss 和 Accuracy
    average_loss = total_loss / total
    accuracy = correct / total * 100

    return average_loss, accuracy


# =====================================
# 3. 初始化训练环境
# =====================================

print("训练程序初始化成功")

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前训练设备:", device)


# =====================================
# 4. 创建 CNN 模型
# =====================================

model = FashionCNN().to(device)

print("模型结构:")
print(model)

print("模型所在设备:", next(model.parameters()).device)


# =====================================
# 5. 定义损失函数和优化器
# =====================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("损失函数:", criterion)
print("优化器:", optimizer.__class__.__name__)


# =====================================
# 6. 加载 Fashion-MNIST 数据集
# =====================================

transform = transforms.ToTensor()

full_train_dataset = datasets.FashionMNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)


# =====================================
# 7. 划分训练集和验证集
# =====================================

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [50000, 10000],
    generator=torch.Generator().manual_seed(42)
)

print("训练集数量:", len(train_dataset))
print("验证集数量:", len(val_dataset))


# =====================================
# 8. 创建 DataLoader
# =====================================

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

print("训练 Batch 数量:", len(train_loader))
print("验证 Batch 数量:", len(val_loader))

# =====================================
# 9. 执行多轮训练
# =====================================

#设置训练轮数
num_epochs = 5

#保存每一轮的结果
history = {
    "train_loss": [],
    "train_accuracy": [],
    "val_loss": [],
    "val_accuracy": []
}

print("\n========== 开始多轮训练 ==========")

for epoch in range(num_epochs):

    print(f"\n========== Epoch [{epoch + 1}/{num_epochs}] 轮训练 ==========")

    #1.训练一个完整的Epoch
    train_loss,train_accuracy = train_one_epoch(
       model,
       train_loader,
       criterion,
       optimizer,
       device
    )

     #2.在验证集上评估
    val_loss,val_accuracy = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

     #3.保存本轮结果
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)

     #4.输出本轮总结
    print(f"\n第 {epoch + 1} 轮训练结果")
    print(f"训练集 Loss: {train_loss:.4f}")
    print(f"训练集 Accuracy: {train_accuracy:.2f}%")
    print(f"验证集 Loss: {val_loss:.4f}")
    print(f"验证集 Accuracy: {val_accuracy:.2f}%")

print(f"\n========== {num_epochs}轮训练完成 ==========")
print("实际记录的训练轮数:", len(history["train_loss"]))

# =====================================
# 10. 绘制训练曲线
# =====================================

#横坐标:Epoch编号
epochs = range(1,len(history["train_loss"]) + 1)

#创建一个包含两个子图的画布
plt.figure(figsize=(12, 5))

# -------------------------------------
# 左图：Loss 曲线
# -------------------------------------

plt.subplot(1,2,1)

plt.plot(
    epochs,
    history["train_loss"],
    marker = "o",
    label = "Train Loss"
)

plt.plot(
    epochs,
    history["val_loss"],
    marker = "s",
    label = "Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.xticks(list(epochs))

# -------------------------------------
# 右图：Accuracy 曲线
# -------------------------------------

plt.subplot(1,2,2)

plt.plot(
    epochs,
    history["train_accuracy"],
    marker = "o",
    label = "Train Accuracy"
)

plt.plot(
    epochs,
    history["val_accuracy"],
    marker = "s",
    label = "Validation Accuracy"
)

# 设置标题和坐标轴
plt.title("Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")

# 添加图例、网格和横坐标刻度
plt.legend()
plt.grid(True)
plt.xticks(list(epochs))
plt.ylim(0,100)

#自动调整布局
plt.tight_layout()

#保存图片
plt.savefig(
    "results/training_curves.png",
    dpi=300,
    bbox_inches="tight"
)

print("训练曲线已保存到 results/training_curves.png")

# 显示图片
plt.show()

#保存原始CNN模型权重
torch.save(
    model.state_dict(),
    "fashion_cnn_baseline.pth"
)

print("原始模型权重保存成功")

