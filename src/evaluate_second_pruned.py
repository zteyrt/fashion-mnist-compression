
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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
# 2. 初始化环境并加载模型
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

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
model.eval()

print("第二阶段剪枝模型加载成功")
print("当前设备:", device)


# =====================================
# 3. 加载 Fashion-MNIST 测试集
# =====================================

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

print("测试集图片数量:", len(test_dataset))


# =====================================
# 4. 在测试集上评估
# =====================================

criterion = nn.CrossEntropyLoss()

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

print("\n========== 第二阶段剪枝评估结果 ==========")
print(f"测试集 Loss: {test_loss:.4f}")
print(f"测试集正确数量: {correct}/{total}")
print(f"测试集 Accuracy: {test_accuracy:.2f}%")

print("\n========== 参数量统计 ==========")

total_params = sum(
    p.numel() for p in model.parameters()
)

print(f"模型参数量: {total_params:,}")

print("\n第二阶段剪枝评估完成")
