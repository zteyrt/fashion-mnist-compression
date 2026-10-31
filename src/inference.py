import os
import torch

from torchvision import datasets, transforms
from model import FashionCNN
from torch.utils.data import DataLoader

#1获取当前脚本所在目录
current_dir = os.path.dirname(
    os.path.abspath(__file__)
)

#2.获取项目根目录
project_root = os.path.dirname(current_dir)

#3.构造模型权重路径
checkpoint_path = os.path.join(
    project_root,
    "fashion_cnn_baseline.pth"
)

#4.检查模型权重文件是否存在
print("模型路径:", checkpoint_path)
print("模型文件是否存在:", os.path.exists(checkpoint_path))


#5.选择运行设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:",device)

#6.创建cnn网络结构
model = FashionCNN().to(device)

#7.从文件中读取模型权重
state_dict = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

#8.将权重加载到模型中
model.load_state_dict(state_dict)

#9.切换到推理模式
model.eval()

print("模型权重加载成功！")
print("模型当前模式:", "训练" if model.training else "推理")

# ==========================================
# 10. 加载 Fashion-MNIST 测试集
# ==========================================

test_dataset = datasets.FashionMNIST(
    root=os.path.join(project_root, "data"),
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

print("测试集加载成功！")
print("测试集数量:", len(test_dataset))


# ==========================================
# 11. 定义 Fashion-MNIST 类别
# ==========================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# ==========================================
# 图片预测函数
# ==========================================

def predict_image(model, image, device, class_names):

    # 增加 Batch 维度
    input_tensor = image.unsqueeze(0).to(device)

    # 模型推理
    with torch.inference_mode():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs, dim=1
        )[0]

        confidence, predicted_idx = torch.max(
            probabilities, dim=0
        )

    # 返回预测类别、置信度和全部类别概率
    return (
        class_names[predicted_idx.item()],
        confidence.item(),
        probabilities.cpu()
    )

# ==========================================
# 批量测试整个测试集
# ==========================================

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

model.eval()

total_correct = 0
total_samples = 0

# 统计每个类别的正确数量
class_correct = [0] * len(class_names)
class_total = [0] * len(class_names)

with torch.inference_mode():

    for images, labels in test_loader:

        # 将数据移动到 GPU
        images = images.to(device)
        labels = labels.to(device)

        # 批量推理
        outputs = model(images)

        # 获取预测类别
        predictions = outputs.argmax(dim=1)

        # 统计整体正确数量
        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

        # 统计各类别准确率
        for i in range(len(class_names)):

            mask = labels == i

            class_total[i] += mask.sum().item()

            class_correct[i] += (
                (predictions == labels) & mask
            ).sum().item()


# ==========================================
# 输出整体准确率
# ==========================================

accuracy = total_correct / total_samples * 100

print("\n========== 批量测试结果 ==========")

print("测试图片总数:", total_samples)
print("预测正确数量:", total_correct)
print("预测错误数量:", total_samples - total_correct)

print(f"整体测试准确率: {accuracy:.2f}%")

print("\n========== 各类别准确率 ==========")

for i, name in enumerate(class_names):

    class_accuracy = (
        class_correct[i] / class_total[i] * 100
    )

    print(
        f"{name}: "
        f"{class_accuracy:.2f}% "
        f"({class_correct[i]}/{class_total[i]})"
    )

print("==================================")




