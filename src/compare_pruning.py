
import csv
from pathlib import Path

import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune

from model import FashionCNN


# =====================================
# 1. 实验配置
# =====================================

# 原始模型权重路径：请确认完整文件名
BASELINE_PATH = Path("fashion_cnn_baseline.pth")

# 实验结果保存目录
RESULT_DIR = Path("results/pruning_comparison")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# 五种剪枝率
PRUNING_RATES = [0.0, 0.1, 0.3, 0.5, 0.7]

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前设备:", device)
print("原始模型路径:", BASELINE_PATH.resolve())

if not BASELINE_PATH.exists():
    raise FileNotFoundError(
        f"找不到原始模型文件：{BASELINE_PATH}\n"
        "请检查 BASELINE_PATH 中的文件名和路径。"
    )


# =====================================
# 2. 加载测试集
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
    shuffle=False,
    num_workers=0
)

print("测试集图片数量:", len(test_dataset))


# =====================================
# 3. 定义模型权重统计函数
# =====================================

def get_prunable_layers(model):
    """返回参与全局剪枝的四层权重。"""
    return [
        (model.conv1, "weight"),
        (model.conv2, "weight"),
        (model.fc1, "weight"),
        (model.fc2, "weight"),
    ]


def calculate_sparsity(model):
    """统计四层权重中的零元素比例。"""
    total_weights = 0
    zero_weights = 0

    for layer, name in get_prunable_layers(model):
        weight = getattr(layer, name)

        total_weights += weight.numel()
        zero_weights += (weight == 0).sum().item()

    sparsity = zero_weights / total_weights * 100

    return zero_weights, total_weights, sparsity


# =====================================
# 4. 定义模型评估函数
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
# 5. 加载原始权重
# =====================================

checkpoint = torch.load(
    BASELINE_PATH,
    map_location="cpu"
)

# 兼容直接保存 state_dict 和常见检查点格式
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

criterion = nn.CrossEntropyLoss()

results = []

print("\n========== 开始剪枝率对比实验 ==========")


# =====================================
# 6. 循环执行五组实验
# =====================================

for amount in PRUNING_RATES:

    percentage = int(amount * 100)

    print(f"\n========== 剪枝率 {percentage}% ==========")

    # 每次都创建新模型，重新加载原始权重
    model = FashionCNN()
    model.load_state_dict(state_dict)
    model = model.to(device)

    # 0% 是原始模型对照组
    if amount > 0:
        parameters_to_prune = get_prunable_layers(model)

        # 在四层权重上执行全局非结构化剪枝
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount
        )

        # 移除剪枝重参数化，保留置零后的普通权重
        for layer, name in parameters_to_prune:
            prune.remove(layer, name)

        # 保存本组剪枝模型
        output_path = RESULT_DIR / f"model_pruned_{percentage}.pth"

        torch.save(
            model.state_dict(),
            output_path
        )

        file_size = output_path.stat().st_size

    else:
        # 对照组直接使用原始模型文件大小
        file_size = BASELINE_PATH.stat().st_size

    # 统计实际稀疏率
    zero_weights, total_weights, sparsity = (
        calculate_sparsity(model)
    )

    # 在同一个测试集上评估
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        criterion,
        device
    )

    result = {
        "pruning_rate": percentage,
        "actual_sparsity": round(sparsity, 4),
        "test_loss": round(test_loss, 4),
        "test_accuracy": round(test_accuracy, 4),
        "file_size_bytes": file_size,
        "file_size_MB": round(file_size / (1024 ** 2), 4),
    }

    results.append(result)

    print("实际零权重数量:", zero_weights)
    print("参与统计的权重总数:", total_weights)
    print(f"实际稀疏率: {sparsity:.2f}%")
    print(f"测试 Loss: {test_loss:.4f}")
    print(f"测试 Accuracy: {test_accuracy:.2f}%")
    print(f"模型文件大小: {file_size:,} bytes")


# =====================================
# 7. 输出实验对比表
# =====================================

print("\n========== 五组实验汇总 ==========")

print(
    f"{'剪枝率':<10}"
    f"{'实际稀疏率':<14}"
    f"{'测试Loss':<12}"
    f"{'测试准确率':<14}"
    f"{'文件大小(bytes)':<18}"
)

for result in results:
    print(
        f"{result['pruning_rate']:<12}"
        f"{result['actual_sparsity']:<16.2f}"
        f"{result['test_loss']:<14.4f}"
        f"{result['test_accuracy']:<16.2f}"
        f"{result['file_size_bytes']:<18,}"
    )


# =====================================
# 8. 保存实验结果 CSV
# =====================================

csv_path = RESULT_DIR / "pruning_comparison.csv"

with csv_path.open(
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)

print("\n实验全部完成。")
print("结果表已保存到:", csv_path)
print("各组剪枝模型保存在:", RESULT_DIR.resolve())
