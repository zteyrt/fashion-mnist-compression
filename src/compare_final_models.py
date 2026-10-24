
import os
import time
import torch
import csv
import statistics
import matplotlib.pyplot as plt
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import FashionCNN

# 解决 Matplotlib 中文显示为方框的问题（Windows）
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei"
]

# 解决负号显示异常的问题
plt.rcParams["axes.unicode_minus"] = False

# =====================================
# 1. 定义第一次剪枝后的模型
# =====================================

class Pruned12CNN(nn.Module):

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
        x = self.fc2(x)
        return x


# =====================================
# 2. 定义第二次剪枝后的模型
# =====================================

class Pruned12_24CNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # 第二层卷积输出 24 个通道：
        # 24 * 7 * 7 = 1176
        self.fc1 = nn.Linear(1176, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# =====================================
# 3. 初始化环境和文件路径
# =====================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 本文件位于 src 文件夹，项目根目录是它的上一级
project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

# 如果你的模型文件名不同，只修改这里
model_paths = {
    "原始模型": os.path.join(
        project_root,
        "fashion_cnn_baseline.pth"
    ),

    "第一次剪枝并微调": os.path.join(
        project_root,
        "fashion_cnn_pruned_12ch_finetuned.pth"
    ),

    "第二次剪枝并微调": os.path.join(
        project_root,
        "fashion_cnn_pruned_12_24ch_finetuned.pth"
    )
}

print("当前设备:", device)
print("项目目录:", project_root)


# =====================================
# 4. 加载三个模型
# =====================================

models = {
    "原始模型": FashionCNN(),
    "第一次剪枝并微调": Pruned12CNN(),
    "第二次剪枝并微调": Pruned12_24CNN()
}

for name, model in models.items():

    checkpoint_path = model_paths[name]

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到模型文件：{checkpoint_path}\n"
            f"请检查 model_paths 中的文件名。"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"{name}加载成功")


# =====================================
# 5. 加载官方测试集
# =====================================

test_dataset = datasets.FashionMNIST(
    root=os.path.join(project_root, "data"),
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

test_loader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False,
    num_workers=0
)

print("\n测试集图片数量:", len(test_dataset))


# =====================================
# 6. 定义统一的评估函数
# =====================================

def evaluate_model(model, test_loader, criterion, device):

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
# 7. 定义推理速度测试函数
# =====================================

def benchmark_inference(
    model,
    device,
    batch_size,
    repeats=200
):

    model.eval()

    # 创建模拟输入，避免数据读取速度干扰推理计时
    images = torch.randn(
        batch_size,
        1,
        28,
        28,
        device=device
    )

    with torch.inference_mode():

        # 预热，降低首次运行的额外开销
        for _ in range(30):
            model(images)

        # CUDA 是异步执行的，需要等待计算完成
        if device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()

        # 正式计时
        for _ in range(repeats):
            model(images)

        if device.type == "cuda":
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - start

    # 每次处理一个 Batch 的平均耗时
    latency_ms = elapsed / repeats * 1000

    # 每秒处理的图片数量
    throughput = batch_size * repeats / elapsed

    return latency_ms, throughput


# =====================================
# 8. 评估准确率、参数量和模型大小
# =====================================

criterion = nn.CrossEntropyLoss()

results = {}

print("\n========== 开始模型评估 ==========")

for name, model in models.items():

    print(f"\n正在评估：{name}")

    test_loss, test_accuracy = evaluate_model(
        model,
        test_loader,
        criterion,
        device
    )

    total_params = sum(
        p.numel() for p in model.parameters()
    )

    checkpoint_path = model_paths[name]

    file_size_bytes = os.path.getsize(checkpoint_path)
    file_size_mb = file_size_bytes / (1024 ** 2)

    results[name] = {
        "loss": test_loss,
        "accuracy": test_accuracy,
        "params": total_params,
        "size_mb": file_size_mb
    }

    print(f"测试集 Loss: {test_loss:.4f}")
    print(f"测试集 Accuracy: {test_accuracy:.2f}%")
    print(f"模型参数量: {total_params:,}")
    print(f"模型文件大小: {file_size_mb:.4f} MB")



# =====================================
# 9. 多次测试推理速度
# =====================================

speed_results = {}

print("\n========== 推理速度对比 ==========")

for name, model in models.items():

    speed_results[name] = {}

    print(f"\n模型：{name}")

    # 分别测试单张图片和一个 Batch
    for batch_size in (1, 64):

        # 每种设置重复测试 5 次
        latency_list = []

        for repeat_idx in range(5):

            latency_ms, throughput = benchmark_inference(
                model,
                device,
                batch_size=batch_size,
                repeats=200
            )

            latency_list.append(latency_ms)

        # 取 5 次测试的中位数
        median_latency = statistics.median(latency_list)

        # 根据中位延迟计算吞吐量
        median_throughput = (
            batch_size * 1000 / median_latency
        )

        speed_results[name][batch_size] = {
            "latency_ms": median_latency,
            "throughput": median_throughput
        }

        print(f"Batch Size: {batch_size}")
        print(f"5 次测试的延迟: {latency_list}")
        print(f"中位延迟: {median_latency:.4f} ms")
        print(f"吞吐量: {median_throughput:.1f} 张/秒")

print("\n推理速度测试完成")


# =====================================
# 10. 汇总模型压缩效果
# =====================================

baseline = results["原始模型"]

print("\n========== 最终实验对比 ==========")

print(
    f"{'模型':<18}"
    f"{'准确率':>10}"
    f"{'参数量':>14}"
    f"{'文件大小(MB)':>16}"
)

print("参与汇总的模型：", list(results.keys()))

for name, result in results.items():

    print(
        f"{name:<18}"
        f"{result['accuracy']:>9.2f}%"
        f"{result['params']:>14,}"
        f"{result['size_mb']:>16.4f}"
    )

    accuracy_change = (
        result["accuracy"] - baseline["accuracy"]
    )

    param_reduction = (
        1 - result["params"] / baseline["params"]
    ) * 100

    size_reduction = (
        1 - result["size_mb"] / baseline["size_mb"]
    ) * 100

    print(f"  相对原始模型准确率变化: {accuracy_change:+.2f} 个百分点")
    print(f"  参数量减少: {param_reduction:.2f}%")
    print(f"  文件大小减少: {size_reduction:.2f}%")


print("\n========== 全部实验完成 ==========")


# =====================================
# 11. 保存全部实验结果到 CSV
# =====================================

# 创建实验结果目录
output_dir = os.path.join(
    project_root,
    "results",
    "pruning_comparison"
)

os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(
    output_dir,
    "final_experiment_results.csv"
)

# CSV 表头
fieldnames = [
    "model",
    "test_loss",
    "test_accuracy",
    "parameters",
    "file_size_MB",
    "batch1_latency_ms",
    "batch1_throughput",
    "batch64_latency_ms",
    "batch64_throughput"
]

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for name, result in results.items():

        row = {
            "model": name,
            "test_loss": round(result["loss"], 4),
            "test_accuracy": round(result["accuracy"], 2),
            "parameters": result["params"],
            "file_size_MB": round(result["size_mb"], 4),
            "batch1_latency_ms": round(
                speed_results[name][1]["latency_ms"], 4
            ),
            "batch1_throughput": round(
                speed_results[name][1]["throughput"], 2
            ),
            "batch64_latency_ms": round(
                speed_results[name][64]["latency_ms"], 4
            ),
            "batch64_throughput": round(
                speed_results[name][64]["throughput"], 2
            )
        }

        writer.writerow(row)

print("\n实验数据已保存：")
print(csv_path)


# =====================================
# 12. 绘制实验对比图
# =====================================

names = list(results.keys())

accuracies = [
    results[name]["accuracy"]
    for name in names
]

model_sizes = [
    results[name]["size_mb"]
    for name in names
]

parameter_counts = [
    results[name]["params"]
    for name in names
]

# -------------------------------------
# 图 1：测试集准确率对比
# -------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(names, accuracies)

plt.ylabel("Test Accuracy (%)")
plt.title("Test Accuracy Comparison")
plt.ylim(0, 100)
plt.xticks(rotation=15, ha="right")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "accuracy_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------
# 图 2：模型文件大小对比
# -------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(names, model_sizes)

plt.ylabel("Model File Size (MB)")
plt.title("Model File Size Comparison")
plt.xticks(rotation=15, ha="right")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "model_size_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------
# 图 3：参数量对比
# -------------------------------------

plt.figure(figsize=(8, 5))

plt.bar(names, parameter_counts)

plt.ylabel("Number of Parameters")
plt.title("Parameter Count Comparison")
plt.xticks(rotation=15, ha="right")

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "parameter_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# -------------------------------------
# 图 4：推理延迟对比
# -------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, batch_size in zip(axes, [1, 64]):

    latencies = [
        speed_results[name][batch_size]["latency_ms"]
        for name in names
    ]

    ax.bar(names, latencies)

    ax.set_title(f"Batch Size = {batch_size}")
    ax.set_ylabel("Latency (ms per batch)")
    ax.tick_params(axis="x", rotation=15)

fig.suptitle("Inference Latency Comparison")

plt.tight_layout()

plt.savefig(
    "results/pruning_comparison/inference_latency_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

fig.savefig(
    os.path.join(output_dir, "inference_latency_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\n========== 实验图表保存完成 ==========")

print("准确率对比图：accuracy_comparison.png")
print("文件大小对比图：model_size_comparison.png")
print("参数量对比图：parameter_comparison.png")
print("推理延迟对比图：inference_latency_comparison.png")

print("\n全部实验数据和图表已保存到：")
print(output_dir)
