
import os
import csv
import time
import statistics

import torch
from torch import nn

from model import FashionCNN


# =====================================
# 1. 定义第一次剪枝后的模型
# =====================================

class PrunedFashionCNN12(nn.Module):

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
        return self.fc2(x)


# =====================================
# 2. 定义第二次剪枝后的模型
# =====================================

class PrunedFashionCNN12_24(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 12, 3, padding=1)
        self.conv2 = nn.Conv2d(12, 24, 3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(24 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


# =====================================
# 3. 配置项目路径和模型
# =====================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

results_dir = os.path.join(project_root, "results")
os.makedirs(results_dir, exist_ok=True)

models_config = [
    (
        "原始模型",
        FashionCNN,
        "fashion_cnn_baseline.pth"
    ),
    (
        "第一次剪枝并微调",
        PrunedFashionCNN12,
        "fashion_cnn_pruned_12ch_finetuned.pth"
    ),
    (
        "第二次剪枝并微调",
        PrunedFashionCNN12_24,
        "fashion_cnn_pruned_12_24ch_finetuned.pth"
    ),
]

# 重复次数和每次测量的推理次数
num_trials = 20
warmup_runs = 30

batch_settings = {
    1: 500,
    64: 200,
}

devices = [torch.device("cpu")]

if torch.cuda.is_available():
    devices.append(torch.device("cuda"))

print("=" * 65)
print("推理性能重复测量与稳定性分析")
print("=" * 65)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

print("重复测量次数:", num_trials)


# =====================================
# 4. 单次推理性能测量
# =====================================

def benchmark_once(
    model,
    device,
    batch_size,
    warmup_runs,
    repeats
):
    model.eval()

    images = torch.randn(
        batch_size, 1, 28, 28, device=device
    )

    with torch.inference_mode():

        # 预热，降低首次运行的干扰
        for _ in range(warmup_runs):
            model(images)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        start_time = time.perf_counter()

        for _ in range(repeats):
            model(images)

        # CUDA 操作是异步的，计时前必须等待执行完成
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        elapsed = time.perf_counter() - start_time

    latency_ms = elapsed / repeats * 1000

    throughput = (
        batch_size * repeats / elapsed
    )

    return latency_ms, throughput


# =====================================
# 5. 重复测试所有模型
# =====================================

results = []

for model_name, model_class, checkpoint_name in models_config:

    checkpoint_path = os.path.join(
        project_root,
        checkpoint_name
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"找不到模型文件: {checkpoint_path}"
        )

    print("\n" + "=" * 65)
    print("模型:", model_name)
    print("权重文件:", checkpoint_name)

    for device in devices:

        model = model_class().to(device)

        state_dict = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )

        model.load_state_dict(state_dict)
        model.eval()

        print(f"\n设备: {device}")

        for batch_size, repeats in batch_settings.items():

            latencies = []
            throughputs = []

            print(
                f"Batch Size={batch_size}, "
                f"每次测量推理 {repeats} 次"
            )

            for trial in range(num_trials):

                latency, throughput = benchmark_once(
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    warmup_runs=warmup_runs,
                    repeats=repeats
                )

                latencies.append(latency)
                throughputs.append(throughput)

                print(
                    f"  第 {trial + 1} 次: "
                    f"延迟={latency:.4f} ms, "
                    f"吞吐量={throughput:.2f} 张/秒"
                )

            mean_latency = statistics.mean(latencies)

            # 样本标准差：反映多次测量的波动程度
            std_latency = statistics.stdev(latencies)

            mean_throughput = statistics.mean(throughputs)

            result = {
                "model": model_name,
                "device": str(device),
                "batch_size": batch_size,
                "trials": num_trials,
                "mean_latency_ms": mean_latency,
                "std_latency_ms": std_latency,
                "mean_throughput": mean_throughput,
            }

            results.append(result)

            print(
                f"平均延迟: {mean_latency:.4f} ms"
            )
            print(
                f"延迟标准差: {std_latency:.4f} ms"
            )
            print(
                f"平均吞吐量: {mean_throughput:.2f} 张/秒"
            )

        del model

        if device.type == "cuda":
            torch.cuda.empty_cache()


# =====================================
# 6. 输出汇总表并保存 CSV
# =====================================

print("\n" + "=" * 100)
print("推理性能重复测量汇总")
print("=" * 100)

print(
    f"{'模型':<22}"
    f"{'设备':<8}"
    f"{'Batch':>7}"
    f"{'平均延迟(ms)':>16}"
    f"{'标准差(ms)':>14}"
    f"{'平均吞吐量(张/秒)':>22}"
)

for result in results:
    print(
        f"{result['model']:<22}"
        f"{result['device']:<8}"
        f"{result['batch_size']:>7}"
        f"{result['mean_latency_ms']:>16.4f}"
        f"{result['std_latency_ms']:>14.4f}"
        f"{result['mean_throughput']:>22.2f}"
    )

csv_path = os.path.join(
    results_dir,
    "benchmark_inference_repeat.csv"
)

with open(
    csv_path,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = [
        "model",
        "device",
        "batch_size",
        "trials",
        "mean_latency_ms",
        "std_latency_ms",
        "mean_throughput",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)

print("\n重复测量完成")
print("CSV 保存位置:", csv_path)
