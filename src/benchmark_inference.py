import os
import csv
import time
import statistics

import torch
from torch import nn

from model import FashionCNN


# =====================================
# 1. 定义第一次剪枝后的模型
# conv1: 1 -> 12
# conv2: 12 -> 32
# =====================================

class Pruned12x32CNN(nn.Module):

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
# conv1: 1 -> 12
# conv2: 12 -> 24
# =====================================

class Pruned12x24CNN(nn.Module):

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
        x = self.fc2(x)
        return x


# =====================================
# 3. 设置项目路径和模型文件
# =====================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# 每个模型按顺序尝试这些文件名。
# 优先使用微调后的权重。
MODEL_CONFIGS = [
    {
        "name": "原始模型",
        "builder": FashionCNN,
        "files": ["fashion_cnn_baseline.pth"],
    },
    {
        "name": "第一次剪枝并微调",
        "builder": Pruned12x32CNN,
        "files": [
            "fashion_cnn_pruned_12ch_finetuned.pth",
            "fashion_cnn_pruned_12ch.pth",
        ],
    },
    {
        "name": "第二次剪枝并微调",
        "builder": Pruned12x24CNN,
        "files": [
            "fashion_cnn_pruned_12_24ch_finetuned.pth",
            "fashion_cnn_pruned_12_24ch.pth",
        ],
    },
]


def find_checkpoint(filenames):
    """在项目根目录和 checkpoints 目录查找权重文件。"""

    for filename in filenames:
        candidates = [
            os.path.join(PROJECT_ROOT, filename),
            os.path.join(PROJECT_ROOT, "checkpoints", filename),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

    raise FileNotFoundError(
        f"找不到权重文件：{filenames}\n"
        "请检查文件名及文件实际所在目录。"
    )


def load_model(config, device):
    """创建模型并加载对应的训练权重。"""

    checkpoint_path = find_checkpoint(config["files"])

    model = config["builder"]()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # 兼容直接保存 state_dict 或常见的字典封装格式
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    print(f"\n模型：{config['name']}")
    print(f"权重文件：{checkpoint_path}")
    print(f"测试设备：{device}")

    return model


# =====================================
# 4. 定义推理性能测试函数
# =====================================

def benchmark_inference(
    model,
    device,
    batch_size,
    repeats=100,
    trials=3,
):
    """
    测量推理延迟和吞吐量。

    latency_ms：每个 Batch 的平均推理时间，单位 ms
    throughput：每秒处理的图片数量
    """

    model.eval()

    images = torch.randn(
        batch_size, 1, 28, 28, device=device
    )

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    trial_latencies = []

    with torch.inference_mode():

        # 预热：避免首次运行开销影响结果
        for _ in range(20):
            model(images)

        synchronize()

        # 多次独立计时，降低偶然波动的影响
        for _ in range(trials):

            synchronize()
            start = time.perf_counter()

            for _ in range(repeats):
                model(images)

            synchronize()
            elapsed = time.perf_counter() - start

            latency_ms = elapsed / repeats * 1000
            trial_latencies.append(latency_ms)

    # 使用多轮测量结果的中位数
    latency_ms = statistics.median(trial_latencies)

    throughput = batch_size * 1000 / latency_ms

    return latency_ms, throughput


# =====================================
# 5. 运行所有模型的性能测试
# =====================================

def main():

    print("=" * 55)
    print("CNN 剪枝模型推理性能基准测试")
    print("=" * 55)

    devices = [torch.device("cpu")]

    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
        print("检测到 GPU：", torch.cuda.get_device_name(0))
    else:
        print("未检测到 CUDA GPU，将只测试 CPU。")

    results = []

    for config in MODEL_CONFIGS:

        for device in devices:

            model = load_model(config, device)

            for batch_size in [1, 64]:

                # 小 Batch 和大 Batch 使用不同的重复次数
                repeats = 200 if batch_size == 1 else 100

                latency_ms, throughput = benchmark_inference(
                    model=model,
                    device=device,
                    batch_size=batch_size,
                    repeats=repeats,
                    trials=3,
                )

                result = {
                    "model": config["name"],
                    "device": str(device),
                    "batch_size": batch_size,
                    "latency_ms": latency_ms,
                    "throughput_img_s": throughput,
                }

                results.append(result)

                print(
                    f"Batch={batch_size:2d} | "
                    f"延迟={latency_ms:.4f} ms | "
                    f"吞吐量={throughput:.2f} 张/秒"
                )

            del model

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # =====================================
    # 6. 保存 CSV 实验结果
    # =====================================

    csv_path = os.path.join(
        RESULTS_DIR,
        "benchmark_inference.csv",
    )

    fieldnames = [
        "model",
        "device",
        "batch_size",
        "latency_ms",
        "throughput_img_s",
    ]

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # =====================================
    # 7. 打印结果汇总
    # =====================================

    print("\n" + "=" * 85)
    print("最终推理性能对比")
    print("=" * 85)

    print(
        f"{'模型':<20}"
        f"{'设备':<10}"
        f"{'Batch':>8}"
        f"{'延迟(ms)':>14}"
        f"{'吞吐量(张/秒)':>20}"
    )

    for result in results:
        print(
            f"{result['model']:<20}"
            f"{result['device']:<10}"
            f"{result['batch_size']:>8}"
            f"{result['latency_ms']:>14.4f}"
            f"{result['throughput_img_s']:>20.2f}"
        )

    print("\n性能测试完成。")
    print(f"CSV 结果保存位置：{csv_path}")


if __name__ == "__main__":
    main()




        
