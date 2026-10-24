
import os
import csv
import torch
from torch import nn

from model import FashionCNN


# =====================================
# 1. 定义第一次剪枝后的模型
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
        return self.fc2(x)


# =====================================
# 2. 定义第二次剪枝后的模型
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
        return self.fc2(x)


# =====================================
# 3. 项目路径与模型权重
# =====================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

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

    for filename in filenames:

        candidates = [
            os.path.join(PROJECT_ROOT, filename),
            os.path.join(PROJECT_ROOT, "checkpoints", filename),
        ]

        for path in candidates:
            if os.path.isfile(path):
                return path

    raise FileNotFoundError(
        f"没有找到模型权重文件：{filenames}"
    )


def load_model(config):

    checkpoint_path = find_checkpoint(config["files"])

    model = config["builder"]()

    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # 兼容常见的 checkpoint 封装格式
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.eval()

    return model, checkpoint_path


# =====================================
# 4. 计算 MACs
# =====================================

def calculate_macs(model):

    total_macs = 0
    handles = []

    def conv_hook(module, inputs, output):
        nonlocal total_macs

        # 单个输出元素需要的乘加次数
        kernel_ops = (
            module.in_channels // module.groups
        ) * module.kernel_size[0] * module.kernel_size[1]

        # 输出元素数量 × 每个输出元素的 MACs
        total_macs += output.numel() * kernel_ops

    def linear_hook(module, inputs, output):
        nonlocal total_macs

        total_macs += output.numel() * module.in_features

    # 只统计卷积层与全连接层
    for module in model.modules():

        if isinstance(module, nn.Conv2d):
            handles.append(
                module.register_forward_hook(conv_hook)
            )

        elif isinstance(module, nn.Linear):
            handles.append(
                module.register_forward_hook(linear_hook)
            )

    # Fashion-MNIST 单张输入
    dummy_input = torch.zeros(1, 1, 28, 28)

    try:
        with torch.inference_mode():
            model(dummy_input)
    finally:
        for handle in handles:
            handle.remove()

    return total_macs


# =====================================
# 5. 分析三个模型
# =====================================

def main():

    print("=" * 65)
    print("Fashion-MNIST 模型计算量分析")
    print("=" * 65)

    results = []

    for config in MODEL_CONFIGS:

        model, checkpoint_path = load_model(config)

        # 参数总量
        params = sum(
            p.numel() for p in model.parameters()
        )

        # 单张图片的 MACs
        macs = calculate_macs(model)

        # 本实验采用 1 MAC = 2 FLOPs 的统计口径
        flops = 2 * macs

        # 模型权重文件大小
        file_size_bytes = os.path.getsize(checkpoint_path)
        file_size_mb = file_size_bytes / (1024 ** 2)

        result = {
            "model": config["name"],
            "parameters": params,
            "MACs": macs,
            "FLOPs": flops,
            "MACs_M": macs / 1e6,
            "FLOPs_M": flops / 1e6,
            "file_size_MB": file_size_mb,
        }

        results.append(result)

        print(f"\n模型：{config['name']}")
        print(f"权重文件：{checkpoint_path}")
        print(f"参数量：{params:,}")
        print(f"MACs：{macs:,} ({macs / 1e6:.4f} M)")
        print(f"FLOPs：{flops:,} ({flops / 1e6:.4f} M)")
        print(f"文件大小：{file_size_mb:.4f} MB")

        del model

    # =====================================
    # 6. 与原始模型对比
    # =====================================

    baseline = results[0]

    print("\n" + "=" * 65)
    print("相对于原始模型的计算量变化")
    print("=" * 65)

    for result in results:

        macs_reduction = (
            1 - result["MACs"] / baseline["MACs"]
        ) * 100

        params_reduction = (
            1 - result["parameters"] / baseline["parameters"]
        ) * 100

        print(f"\n模型：{result['model']}")
        print(f"参数量变化：{params_reduction:+.2f}%")
        print(f"MACs 变化：{macs_reduction:+.2f}%")

    # =====================================
    # 7. 保存 CSV
    # =====================================

    csv_path = os.path.join(
        RESULTS_DIR,
        "model_compute.csv",
    )

    fieldnames = [
        "model",
        "parameters",
        "MACs",
        "FLOPs",
        "MACs_M",
        "FLOPs_M",
        "file_size_MB",
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

    print("\n计算量分析完成")
    print(f"CSV 保存位置：{csv_path}")


if __name__ == "__main__":
    main()
