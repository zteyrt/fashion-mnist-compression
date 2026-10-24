import os
import csv
import matplotlib.pyplot as plt

# =====================================
# 1. 设置项目路径并读取 CSV
# =====================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

csv_path = os.path.join(
    project_root,
    "results",
    "benchmark_inference_repeat.csv"
)

output_dir = os.path.join(
    project_root,
    "results"
)

os.makedirs(output_dir, exist_ok=True)

# 读取实验数据
if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"没有找到 CSV 文件：{csv_path}"
    )

with open(
    csv_path,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("CSV 文件中没有实验数据")

# 检查 CSV 必需的字段
required_columns = {
    "model",
    "device",
    "batch_size",
    "mean_latency_ms",
    "std_latency_ms",
    "mean_throughput"
}

if not required_columns.issubset(rows[0].keys()):
    raise ValueError(
        "CSV 缺少必要字段，请检查表头："
        f"{required_columns}"
    )

# 模型名称及显示标签
models = [
    "原始模型",
    "第一次剪枝并微调",
    "第二次剪枝并微调"
]

model_labels = [
    "Original",
    "Pruned-1",
    "Pruned-2"
]

print("=" * 55)
print("Fashion-MNIST 推理性能可视化")
print("=" * 55)
print("CSV 路径：", csv_path)
print("读取数据成功，共", len(rows), "条记录")


# =====================================
# 2. 定义数据筛选函数
# =====================================

def get_selected_rows(device, batch_size):
    """筛选指定设备和 Batch Size 的三组实验数据。"""

    selected = [
        row for row in rows
        if row["device"].strip().lower() == device
        and int(row["batch_size"]) == batch_size
    ]

    # 按模型顺序排列
    selected.sort(
        key=lambda row: models.index(row["model"])
    )

    if len(selected) != 3:
        raise ValueError(
            f"数据不完整：设备={device}, "
            f"Batch Size={batch_size}, "
            f"实际记录数={len(selected)}"
        )

    return selected


# =====================================
# 3. 绘制平均延迟与标准差
# =====================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9)
)

latency_settings = [
    ("cpu", 1, axes[0, 0]),
    ("cpu", 64, axes[0, 1]),
    ("cuda", 1, axes[1, 0]),
    ("cuda", 64, axes[1, 1]),
]

for device, batch_size, ax in latency_settings:

    selected = get_selected_rows(
        device,
        batch_size
    )

    means = [
        float(row["mean_latency_ms"])
        for row in selected
    ]

    stds = [
        float(row["std_latency_ms"])
        for row in selected
    ]

    # 绘制柱状图和标准差误差线
    ax.bar(
        model_labels,
        means,
        yerr=stds,
        capsize=6
    )

    ax.set_title(
        f"{device.upper()} - Batch Size {batch_size}"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Latency (ms)")

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.set_axisbelow(True)

# 总标题
fig.suptitle(
    "Inference Latency: Mean and Standard Deviation",
    fontsize=14
)

fig.tight_layout(
    pad=2.0,
    rect=[0.06, 0, 1, 0.94]
)

fig.subplots_adjust(
    left=0.12,
    wspace=0.22,
    hspace=0.30
)

# 保存延迟图
latency_path = os.path.join(
    output_dir,
    "repeat_latency.png"
)

fig.savefig(
    latency_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2
)

plt.close(fig)

print("\n延迟对比图已保存：")
print(latency_path)


# =====================================
# 4. 绘制平均吞吐量
# =====================================

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 9)
)

# 注意：这里必须使用第二张图新创建的 axes
throughput_settings = [
    ("cpu", 1, axes[0, 0]),
    ("cpu", 64, axes[0, 1]),
    ("cuda", 1, axes[1, 0]),
    ("cuda", 64, axes[1, 1]),
]

for device, batch_size, ax in throughput_settings:

    selected = get_selected_rows(
        device,
        batch_size
    )

    throughputs = [
        float(row["mean_throughput"])
        for row in selected
    ]

    # 绘制吞吐量柱状图
    ax.bar(
        model_labels,
        throughputs
    )

    ax.set_title(
        f"{device.upper()} - Batch Size {batch_size}"
    )

    ax.set_xlabel("Model")
    ax.set_ylabel("Throughput (images/second)")

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.set_axisbelow(True)

# 总标题
fig.suptitle(
    "Inference Throughput Comparison",
    fontsize=14
)

# 自动调整布局
fig.tight_layout(
    pad=2.0,
    rect=[0, 0, 1, 0.95]
)

# 保存吞吐量图
throughput_path = os.path.join(
    output_dir,
    "repeat_throughput.png"
)

fig.savefig(
    throughput_path,
    dpi=300,
    bbox_inches="tight",
    pad_inches=0.2
)

plt.close(fig)

print("\n吞吐量对比图已保存：")
print(throughput_path)


# =====================================
# 5. 实验完成
# =====================================

print("\n" + "=" * 55)
print("全部绘图完成")
print("=" * 55)

print("生成的文件：")
print("1. results/repeat_latency.png")
print("2. results/repeat_throughput.png")