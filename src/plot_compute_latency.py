
import os
import csv
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ==========================================
# 1. 路径与中文字体设置
# ==========================================

project_root = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

results_dir = os.path.join(project_root, "results")

csv_path = os.path.join(
    results_dir,
    "benchmark_inference_repeat.csv"
)

output_path = os.path.join(
    results_dir,
    "compute_latency_comparison.png"
)

os.makedirs(results_dir, exist_ok=True)

# 自动选择当前电脑上已安装的中文字体
available_fonts = {
    font.name for font in font_manager.fontManager.ttflist
}

font_candidates = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC"
]

selected_font = next(
    (name for name in font_candidates if name in available_fonts),
    None
)

if selected_font:
    plt.rcParams["font.sans-serif"] = [selected_font]
    print("使用中文字体：", selected_font)
else:
    print("警告：没有找到常见中文字体，请检查系统字体安装。")

plt.rcParams["axes.unicode_minus"] = False

# ==========================================
# 2. 读取 20 次重复测量的推理数据
# ==========================================

if not os.path.exists(csv_path):
    raise FileNotFoundError(
        f"找不到实验数据：{csv_path}"
    )

with open(
    csv_path,
    "r",
    encoding="utf-8-sig",
    newline=""
) as f:
    rows = list(csv.DictReader(f))

if not rows:
    raise ValueError("CSV 文件中没有实验数据。")

models = [
    "原始模型",
    "第一次剪枝并微调",
    "第二次剪枝并微调"
]

labels = [
    "原始模型",
    "第一次剪枝",
    "第二次剪枝"
]

print(f"读取重复测量数据成功，共 {len(rows)} 条记录。")

# 检查每种设备和 Batch Size 都有三个模型
for device in ["cpu", "cuda"]:
    for batch_size in [1, 64]:
        selected = [
            row for row in rows
            if row["device"].strip().lower() == device
            and int(row["batch_size"]) == batch_size
        ]

        found_models = {row["model"] for row in selected}

        if len(selected) != 3 or found_models != set(models):
            raise ValueError(
                f"数据不完整：设备={device}, "
                f"Batch Size={batch_size}。"
                f"请检查 CSV 中的模型名称和记录数量。"
            )

# ==========================================
# 3. 准备计算量数据
# ==========================================

# 来自 measure_compute.py 的实际统计结果
# 单位：MACs
macs = [
    1218048,
    964032,
    744512
]

macs_millions = [
    value / 1_000_000 for value in macs
]

# ==========================================
# 4. 绘制计算量与推理延迟对比图
# ==========================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# 第一个子图：MACs
ax = axes[0, 0]

bars = ax.bar(labels, macs_millions)

ax.set_title("模型计算量对比（MACs）")
ax.set_ylabel("MACs（百万）")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

for bar, value in zip(bars, macs_millions):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{value:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

# 其余四个子图：统一使用重复测量 CSV
settings = [
    ("cpu", 1, axes[0, 1], "CPU 单张推理延迟"),
    ("cpu", 64, axes[0, 2], "CPU Batch=64 延迟"),
    ("cuda", 1, axes[1, 0], "GPU 单张推理延迟"),
    ("cuda", 64, axes[1, 1], "GPU Batch=64 延迟"),
]

for device, batch_size, ax, title in settings:

    selected = [
        row for row in rows
        if row["device"].strip().lower() == device
        and int(row["batch_size"]) == batch_size
    ]

    selected.sort(
        key=lambda row: models.index(row["model"])
    )

    means = [
        float(row["mean_latency_ms"])
        for row in selected
    ]

    stds = [
        float(row["std_latency_ms"])
        for row in selected
    ]

    bars = ax.bar(
        labels,
        means,
        yerr=stds,
        capsize=5
    )

    ax.set_title(title)
    ax.set_ylabel("延迟（ms）")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=10)

    # 在柱子上显示平均延迟
    for bar, value in zip(bars, means):
        ax.annotate(
            f"{value:.3f}",
            (
                bar.get_x() + bar.get_width() / 2,
                bar.get_height()
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8
        )

# 第六个位置不需要图表
axes[1, 2].axis("off")

fig.suptitle(
    "结构化剪枝：计算量与实际推理延迟对比",
    fontsize=16
)

fig.tight_layout(rect=[0, 0, 1, 0.95])

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("\n图表生成成功！")
print("图片保存位置：", output_path)
print("计算量数据：", macs)
print("延迟数据来源：benchmark_inference_repeat.csv")
