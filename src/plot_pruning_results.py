
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# =====================================
# 1. 读取剪枝实验结果
# =====================================

csv_path = Path(
    "results/pruning_comparison/pruning_comparison.csv"
)

output_dir = Path("results/pruning_comparison")
output_dir.mkdir(parents=True, exist_ok=True)

with csv_path.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as file:
    rows = list(csv.DictReader(file))

if not rows:
    raise ValueError("CSV 文件中没有实验数据。")

# 提取需要绘图的数据
pruning_rates = [
    float(row["pruning_rate"]) for row in rows
]

accuracies = [
    float(row["test_accuracy"]) for row in rows
]

file_sizes_kb = [
    int(row["file_size_bytes"]) / 1024
    for row in rows
]

print("读取实验数据成功")
print("剪枝率:", pruning_rates)
print("测试准确率:", accuracies)
print("模型大小（KB）:", file_sizes_kb)


# =====================================
# 2. 图一：剪枝率与测试准确率
# =====================================

fig1, ax1 = plt.subplots(figsize=(8, 5))

ax1.plot(
    pruning_rates,
    accuracies,
    marker="o",
    label="Test Accuracy"
)

ax1.set_xlabel("Pruning Rate (%)")
ax1.set_ylabel("Test Accuracy (%)")
ax1.set_title("Pruning Rate vs Test Accuracy")

ax1.set_xticks(pruning_rates)
ax1.grid(True)
ax1.legend()

fig1.tight_layout()

fig1.savefig(
    output_dir / "pruning_rate_vs_accuracy.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================
# 3. 图二：剪枝率与模型文件大小
# =====================================

fig2, ax2 = plt.subplots(figsize=(8, 5))

ax2.plot(
    pruning_rates,
    file_sizes_kb,
    marker="s",
    label="Model File Size"
)

ax2.set_xlabel("Pruning Rate (%)")
ax2.set_ylabel("Model File Size (KB)")
ax2.set_title("Pruning Rate vs Model File Size")

ax2.set_xticks(pruning_rates)

# 从 0 开始显示纵轴，避免夸大微小的文件大小变化
ax2.set_ylim(
    0,
    max(file_sizes_kb) * 1.05
)

ax2.grid(True)
ax2.legend()

fig2.tight_layout()

fig2.savefig(
    output_dir / "pruning_rate_vs_file_size.png",
    dpi=300,
    bbox_inches="tight"
)


# =====================================
# 4. 显示并保存结果
# =====================================

print("\n两张图已保存：")
print(output_dir / "pruning_rate_vs_accuracy.png")
print(output_dir / "pruning_rate_vs_file_size.png")

plt.show()
