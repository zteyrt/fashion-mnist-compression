
from pathlib import Path
import torch

# 获取项目根目录
project_root = Path(__file__).resolve().parent.parent

model_files = {
    "原始模型": "fashion_cnn_baseline.pth",
    "第一次剪枝": "fashion_cnn_pruned_12ch_finetuned.pth",
    "第二次剪枝": "fashion_cnn_pruned_12_24ch_finetuned.pth"
}

for name, filename in model_files.items():

    path = project_root / filename

    # 读取模型权重
    state_dict = torch.load(
        path,
        map_location="cpu",
        weights_only=True
    )

    print(f"\n========== {name} ==========")

    # 检查关键层的权重形状
    for layer in [
        "conv1.weight",
        "conv2.weight",
        "fc1.weight",
        "fc2.weight"
    ]:
        print(layer, state_dict[layer].shape)

print("\n权重形状检查完成！")
