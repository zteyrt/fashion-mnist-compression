
from pathlib import Path

import torch

from src.model import FashionCNN


# 项目根目录
project_root = Path(__file__).resolve().parent.parent

# 选择设备
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 三种模型的配置
model_configs = {
    "baseline": {
        "filename": "fashion_cnn_baseline.pth",
        "conv1": 16,
        "conv2": 32
    },
    "pruned1": {
        "filename": "fashion_cnn_pruned_12ch_finetuned.pth",
        "conv1": 12,
        "conv2": 32
    },
    "pruned2": {
        "filename": "fashion_cnn_pruned_12_24ch_finetuned.pth",
        "conv1": 12,
        "conv2": 24
    }
}


# 加载全部模型
models = {}

for name, config in model_configs.items():

    print(f"\n正在加载模型: {name}")

    model = FashionCNN(
        conv1_channels=config["conv1"],
        conv2_channels=config["conv2"]
    ).to(device)

    weight_path = project_root / config["filename"]

    state_dict = torch.load(
        weight_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    models[name] = model

    print("模型加载成功:", name)


# 使用模拟图片测试三个模型
test_image = torch.randn(
    1, 1, 28, 28,
    device=device
)

with torch.inference_mode():

    for name, model in models.items():

        output = model(test_image)

        print(
            f"{name} 推理成功，输出形状:",
            output.shape
        )

print("\n三个模型全部测试完成！")
