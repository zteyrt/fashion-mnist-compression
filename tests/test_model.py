
import torch
import pytest

from src.model import FashionCNN


# 测试三种 CNN 模型结构
@pytest.mark.parametrize(
    "conv1, conv2, expected_params",
    [
        (16, 32, 206922),
        (12, 32, 205730),
        (12, 24, 154682),
    ]
)
def test_model_structure(
    conv1,
    conv2,
    expected_params
):

    # 创建 CNN 模型
    model = FashionCNN(
        conv1_channels=conv1,
        conv2_channels=conv2
    )

    # 模拟一张 Fashion-MNIST 图片
    x = torch.randn(1, 1, 28, 28)

    # 执行前向传播
    model.eval()

    with torch.inference_mode():
        output = model(x)

    # 检查输出形状
    assert output.shape == (1, 10)

    # 检查模型参数数量
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    assert total_params == expected_params
