
import pytest
from fastapi.testclient import TestClient

from src.inference_api import app


# 创建 API 测试客户端
client = TestClient(app)


# ==========================================
# 1. 测试健康检查接口
# ==========================================

def test_health():

    response = client.get("/health")

    # 检查 HTTP 状态码
    assert response.status_code == 200

    data = response.json()

    # 检查服务状态
    assert data["status"] == "ok"

    # 检查模型是否加载
    assert data["model_loaded"] is True

    # 检查三个模型名称
    assert set(data["available_models"]) == {
        "baseline",
        "pruned1",
        "pruned2"
    }


# ==========================================
# 2. 测试模型列表接口
# ==========================================

def test_models():

    response = client.get("/models")

    assert response.status_code == 200

    data = response.json()

    assert len(data["models"]) == 3

    model_names = {
        model["model_name"]
        for model in data["models"]
    }

    assert model_names == {
        "baseline",
        "pruned1",
        "pruned2"
    }


# ==========================================
# 3. 测试非法模型名称
# ==========================================

def test_invalid_model_name():

    response = client.post(
        "/predict",
        data={"model_name": "wrong_model"},
        files={
            "file": (
                "test.png",
                b"fake image data",
                "image/png"
            )
        }
    )

    # 非法模型名称应该返回 HTTP 400
    assert response.status_code == 400


# ==========================================
# 4. 测试真实图片的单模型预测
# ==========================================

def test_predict_real_image():

    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    image_path = project_root / "assets" / "fashion_mnist_sample.png"

    # 读取真实测试图片
    with open(image_path, "rb") as image_file:

        response = client.post(
            "/predict",
            data={
                "model_name": "baseline"
            },
            files={
                "file": (
                    "fashion_mnist_sample.png",
                    image_file,
                    "image/png"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    # 检查使用的模型
    assert data["model_name"] == "baseline"

    # 这张示例图片的真实类别为 Ankle boot
    assert data["predicted_class"] == "Ankle boot"

    # 置信度应在 0～100 之间
    assert 0 <= data["confidence"] <= 100

    # 应该返回十个类别的概率
    assert len(data["probabilities"]) == 10


# ==========================================
# 5. 测试三模型同时对比
# ==========================================

def test_compare_three_models():

    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent

    image_path = project_root / "assets" / "fashion_mnist_sample.png"

    with open(image_path, "rb") as image_file:

        response = client.post(
            "/compare",
            files={
                "file": (
                    "fashion_mnist_sample.png",
                    image_file,
                    "image/png"
                )
            }
        )

    assert response.status_code == 200

    data = response.json()

    # 必须返回三个模型
    assert len(data["results"]) == 3

    # 检查三个模型名称
    model_names = {
        model["model_name"]
        for model in data["results"]
    }

    assert model_names == {
        "baseline",
        "pruned1",
        "pruned2"
    }

    # 检查每个模型的数据
    for model in data["results"]:

        assert 0 <= model["confidence"] <= 100

        assert model["parameters"] > 0

        assert len(model["probabilities"]) == 10


# ==========================================
# 6. 测试损坏的图片文件
# ==========================================

def test_invalid_image():

    response = client.post(
        "/predict",
        data={
            "model_name": "baseline"
        },
        files={
            "file": (
                "invalid.png",
                b"this is not a real image",
                "image/png"
            )
        }
    )

    assert response.status_code == 400
