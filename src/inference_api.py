
import io
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image, UnidentifiedImageError

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException,
    Form
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.model import FashionCNN


# ==========================================
# 1. 项目路径
# ==========================================

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

web_dir = project_root / "web"


# ==========================================
# 2. 选择运行设备
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("当前运行设备:", device)


# ==========================================
# 3. 配置三个 CNN 模型
# ==========================================

model_configs = {
    "baseline": {
        "filename": "fashion_cnn_baseline.pth",
        "conv1": 16,
        "conv2": 32,
        "display_name": "原始 CNN"
    },

    "pruned1": {
        "filename": "fashion_cnn_pruned_12ch_finetuned.pth",
        "conv1": 12,
        "conv2": 32,
        "display_name": "第一次剪枝并微调"
    },

    "pruned2": {
        "filename": "fashion_cnn_pruned_12_24ch_finetuned.pth",
        "conv1": 12,
        "conv2": 24,
        "display_name": "第二次剪枝并微调"
    }
}


# ==========================================
# 4. 加载三个模型
# ==========================================

models = {}

print("\n========== 正在加载 CNN 模型 ==========")

for model_name, config in model_configs.items():

    # 获取权重路径
    weight_path = project_root / config["filename"]

    # 检查权重文件
    if not weight_path.exists():
        raise FileNotFoundError(
            f"模型权重文件不存在: {weight_path}"
        )

    # 创建对应的 CNN 结构
    model = FashionCNN(
        conv1_channels=config["conv1"],
        conv2_channels=config["conv2"]
    ).to(device)

    # 读取权重
    state_dict = torch.load(
        weight_path,
        map_location=device,
        weights_only=True
    )

    # 加载权重并进入推理模式
    model.load_state_dict(state_dict)
    model.eval()

    # 保存已加载的模型
    models[model_name] = model

    print(
        f"模型加载成功: {model_name} "
        f"({config['display_name']})"
    )

print("三个模型全部加载完成！")
print("====================================\n")


# ==========================================
# 5. 定义 Fashion-MNIST 类别
# ==========================================

class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]


# ==========================================
# 6. 图片预处理
# ==========================================

transform = transforms.ToTensor()


# ==========================================
# 7. 创建 FastAPI 应用
# ==========================================

app = FastAPI(
    title="Fashion-MNIST CNN Classification API",
    description="支持原始 CNN 和结构化剪枝 CNN 的服饰分类接口",
    version="2.0.0"
)


# ==========================================
# 8. 提供静态网页文件
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory=str(web_dir)),
    name="static"
)


@app.get("/web")
def web_page():

    html_path = web_dir / "index.html"

    return FileResponse(html_path)


# ==========================================
# 9. API 首页
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Fashion-MNIST CNN API is running",
        "device": str(device),
        "available_models": list(models.keys())
    }


# ==========================================
# 10. 健康检查接口
# ==========================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(device),
        "available_models": list(models.keys())
    }


# ==========================================
# 11. 获取模型列表接口
# ==========================================

@app.get("/models")
def get_models():

    return {
        "models": [
            {
                "model_name": name,
                "display_name": config["display_name"],
                "conv1_channels": config["conv1"],
                "conv2_channels": config["conv2"]
            }
            for name, config in model_configs.items()
        ]
    }


# ==========================================
# 12. 图片预测接口
# ==========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("baseline")
):

    # ======================================
    # 12.1 检查模型名称
    # ======================================

    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=(
                "无效的模型名称。"
                "可选值: baseline, pruned1, pruned2"
            )
        )

    # 选择对应模型
    selected_model = models[model_name]

    # ======================================
    # 12.2 读取上传文件
    # ======================================

    try:
        contents = await file.read()
    finally:
        await file.close()

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="上传的图片为空"
        )

    # 限制文件大小为 5MB
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="图片大小不能超过 5MB"
        )

    # ======================================
    # 12.3 读取和转换图片
    # ======================================

    try:
        with Image.open(io.BytesIO(contents)) as image:

            if image.format not in ("PNG", "JPEG"):
                raise HTTPException(
                    status_code=400,
                    detail="仅支持 PNG 或 JPG 图片"
                )

            # 转换为灰度图
            image = image.convert("L")

            # 调整图片尺寸
            image = image.resize(
                (28, 28),
                Image.Resampling.LANCZOS
            )

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="无法识别图片文件"
        )

    # ======================================
    # 12.4 图片转换为 Tensor
    # ======================================

    input_tensor = transform(image)

    # [1,28,28] -> [1,1,28,28]
    input_tensor = input_tensor.unsqueeze(0).to(device)

    # ======================================
    # 12.5 模型推理
    # ======================================

    with torch.inference_mode():

        outputs = selected_model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[0]

        confidence, predicted_idx = torch.max(
            probabilities,
            dim=0
        )

    # ======================================
    # 12.6 获取预测结果
    # ======================================

    predicted_index = predicted_idx.item()

    predicted_class = class_names[predicted_index]

    confidence_percent = confidence.item() * 100

    all_probabilities = {
        class_names[i]: round(
            probabilities[i].item() * 100,
            2
        )
        for i in range(len(class_names))
    }

    # ======================================
    # 12.7 返回 JSON
    # ======================================

    return {
        "filename": file.filename,
        "model_name": model_name,
        "model_display_name": model_configs[model_name]["display_name"],
        "predicted_class": predicted_class,
        "class_index": predicted_index,
        "confidence": round(confidence_percent, 2),
        "device": str(device),
        "probabilities": all_probabilities
    }


# ==========================================
# 13. 本地启动入口
# ==========================================


# ==========================================
# 13. 三模型同时对比接口
# ==========================================

@app.post("/compare")
async def compare_models(file: UploadFile = File(...)):

    # 1. 读取上传的图片
    try:
        contents = await file.read()
    finally:
        await file.close()

    # 检查文件是否为空
    if not contents:
        raise HTTPException(
            status_code=400,
            detail="上传的图片为空"
        )

    # 限制图片最大为 5MB
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="图片大小不能超过 5MB"
        )

    # 2. 图片预处理
    try:
        with Image.open(io.BytesIO(contents)) as image:

            if image.format not in ("PNG", "JPEG"):
                raise HTTPException(
                    status_code=400,
                    detail="仅支持 PNG 或 JPG 图片"
                )

            image = image.convert("L")

            image = image.resize(
                (28, 28),
                Image.Resampling.LANCZOS
            )

            input_tensor = transform(image).unsqueeze(0).to(device)

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="无法识别图片文件"
        )

    # 3. 存储三个模型的预测结果
    comparison_results = []

    # 4. 依次调用三个模型
    with torch.inference_mode():

        for model_name, model in models.items():

            # 模型推理
            outputs = model(input_tensor)

            # 转换为概率
            probabilities = torch.softmax(
                outputs,
                dim=1
            )[0]

            # 找到概率最高的类别
            confidence, predicted_idx = torch.max(
                probabilities,
                dim=0
            )

            predicted_index = predicted_idx.item()

            # 统计模型参数量
            total_params = sum(
                p.numel()
                for p in model.parameters()
            )

            # 记录每个模型的结果
            comparison_results.append({
                "model_name": model_name,
                "model_display_name": model_configs[model_name]["display_name"],
                "predicted_class": class_names[predicted_index],
                "confidence": round(confidence.item() * 100, 2),
                "parameters": total_params,
                "probabilities": {
                    class_names[i]: round(
                        probabilities[i].item() * 100,
                        2
                    )
                    for i in range(len(class_names))
                }
            })

    # 5. 返回三个模型的结果
    return {
        "filename": file.filename,
        "device": str(device),
        "results": comparison_results
    }

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
