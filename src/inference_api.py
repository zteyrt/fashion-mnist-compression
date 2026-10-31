
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
# 1. 项目路径配置
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
# 3. 三个 CNN 模型的配置
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
# 4. 加载三个 CNN 模型
# ==========================================

models = {}

print("\n========== 正在加载 CNN 模型 ==========")

for model_name, config in model_configs.items():

    # 模型权重文件的完整路径
    weight_path = project_root / config["filename"]

    # 检查模型权重是否存在
    if not weight_path.exists():
        raise FileNotFoundError(
            f"模型权重文件不存在: {weight_path}"
        )

    # 根据通道数创建对应 CNN 结构
    model = FashionCNN(
        conv1_channels=config["conv1"],
        conv2_channels=config["conv2"]
    ).to(device)

    # 读取模型参数
    state_dict = torch.load(
        weight_path,
        map_location=device,
        weights_only=True
    )

    # 加载训练好的参数
    model.load_state_dict(state_dict)

    # 设置为推理模式
    model.eval()

    # 加入模型字典
    models[model_name] = model

    print(
        f"模型加载成功: {model_name} "
        f"({config['display_name']})"
    )

print("三个模型全部加载完成！")
print("====================================\n")


# ==========================================
# 5. Fashion-MNIST 十个类别
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
# 6. 图片预处理配置
# ==========================================

transform = transforms.ToTensor()

MAX_FILE_SIZE = 5 * 1024 * 1024


# ==========================================
# 7. 创建 FastAPI 应用
# ==========================================

app = FastAPI(
    title="Fashion-MNIST CNN Classification API",
    description="支持原始 CNN 和两阶段结构化剪枝 CNN 的服饰分类接口",
    version="2.0.0"
)


# ==========================================
# 8. 挂载静态文件
# ==========================================

app.mount(
    "/static",
    StaticFiles(directory=str(web_dir)),
    name="static"
)


# ==========================================
# 9. Web 页面
# ==========================================

@app.get("/web")
def web_page():

    html_path = web_dir / "index.html"

    return FileResponse(html_path)


# ==========================================
# 10. API 首页
# ==========================================

@app.get("/")
def home():

    return {
        "message": "Fashion-MNIST CNN API is running",
        "device": str(device),
        "available_models": list(models.keys())
    }


# ==========================================
# 11. 健康检查
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
# 12. 获取模型列表
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
# 13. 公共函数：读取并预处理上传图片
# ==========================================

async def preprocess_uploaded_image(file: UploadFile):

    # --------------------------------------
    # 13.1 读取文件内容
    # --------------------------------------

    try:
        contents = await file.read()
    finally:
        await file.close()

    # --------------------------------------
    # 13.2 检查文件是否为空
    # --------------------------------------

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="上传的图片为空"
        )

    # --------------------------------------
    # 13.3 检查文件大小
    # --------------------------------------

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="图片大小不能超过 5MB"
        )

    # --------------------------------------
    # 13.4 验证并转换图片
    # --------------------------------------

    try:

        with Image.open(io.BytesIO(contents)) as image:

            # 检查图片真实格式
            if image.format not in ("PNG", "JPEG"):
                raise HTTPException(
                    status_code=400,
                    detail="仅支持 PNG 或 JPG 图片"
                )

            # 转换为单通道灰度图
            image = image.convert("L")

            # 调整为 Fashion-MNIST 输入尺寸
            image = image.resize(
                (28, 28),
                Image.Resampling.LANCZOS
            )

            # 转换为 PyTorch Tensor
            input_tensor = transform(image)

            # [1, 28, 28] -> [1, 1, 28, 28]
            input_tensor = input_tensor.unsqueeze(0)

            # 移动到 CPU 或 CUDA
            input_tensor = input_tensor.to(device)

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="无法识别图片文件"
        )

    return input_tensor


# ==========================================
# 14. 公共函数：单个模型推理
# ==========================================

def run_model_inference(model_name, input_tensor):

    # 检查模型名称
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=(
                "无效的模型名称。"
                "可选值: baseline, pruned1, pruned2"
            )
        )

    selected_model = models[model_name]

    # --------------------------------------
    # 14.1 模型预测
    # --------------------------------------

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

    # --------------------------------------
    # 14.2 处理预测结果
    # --------------------------------------

    predicted_index = predicted_idx.item()

    predicted_class = class_names[predicted_index]

    confidence_percent = round(
        confidence.item() * 100,
        2
    )

    all_probabilities = {
        class_names[i]: round(
            probabilities[i].item() * 100,
            2
        )
        for i in range(len(class_names))
    }

    # 统计模型参数量
    total_params = sum(
        parameter.numel()
        for parameter in selected_model.parameters()
    )

    # --------------------------------------
    # 14.3 返回模型预测数据
    # --------------------------------------

    return {
        "model_name": model_name,
        "model_display_name": model_configs[model_name]["display_name"],
        "predicted_class": predicted_class,
        "class_index": predicted_index,
        "confidence": confidence_percent,
        "parameters": total_params,
        "probabilities": all_probabilities
    }


# ==========================================
# 15. 单模型预测接口
# ==========================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_name: str = Form("baseline")
):

    # 先检查模型名称
    if model_name not in models:
        raise HTTPException(
            status_code=400,
            detail=(
                "无效的模型名称。"
                "可选值: baseline, pruned1, pruned2"
            )
        )

    # 使用公共函数预处理图片
    input_tensor = await preprocess_uploaded_image(file)

    # 使用指定模型预测
    prediction = run_model_inference(
        model_name,
        input_tensor
    )

    # 保留原来的 API 返回格式
    return {
        "filename": file.filename,
        **prediction,
        "device": str(device)
    }


# ==========================================
# 16. 三模型同时对比接口
# ==========================================

@app.post("/compare")
async def compare_models(
    file: UploadFile = File(...)
):

    # 只需要预处理一次图片
    input_tensor = await preprocess_uploaded_image(file)

    comparison_results = []

    # 依次对三个模型进行推理
    for model_name in models.keys():

        prediction = run_model_inference(
            model_name,
            input_tensor
        )

        comparison_results.append(prediction)

    return {
        "filename": file.filename,
        "device": str(device),
        "results": comparison_results
    }


# ==========================================
# 17. 本地启动入口
# ==========================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8000
    )
