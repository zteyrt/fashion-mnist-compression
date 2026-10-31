
import os

from torchvision import datasets

# 获取项目根目录
current_dir = os.path.dirname(
    os.path.abspath(__file__)
)

project_root = os.path.dirname(current_dir)

# 加载 Fashion-MNIST 测试集
test_dataset = datasets.FashionMNIST(
    root=os.path.join(project_root, "data"),
    train=False,
    download=True,
    transform=None
)

# Fashion-MNIST 类别
class_names = [
    "T-shirt_top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle_boot"
]

# 创建 assets 文件夹
assets_dir = os.path.join(
    project_root,
    "assets"
)

os.makedirs(assets_dir, exist_ok=True)

# 提取第一张测试图片
image, label = test_dataset[0]

# 保存图片
output_path = os.path.join(
    assets_dir,
    "fashion_mnist_sample.png"
)

image.save(output_path)

print("图片提取成功！")
print("真实类别:", class_names[label])
print("图片尺寸:", image.size)
print("保存位置:", output_path)
