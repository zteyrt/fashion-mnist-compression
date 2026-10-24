import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


#定义图片转换方式
transform = transforms.ToTensor()

#下载训练集
train_dataset = datasets.FashionMNIST(
    root = "./data",
    train = True,
    download = True,
    transform = transform
)

#下载测试集
test_dataset = datasets.FashionMNIST(
    root = "./data",
    train = False,
    download = True,
    transform = transform
)

print("训练集数量:", len(train_dataset))
print("测试集数量:", len(test_dataset))

#读取训练集中的第一章图片
image,label = train_dataset[0]

print("图片形状:",image.shape)
print("图片数据类型:",image.dtype)
print("图片标签:",label)
print("最小像素值：",image.min().item())
print("最大像素值：",image.max().item())
print("平均像素值：",image.mean().item())

#创建训练集DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size = 64,
    shuffle = True
)

images,labels = next(iter(train_loader))

print("批次图片形状:",images.shape)
print("批次标签形状:",labels.shape)
print("批次数量:",len(train_loader))

test_loader = DataLoader(
    test_dataset,
    batch_size = 64,
    shuffle = False
)

print("测试集批次数量:",len(test_loader))

