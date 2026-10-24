import torch
from torch import nn

#创建神经网络类
class FashionCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=16,
            kernel_size=3,
            padding=1
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        #ReLU激活层
        self.relu = nn.ReLU()

        #最大池化层
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        
        #展平层
        self.flatten = nn.Flatten()

        #第一个全连接层
        self.fc1 = nn.Linear(
            in_features=1568,
            out_features=128
        )

        #第二个全连接层
        self.fc2 = nn.Linear(
            in_features=128,
            out_features=10
        )
        
    def forward(self,x):

        #第一步：卷积
        x = self.conv1(x)

        #第二步：ReLU激活
        x = self.relu(x)

        #第三步：最大池化
        x = self.pool(x)
        
        #第二个卷积模块
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        #将卷积特征展平
        x = self.flatten(x)

        #第一个连接层
        x = self.fc1(x)
        
        x = self.relu(x)
        #第二个连接层
        x = self.fc2(x)

        return x

if __name__ == "__main__":
    
    model = FashionCNN()

    x = torch.randn(64,1,28,28)

    output = model(x)
    
    total_params = sum(
        p.numel() for p in model.parameters()
    )

    print("输入形状:",x.shape)
    print("输出形状:",output.shape)
    print("模型参数总数:",total_params)
    
