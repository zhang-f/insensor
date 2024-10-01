import torch
from torchvision import datasets, transforms

# 数据转换器
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 下载并加载训练集
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# 加载测试集
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 数据加载器
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)