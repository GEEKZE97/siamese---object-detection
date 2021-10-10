import cv2
from xml.dom import minidom
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import random


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 输入图像channel：1；输出channel：6；5x5卷积核
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.conv3 = nn.Conv2d(16, 24, 5, padding=2)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(24 * 31 * 31, 1200)  # 16 * 59 * 59
        self.fc2 = nn.Linear(1200, 84)
        self.fc3 = nn.Linear(84, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        # 2x2 Max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # 如果是方阵,则可以只使用一个数字进行定义
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


dataX = []
dataY = []
for j in range(8):
    for i in range(6):
        for k in range(24):
            addr = "/home/yanze/Desktop/Python/训练数据/Image248All/img" + str(i) + "%%" + str(j) + "%" + str(k) + ".bmp"
            if os.path.exists(addr):
                print(1)
                if os.path.exists(
                        "/home/yanze/Desktop/Python/训练数据/Image248AllXml/img" + str(i) + "%%" + str(j) + "%" + str(
                            k) + ".xml"):
                    print(2)
                    imgNumpy = cv2.imread(addr, 0)
                    dataX.append(imgNumpy)
                    dataY.append(1.0)
                    img_H = cv2.flip(imgNumpy, 1, dst=None)  # 水平
                    img_V = cv2.flip(imgNumpy, 0, dst=None)  # 垂直
                    img_HV = cv2.flip(imgNumpy, -1, dst=None)  # 对角
                    dataX.append(img_H)
                    dataX.append(img_V)
                    dataX.append(img_HV)
                    dataY.append(1.0)
                    dataY.append(1.0)
                    dataY.append(1.0)
                else:
                    imgNumpy = cv2.imread(addr, 0)
                    dataX.append(imgNumpy)
                    dataY.append(0.0)

XandY = list(zip(dataX, dataY))
random.shuffle(XandY)
dataX[:], dataY[:] = zip(*XandY)

print(len(dataY))

net = Net()
criterion = nn.BCELoss()
# optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=1e-5)
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-5)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
# optimizer = torch.optim.Adam(net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

for epoch in range(5):
    for i in range(100):
        # imgNumpy = cv2.imread(dataX[i], 0)
        imgTorch = torch.from_numpy(dataX[i].reshape(1, 1, 248, 248)).float()
        # imgTorch = transforms.Normalize((0.5,), (1,))(imgTorch)
        y_pred = net(imgTorch)
        y_data = torch.tensor([[dataY[i]]])
        loss = criterion(y_pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, loss.item(), y_pred.detach().numpy(), y_data)
torch.save(net,"/home/yanze/Desktop/Python/object1.pt")
