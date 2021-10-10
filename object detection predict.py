import cv2
from xml.dom import minidom
import os.path
import torch
import torch.nn as nn
import torch.nn.functional as F


# XML解析
def xmlAnalyze(filename):
    if os.path.exists(filename):
        xml = minidom.parse(filename)  # 打开xml文件
        root = xml.documentElement  # 获取根节点
        elements = root.getElementsByTagName('object')  # 子节点
        elements2 = elements[0].getElementsByTagName('bndbox')  # 子节点有多个，加[0]
        xmin = elements2[0].getElementsByTagName('xmin')
        ymin = elements2[0].getElementsByTagName('ymin')
        xmax = elements2[0].getElementsByTagName('xmax')
        ymax = elements2[0].getElementsByTagName('ymax')
        xmindata = xmin[0].childNodes[0].data
        ymindata = ymin[0].childNodes[0].data
        xmaxdata = xmax[0].childNodes[0].data
        ymaxdata = ymax[0].childNodes[0].data
        return [xmindata, ymindata, xmaxdata, ymaxdata]
    else:
        return [0, 0, 0, 0]


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
        self.fc4 = nn.Linear(10, 4)

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
        x = self.fc4(x)
        return x


dataX = []
dataY = []
addrX = []
addrY = []

for j in range(8):
    for i in range(6):
        for k in range(24):
            addr = "D:/AOldOrdinateur/F/pourPaper/Image248All/img" + str(i) + "%%" + str(j) + "%" + str(k) + ".bmp"
            if (os.path.exists(addr)):
                if os.path.exists(
                        "D:/AOldOrdinateur/F/pourPaper/Image248AllXml/img" + str(i) + "%%" + str(j) + "%" + str(
                            k) + ".xml"):
                    imgNumpy = cv2.imread(addr, 0)
                    dataX.append(imgNumpy)
                    dataY.append(
                        xmlAnalyze(
                            "D:/AOldOrdinateur/F/pourPaper/Image248AllXml/img" + str(i) + "%%" + str(j) + "%" + str(
                                k) + ".xml"))
                    addrX.append(addr)
                    addrY.append(
                        "D:/AOldOrdinateur/F/pourPaper/Image248AllXml/img" + str(i) + "%%" + str(j) + "%" + str(
                            k) + ".xml")

print(len(dataX))
print(len(dataY))
model = Net()
criterion = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
for epoch in range(300):
    for i in range(len(dataX)):
        imgNumpy = dataX[i]
        imgTorch = torch.from_numpy(imgNumpy.reshape(1, 1, 248, 248)).float()
        y_pred = model(imgTorch)
        y_data = torch.tensor(
            [[float(dataY[i][0]) / 248, float(dataY[i][1]) / 248, float(dataY[i][2]) / 248,
              float(dataY[i][3]) / 248]])
        loss = criterion(y_pred, y_data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, loss.item(), y_pred.detach().numpy() * 248, y_data.detach().numpy() * 248)
