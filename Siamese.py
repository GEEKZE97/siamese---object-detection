import numpy as np
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt


def show_plot(iteration, loss):
    # 绘制损失变化图
    plt.plot(iteration, loss)
    plt.show()


def read_image(filename, byteorder='>'):
    # first we read the image, as a raw file to the buffer
    with open(filename, 'rb') as f:
        buffer = f.read()

    # using regex, we extract the header, width, height and maxval of the image
    header, width, height, maxval = re.search(
        b"(^P5\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n])*"
        b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()

    # then we convert the image to numpy array using np.frombuffer which interprets buffer as one dimensional array
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def get_data(size, total_sample_size):
    # read the image
    image = read_image('D:/Final/Spyder/train/s' + str(1) + '/' + str(1) + '.pgm', 'rw+')
    # reduce the size
    image = image[::size, ::size]
    # get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]
    # print(dim1)
    # print(dim2)

    count = 0

    # initialize the numpy array with the shape of [total_sample, no_of_pairs, dim1, dim2]
    x_geuine_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])  # 2 is for pairs
    y_genuine = np.zeros([total_sample_size, 1])

    for i in range(40):
        for j in range(int(total_sample_size / 40)):
            ind1 = 0
            ind2 = 0

            # read images from same directory (genuine pair)
            while ind1 == ind2:
                ind1 = np.random.randint(10)
                ind2 = np.random.randint(10)

            # read the two images
            img1 = read_image('D:/Final/Spyder/train/s' + str(i + 1) + '/' + str(ind1 + 1) + '.pgm', 'rw+')
            img2 = read_image('D:/Final/Spyder/train/s' + str(i + 1) + '/' + str(ind2 + 1) + '.pgm', 'rw+')

            # reduce the size
            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            # store the images to the initialized numpy array
            x_geuine_pair[count, 0, 0, :, :] = img1
            x_geuine_pair[count, 1, 0, :, :] = img2

            # as we are drawing images from the same directory we assign label as 1. (genuine pair)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, 1, dim1, dim2])
    y_imposite = np.zeros([total_sample_size, 1])

    for i in range(int(total_sample_size / 10)):
        for j in range(10):

            # read images from different directory (imposite pair)
            while True:
                ind1 = np.random.randint(40)
                ind2 = np.random.randint(40)
                if ind1 != ind2:
                    break

            img1 = read_image('D:/Final/Spyder/train/s' + str(ind1 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')
            img2 = read_image('D:/Final/Spyder/train/s' + str(ind2 + 1) + '/' + str(j + 1) + '.pgm', 'rw+')

            img1 = img1[::size, ::size]
            img2 = img2[::size, ::size]

            x_imposite_pair[count, 0, 0, :, :] = img1
            x_imposite_pair[count, 1, 0, :, :] = img2
            # as we are drawing images from the different directory we assign label as 0. (imposite pair)
            y_imposite[count] = 0
            count += 1

    # now, concatenate, genuine pairs and imposite pair to get the whole data
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0) / 255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y


# 搭建模型
class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 56 * 46, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


# 自定义ContrastiveLoss
class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive


size = 2
total_sample_size = 300 # 100
X, Y = get_data(size, total_sample_size)
print(X.shape)
print(Y.shape)
XandY = list(zip(X, Y))
random.shuffle(XandY)
X[:], Y[:] = zip(*XandY)

train_batch_size = 32
train_number_epochs = 50 # 20

net = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)

counter = []
loss_history = []
iteration_number = 0

for epoch in range(0, train_number_epochs):
    for terms in range(5):
        lossss = 0
        for num in range(train_number_epochs):
            img0 = X[5 * terms + num][0]
            img1 = X[5 * terms + num][1]
            label = Y[5 * terms + num]

            templabel = label

            imgTorch0 = torch.from_numpy(img0.reshape(1, 1, 56, 46)).float()
            imgTorch1 = torch.from_numpy(img1.reshape(1, 1, 56, 46)).float()
            label = torch.tensor(label)
            optimizer.zero_grad()
            output1, output2 = net(imgTorch0, imgTorch1)
            loss_contrastive = criterion(output1, output2, label)

            euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
            if templabel == 0:
                f = open("D:/AOldOrdinateur/F/pourPaper/NET4.txt", 'a')
                f.write(str(euclidean_distance.item()))
                f.write("\n")
            else:
                f = open("D:/AOldOrdinateur/F/pourPaper/NET5.txt", 'a')
                f.write(str(euclidean_distance.item()))
                f.write("\n")

            print(loss_contrastive)
            lossss = lossss + loss_contrastive.item()
            print(lossss)
            loss_contrastive.backward()
            optimizer.step()


