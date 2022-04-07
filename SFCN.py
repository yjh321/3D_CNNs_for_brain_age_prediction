import os
import torch
import torch.nn as nn

class CNN3D(nn.Module):
    def __init__(self, n_classes = 20):
        super(CNN3D,self).__init__()
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3,stride=(1,1,1),padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3,stride=(1,1,1),padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3,stride=(1,1,1),padding=1)
        self.conv4 = nn.Conv3d(128, 256, kernel_size=3,stride=(1,1,1),padding=1)
        self.conv5 = nn.Conv3d(256, 256, kernel_size=3,stride=(1,1,1),padding=1)
        self.conv6 = nn.Conv3d(256, 64, kernel_size=1,stride=(1,1,1))

        self.batchnorm1 = nn.BatchNorm3d(32)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.batchnorm3 = nn.BatchNorm3d(128)
        self.batchnorm4 = nn.BatchNorm3d(256)
        self.batchnorm5 = nn.BatchNorm3d(256)
        self.batchnorm6 = nn.BatchNorm3d(64)

        self.maxpool = nn.MaxPool3d(kernel_size=(2,2,2),stride=(2,2,2))
        self.avgpool = nn.AvgPool3d(kernel_size=(5,6,5),stride=(1,1,1))
        self.dropout = nn.Dropout3d(p=0.5)
        self.relu = nn.ReLU()
        self.classifier = nn.Conv3d(64, n_classes, kernel_size=1, stride=(1,1,1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self,x):
        #input size = (B, 1, 160 , 192, 160)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        x = self.maxpool(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.batchnorm5(x)
        x = self.maxpool(x)
        x = self.relu(x)
        # feature map with (B, 256, 5, 6, 5)
        x = self.conv6(x)
        x = self.batchnorm6(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = self.dropout(x)
        x = self.classifier(x)
        x = x.view(x.shape[0], -1)

        return x
