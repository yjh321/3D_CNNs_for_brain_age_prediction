import os
import torch.nn as nn
import torch


class CNN3D(nn.Module):
    def __init__(self, n_classes=1):
        super(CNN3D,self).__init__()
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, stride=(1,1,1), padding=1)
        self.conv2 = nn.Conv3d(8, 8, kernel_size=3, stride=(1,1,1), padding=1)
        self.conv3 = nn.Conv3d(8, 16, kernel_size=3, stride=(1,1,1), padding=1)
        self.conv4 = nn.Conv3d(16, 16, kernel_size=3, stride=(1,1,1), padding=1)
        self.conv5 = nn.Conv3d(16, 32, kernel_size=3, stride=(1,1,1), padding=1)
        self.conv6 = nn.Conv3d(32, 32, kernel_size=3, stride=(1,1,1), padding=1)
        self.conv7 = nn.Conv3d(32, 64, kernel_size=3, stride=(1,1,1), padding=1)
        self.conv8 = nn.Conv3d(64, 64, kernel_size=3, stride=(1,1,1), padding=1)
        self.conv9 = nn.Conv3d(64, 128, kernel_size=3, stride=(1,1,1), padding=1)
        self.conv10 = nn.Conv3d(128, 128, kernel_size=3, stride=(1,1,1), padding=1)

        self.batchnorm1 = nn.BatchNorm3d(8)
        self.batchnorm2 = nn.BatchNorm3d(16)
        self.batchnorm3 = nn.BatchNorm3d(32)
        self.batchnorm4 = nn.BatchNorm3d(64)
        self.batchnorm5 = nn.BatchNorm3d(128)

        self.maxpool = nn.MaxPool3d(kernel_size=(2,2,2))
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(4608, 4608)
        self.fc2 = nn.Linear(4608, 4608)
        self.fc3 = nn.Linear(4608, n_classes)
        self.act = nn.Sigmoid()

    def forward(self,x):
        # input size = (B, 1, 121, 145, 121)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv5(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv7(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.conv8(x)
        x = self.batchnorm4(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv9(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.conv10(x)
        x = self.batchnorm5(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.act(x)
   
        return x
