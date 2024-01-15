
import torch
import torch.nn as nn
from networks.resnet import resnet50

class fusing_resnet(nn.Module):
    def __init__(self) :
        super().__init__()
        self.resnet_feature_extractor_rgb = nn.Sequential(*list(resnet50().children())[:-2])
        self.resnet_feature_extractor_hsv = nn.Sequential(*list(resnet50().children())[:-2])
        # self.model.fc = nn.Linear(512, 1)
        self.avgpool=  nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(4096, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        # self.sigmoid= nn.Sigmoid()

    def forward(self, input_rgb, input_hsv):
        x = self.resnet_feature_extractor_rgb(input_rgb)
        y = self.resnet_feature_extractor_hsv(input_hsv)
        z = torch.cat((x, y), dim=1)
        z=self.avgpool(z)
        z = z.view(z.size(0), -1)
        z=self.fc1(z)
        z=self.relu(z)
        z=self.fc2(z)
        # z=self.sigmoid(z)

        return z

class fusing_resnet_add(nn.Module):
    def __init__(self) :
        super().__init__()
        self.resnet_feature_extractor_rgb = nn.Sequential(*list(resnet50().children())[:-2])
        self.resnet_feature_extractor_hsv = nn.Sequential(*list(resnet50().children())[:-2])
        # self.model.fc = nn.Linear(512, 1)
        self.avgpool=  nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 1)
        # self.sigmoid= nn.Sigmoid()

    def forward(self, input_rgb, input_hsv):
        x = self.resnet_feature_extractor_rgb(input_rgb)
        y = self.resnet_feature_extractor_hsv(input_hsv)
        z = torch.add(x, y)
        z=self.avgpool(z)
        z = z.view(z.size(0), -1)
        z=self.fc1(z)
        z=self.relu(z)
        z=self.fc2(z)
        # z=self.sigmoid(z)

        return z        