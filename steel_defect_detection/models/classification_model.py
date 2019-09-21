import torch.nn as nn
from torchvision.models import resnet34

from model_utils import ClassificationHeadResNet

class ResNet34(nn.Module):
    def __init__(self, pre=True, num_classes=4, use_simple_head=True):
        super().__init__()
        encoder = resnet34(pretrained=pre)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = encoder.bn1
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer0 = nn.Sequential(self.conv1,self.relu,self.bn1,self.maxpool)
        self.layer1 = encoder.layer1
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4
        self.head = ClassificationHeadResNet(num_classes, use_simple_head=use_simple_head)

    def forward(self, x):
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.head(x)
        return x

if __name__ == "__main__":
    from torchsummary import summary
    model = ResNet34(pre=None, num_classes=4, use_simple_head=True)
    print(model)
