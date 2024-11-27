import torchvision.models as models
from torch import nn


class MobileNetv3(nn.Module):
    def __init__(self,slice):
        super(MobileNetv3,self).__init__()
        self.model = None
        if slice == 1:
            self.model = models.mobilenet_v3_small(pretrained=True).features[:4]
        elif slice == 2:
            self.model = models.mobilenet_v3_small(pretrained=True).features[4:9]
        else:
            self.model = models.mobilenet_v3_small(pretrained=True).features[9:]
    def forward(self,x):
        return self.model(x)