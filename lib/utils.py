from torchvision import models
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import numpy as np

# TODO Do we need heatmap -> bboxes
# TODO Do we need IOU?
# TODO ask about the bbox: top-down?
# TODO FPR and FNR respectively to measure over- and under-predicted areas.

class FeatureExtrator(ResNet):
    def __init__(self):
        super(FeatureExtrator, self).__init__(Bottleneck, [3, 4, 6, 3])
        # self.channels = self.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # low level features for MI computation
        low = x

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return low, x

    def get_channel_num():
        return 512 * Bottleneck.expansion

def get_feature_extractor():
    model = FeatureExtrator()
    model.load_state_dict(models.resnet50(pretrained=True).state_dict())
    return model

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

# NOTE tensors proved to be easier to use than an object class
class BBoxInfo():
    def __init__(self, disease, x, y, w, h):
        self.original = 1024
        self.name = disease
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __str__(self):
        return "Name: " + self.name + " X: " + str(self.x) + " Y: " + str(self.y) + " W " + str(self.w) + " H " + str(self.h)

    def resize(self, size):
        ratio = size/self.original
        self.x = int(self.x * ratio)
        self.y = int(self.y * ratio)
        self.w = int(self.w * ratio)
        self.h = int(self.h * ratio)
