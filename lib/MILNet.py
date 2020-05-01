from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torchvision.models.resnet import conv1x1
from lib.mi_networks import *

# TODO
# Add MI computation network
# Add MI losses

class MILNet(nn.Module):
    def __init__(self, feature_extractor, classifier, t = 0.5):
        self.fe = feature_extractor
        self.cnet = classifier # classifier network: cannot use pretrained, maybe resnet18?
        self.mask_generator = conv1x1(feature_extractor.get_channel_num, 1)

    def forward(self, x):
        shape = x.shape[2:]
        low, feat = self.fe(x) # the output of this layer is preserved for local MI maximization and global MI minimization:I(z,x)
        z = self.mask_generator(feat)
        m = F.interpolate(z, shape, mode = 'bicubic') # Is there a better mode for interpolate instead of bicubic?
        m = F.relu(torch.sigmoid(m) - t)
        x = x * m  # NOTE be sure their shape matched here.
        y = self.cnet(x)
        return low, z, y, m

class MIEncoder(nn.Module):
    def __init__(self, h, w, mi_units = 64):
        self.Xnet = MI1x1ConvNet() # NOTE should I simply do resent here according to the imp?
        self.Xnet = nn.Linear(mi_units, mi_units) # NOTE Or even just pooling + flatten + linear?
        self.Zlayer = nn.Linear(h*w, mi_units)
        self.ZXlayer_1 = nn.Linear(mi_units, mi_units) # NOTE Can MI be minimized??
        self.ZXlayer_2 = nn.Linear(mi_units)
        self.ZYlayer = nn.Linear(mi_units, 1) # For multi-class problem this should be 14*1 vector
        self.bn = BatchNorm1d(mi_units)
        self.relu = nn.ReLU()
        self.Ynet = nn.Linear(1, 1)

    def forward(self, x, z, y):
        z = self.relu(self.bn(self.Zlayer(z)))
        zy = self.ZYlayer(z)
        zx = self.relu(self.bn(self.ZXlayer_1(z)))
        y = y.unsqueeze(1)
        y = self.Ynet(y)
        return x, zx, zy, y
