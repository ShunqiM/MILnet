from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torchvision.models.resnet import conv1x1, resnet18
from lib.mi_networks import *

# TODO
# Concate / Add the orignal X together with the masked X to improve performance
# Double check do I have a wrong loss function

class LinearSeq(nn.Module):
    def __init__(self, in_units, out_units):
        super(LinearSeq, self).__init__()
        self.linear = nn.Linear(in_units, out_units)
        self.norm = nn.BatchNorm1d(out_units)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.linear(x)))


class MILNet(nn.Module):
    def __init__(self, feature_extractor, classifier, t = 0.5):
        super(MILNet, self).__init__()
        self.fe = feature_extractor
        self.cnet = classifier # classifier network: cannot use pretrained, maybe resnet18?
        self.mask_generator = conv1x1(feature_extractor.get_channel_num(), 1)
        self.t = t

    def forward(self, x):
        shape = x.shape[2:]
        low, feat = self.fe(x) # the output of this layer is preserved for local MI maximization and global MI minimization:I(z,x)
        z = self.mask_generator(feat)
        m = F.interpolate(z, shape, mode = 'bicubic') # Is there a better mode for interpolate instead of bicubic?
        m = F.relu(torch.sigmoid(m) - self.t)
        x = x * m  # NOTE be sure their shape matched here.
        y = self.cnet(x)
        return low, z, y, m

class MIEncoder(nn.Module):
    def __init__(self, h, w, in_channel, mi_units = 64):
        super(MIEncoder, self).__init__()
        self.Xnet_1 = resnet18(False, True, num_classes = mi_units) # NOTE should I simply do resent here according to the imp?
        self.Xnet_1.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)

        self.Xnet_2 = LinearSeq(mi_units, mi_units) # NOTE Or even just pooling + flatten + linear?
        self.Xnet_3 = LinearSeq(mi_units, mi_units)
        self.Zlayer = LinearSeq(h * w, mi_units)
        self.ZXlayer_1 = LinearSeq(mi_units, mi_units) # NOTE Can MI be minimized??
        self.ZXlayer_2 = LinearSeq(mi_units, mi_units)
        self.ZYlayer_1 = LinearSeq(mi_units, mi_units) # For multi-class problem this should be 14*1 vector
        self.ZYlayer_2 = nn.Linear(mi_units, 1)
        self.Ynet_1 = LinearSeq(1, 1)
        self.Ynet_2 = nn.Linear(1, 1)

    def forward(self, x, z, y):
        N, C, H, W = z.size()
        # z = z.view(N, 1, C * H * W)
        z = z.view(N, -1)
        x = self.Xnet_3(self.Xnet_2(self.Xnet_1(x)))
        z = self.Zlayer(z)
        zy = self.ZYlayer_2(self.ZYlayer_1(z))
        zx = self.ZXlayer_2(self.ZXlayer_1(z))
        y = y.unsqueeze(1)
        y = self.Ynet_2(self.Ynet_1(y))
        return x, zx, zy, y
