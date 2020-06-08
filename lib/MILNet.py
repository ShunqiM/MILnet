from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from torchvision.models.resnet import conv1x1, resnet18, ResNet, BasicBlock, Bottleneck
from lib.mi_networks import *
from lib.utils import grad_reverse, GRL, normalize_, normalize

# TODO
# Concate / Add the orignal X together with the masked X to improve performance
# Should I remove unnecessarry low feature preservation?

class LinearSeq(nn.Module):
    def __init__(self, in_units, out_units):
        super(LinearSeq, self).__init__()
        self.linear = nn.Linear(in_units, out_units)
        self.norm = nn.BatchNorm1d(out_units)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.linear(x)))


class MILNet(nn.Module):
    def __init__(self, feature_extractor, classifier, t = 0.5, zt = 0.2):
        super(MILNet, self).__init__()
        self.mask_generator = conv1x1(feature_extractor.get_channel_num(), 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        self.fe = feature_extractor
        self.cnet = classifier # classifier network: cannot use pretrained, maybe resnet18?
        self.t = t
        self.zt = zt
        self.norm = NopNet((2,3))

    def forward(self, x):
        shape = x.shape[2:]
        low, feat = self.fe(x) # the output of this layer is preserved for local MI maximization and global MI minimization:I(z,x)
        z = self.mask_generator(feat)
        # np.savetxt("tensorz.csv", z[0][0].detach().cpu().numpy(), delimiter=",")
        # exit()
        z = normalize(self.norm(z))
        m = F.interpolate(z, shape, mode = 'bicubic') # Is there a better mode for interpolate instead of bicubic?
        # m = F.relu(torch.sigmoid(m) - self.t)
        m = m - self.t
        x = x * m  # NOTE be sure their shape matched here.
        y = self.cnet(x)
        # threshold added
        # z = F.sigmoid(z) # z.shape = torch.Size([16, 1, 7, 7])
        # b, c, h, w = z.shape
        # z = z.view(b, c, -1)
        # z = F.softmax(z, dim = 2)
        # z = z.view(b, c, h, w)
        tmp = torch.zeros(z.shape)
        z = torch.where(z > self.zt, z, tmp)
        # z = F.relu(torch.sigmoid(z) - self.zt)
        # z = F.relu(z - self.zt)
        return low, z, y, m

class MIEncoder(nn.Module):
    def __init__(self, h, w, in_channel, mi_units = 64, Lambda = 1, compress = 1):
        super(MIEncoder, self).__init__()
        self.Xnet = XEncoder(mi_units, in_channel, compress)
        self.Zlayer = LinearSeq(h * w, mi_units)
        self.ZXlayer_1 = LinearSeq(mi_units, mi_units) # NOTE Can MI be minimized??
        self.ZXlayer_2 = LinearSeq(mi_units, mi_units)
        self.ZYlayer_1 = LinearSeq(mi_units, int(mi_units/8)) # For multi-class problem this should be 14*1 vector
        self.ZYlayer_2 = nn.Linear(int(mi_units/8) , 1)
        self.Ynet_1 = LinearSeq(1, 1)
        self.Ynet_2 = nn.Linear(1, 1)
        # self.grad_reverse = grad_reverse
        self.grl = GRL(Lambda)
        self.grad_reverse = self.grl.apply

    """ GRL is still needed to avoid two complete backward (the second backward is only on mi_net now) """
    def forward(self, x, z, y):
        N, C, H, W = z.size()
        # x = self.grad_reverse(x)
        z = z.view(N, -1)
        x = self.Xnet(x)
        z = self.Zlayer(z)
        zy = self.ZYlayer_2(self.ZYlayer_1(z))
        zx = self.ZXlayer_2(self.ZXlayer_1(self.grad_reverse(z)))
        y = y.unsqueeze(1)
        y = self.Ynet_2(self.Ynet_1(y))
        return x, zx, zy, y

    def update_GRL(self, delta):
        GRL.Lambda += delta
        self.grad_reverse = self.grl.apply

""" An convolutional encoder that is able to preserve spatial properties """
class XEncoder(ResNet):
    def __init__(self, mi_units, in_channel, compress, img_size = 224):
        super(XEncoder, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=1, zero_init_residual=True)
        self.in_channels = in_channel
        self.channel_merger = conv1x1(512, compress)
        self.out_bn = nn.BatchNorm2d(compress)
        self.conv1 = nn.Conv2d(self.in_channels, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        h, w = self.get_flattened_units(img_size)[2:]
        self.Xnet_1 = LinearSeq(h * w * compress, mi_units)
        self.Xnet_2 = LinearSeq(mi_units, mi_units)
        self.Xnet_3 = LinearSeq(mi_units, mi_units)


    def conv_forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.conv_forward(x)
        x = F.relu(self.out_bn(self.channel_merger(x)))
        x = torch.flatten(x, 1)
        x = self.Xnet_3(self.Xnet_2(self.Xnet_1(x)))
        return x

    def get_flattened_units(self, img_size):
        random = torch.randn(1, self.in_channels, 56, 56).float() # turn from 49 to 784
        # The out shape is b, 512, 14, 14
        shape = self.conv_forward(random).shape
        return shape
