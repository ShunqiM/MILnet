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
from lib.utils import grad_reverse, GRL, normalize_, normalize, from_01, GradientMultiplier

# TODO
# Concate / Add the orignal X together with the masked X to improve performance
# Should I remove unnecessarry low feature preservation?

class LinearSeq(nn.Module):
    def __init__(self, in_units, out_units, drop_rate = 0.2):
        super(LinearSeq, self).__init__()
        # if in_units == 1:
        #     drop_rate = 0
        # self.drop = nn.Dropout(drop_rate)
        self.linear = nn.Linear(in_units, out_units)
        self.norm = nn.BatchNorm1d(out_units)
        # self.norm = nn.LayerNorm(out_units)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.linear(x)))
        # return self.relu(self.norm(self.linear(self.drop(x))))


class MILNet(nn.Module):
    def __init__(self, feature_extractor, classifier, t = 0.5, zt = 0.2):
        super(MILNet, self).__init__()
        self.mask_generator = conv1x1(feature_extractor.get_channel_num(), 1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain = 1)
        self.fe = feature_extractor
        self.cnet = classifier # classifier network: cannot use pretrained, maybe resnet18?
        self.t = t
        self.zt = zt
        self.norm = NopNet((2,3))

    def forward(self, x):
        low, feat = self.fe(x) # the output of this layer is preserved for local MI maximization and global MI minimization:I(z,x)
        # shape = low.shape[2:] # low.shape = 16,128,28,28
        shape = x.shape[2:]

        z = self.mask_generator(feat)

        # np.savetxt("tensorz.csv", z[0][0].detach().cpu().numpy(), delimiter=",")
        # np.savetxt("tensorznorm.csv", z[0][0].detach().cpu().numpy(), delimiter=",")
        # exit()
        m = F.interpolate(z, shape, mode = 'bicubic') # Is there a better mode for interpolate instead of bicubic?
        # m1 = F.relu(torch.sigmoid(m) - self.t)
        # x = x * m1 + x # NOTE be sure their shape matched here.
        y = self.cnet(x, z, self.t)

        # y = self.cnet(x, m1)

        # threshold added
        # z = F.sigmoid(z) # z.shape = torch.Size([16, 1, 7, 7])
        # tmp = torch.zeros(z.shape)
        # z = torch.where(z >= self.zt, z, tmp)

        return low, z, y, m

class MIEncoder(nn.Module):
    def __init__(self, h, w, in_channel, mi_units = 64, x_units = 32, Lambda = 1, compress = 1, y_weight = 0.1):
        super(MIEncoder, self).__init__()
        self.Xnet = XEncoder(x_units, in_channel, compress)
        # self.Xnet = LinearXEncoder(x_units, 7)
        self.Zlayer = LinearSeq(h * w, mi_units)
        self.ZXlayer_1 = LinearSeq(mi_units, int(mi_units)) # NOTE Can MI be minimized??
        self.ZXlayer_2 = LinearSeq(int(mi_units), x_units)
        self.ZYlayer_1 = LinearSeq(mi_units, int(mi_units/8)) # For multi-class problem this should be 14*1 vector
        self.ZYlayer_2 = nn.Linear(int(mi_units/8) , 1)
        self.Ynet_1 = LinearSeq(1, 1)
        self.Ynet_2 = nn.Linear(1, 1)
        # self.grad_reverse = grad_reverse
        self.grl = GRL(Lambda)
        self.grad_reverse = self.grl.apply
        self.gml = GradientMultiplier(y_weight)
        self.grad_multi = self.gml.apply

    """ GRL is still needed to avoid two complete backward (the second backward is only on mi_net now) """
    def forward(self, x, z, y):

        N, C, H, W = z.size()
        # x = self.grad_reverse(x)
        z = z.view(N, -1)
        x = self.Xnet(x)
        z = self.Zlayer(z)

        zy = self.ZYlayer_2(self.ZYlayer_1(self.grad_multi(z)))
        # zy = self.ZYlayer_2(self.ZYlayer_1(z))
        zx = self.ZXlayer_2(self.ZXlayer_1(self.grad_reverse(z)))
        y = y.unsqueeze(1)
        y = self.Ynet_2(self.Ynet_1(y))
        # y = y - 0.5
        # print(y)
        return x, zx, zy, y

    def update_GRL(self, delta):
        if GRL.Lambda >= 1:
            return
        # GRL.Lambda += delta
        self.grl.Lambda += delta
        self.grad_reverse = self.grl.apply

""" An convolutional encoder that is able to preserve spatial properties """
class XEncoder(ResNet):
    def __init__(self, mi_units, in_channel, compress, img_size = 224, block = BasicBlock, layers =  [2, 2, 2, 2]):
        super(XEncoder, self).__init__(BasicBlock, layers, num_classes=1, zero_init_residual=True)
        self.in_channels = in_channel
        # self.in_channels = 3
        self.channel_merger = conv1x1(64, compress)
        # z = torch.mean(feat, dim=1, keepdim=True)
        self.out_bn = nn.BatchNorm2d(compress)
        self.inplanes = 8
        self.conv1 = nn.Conv2d(self.in_channels, self.inplanes, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        replace_stride_with_dilation = [False, False, False]
        self.layer1 = self._make_layer(block, 8, layers[0])
        self.layer2 = self._make_layer(block, 16, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 32, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 64, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain = 1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # self.pol = nn.AdaptiveAvgPool2d((112, 112))

        h, w = self.get_flattened_units(img_size)[2:]
        self.Xnet_1 = LinearSeq(h * w * compress, mi_units)
        self.Xnet_2 = LinearSeq(mi_units, mi_units)
        # self.Xnet_3 = LinearSeq(mi_units, int(mi_units/2))
        self.Xnet_3 = LinearSeq(mi_units, mi_units)
        # self.Xnet = MI1x1ConvNet(512, mi_units)


    def conv_forward(self, x):
        # x = self.pol(x)

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
        # x = torch.mean(x, dim=1, keepdim=True)
        x = self.conv_forward(x)
        # x = torch.mean(x, dim=1, keepdim=True)
        # x = self.Xnet(x)
        x = F.relu(self.out_bn(self.channel_merger(x)))
        # x = F.relu((x))
        x = torch.flatten(x, 1)
        x = self.Xnet_3(self.Xnet_2(self.Xnet_1(x)))
        return x

    def get_flattened_units(self, img_size):
        random = torch.randn(1, self.in_channels, 56, 56).float() # turn from 49 to 784
        # random = torch.randn(1, self.in_channels, 224, 224).float() # turn from 49 to 784
        # The out shape is b, 512, 14, 14
        shape = self.conv_forward(random).data.shape
        print("X Shape: ", shape)
        return shape

class LinearXEncoder(nn.Module):
    def __init__(self, mi_units, size = 7):
        super(LinearXEncoder, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((size, size))
        self.Xnet_1 = LinearSeq(size * size, mi_units)
        self.Xnet_2 = LinearSeq(mi_units, mi_units)
        self.Xnet_3 = LinearSeq(mi_units, mi_units)

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdim=True)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.Xnet_3(self.Xnet_2(self.Xnet_1(x)))
        return x


class Classifier(ResNet):
    def __init__(self):
        super(Classifier, self).__init__(Bottleneck, [3, 4, 6, 3])

    def forward(self, x, z, t):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        shape = x.shape[2:]
        # m = m.view(b, m.size(2), m.size(3))
        m = F.interpolate(z, shape, mode = 'bicubic')
        m = F.relu(torch.sigmoid(m) - t)
        x = x * m + x

        x = self.layer3(x)
        x = self.layer4(x)


        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def get_classifier(channel):
    model = Classifier()
    model.load_state_dict(models.resnet50(pretrained=True).state_dict())
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model
