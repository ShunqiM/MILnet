import os
import shutil
from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable, Function
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck
import numpy as np

class FeatureExtrator(ResNet):
    def __init__(self):
        super(FeatureExtrator, self).__init__(Bottleneck, [3, 4, 6, 3])
        # self.drop = nn.Dropout2d(p = 0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # low level features for MI computation
        low = x

        # x = self.drop(x)
        x = self.layer2(x)
        # x = self.drop(x)
        x = self.layer3(x)
        # x = self.drop(x)
        x = self.layer4(x)

        return low.detach(), x

    def get_channel_num(self):
        return 512 * Bottleneck.expansion

    def get_local_channel_num(self):
        return self.layer2[0].conv1.in_channels

def get_feature_extractor():
    model = FeatureExtrator()
    model.load_state_dict(models.resnet50(pretrained=True).state_dict())
    return model


class GradReverse(Function):
    def forward(self, x):
        return x

    def backward(self, grad_output):
        return (-grad_output)

def grad_reverse(x):
    return GradReverse()(x)

class RevGrad(Function):
    @staticmethod
    def forward(ctx, input_):
        ctx.save_for_backward(input_)
        output = input_
        return output

    @staticmethod
    def backward(ctx, grad_output):  # pragma: no cover
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = -grad_output
        return grad_input

class GRL(Function):
    Lambda = 1

    def __init__(self, L):
        super(GRL, self).__init__()
        GRL.Lambda = L
    @staticmethod
    def forward(self, x):
        return x.view_as(x)
    @staticmethod
    def backward(self, grad_output):
        grad_input = grad_output.clone()
        # return grad_input*(-0.5)
        # print(GRL.Lambda)
        return grad_input*(-GRL.Lambda)
    def set_lambda(self, L):
        GRL.Lambda = L


class GaussianNoise(nn.Module):
    """Gaussian noise regularizer."""

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach
        self.noise = torch.tensor(0).to(device)

    def forward(self, x):
        if self.training and self.sigma != 0:
            scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
            sampled_noise = self.noise.repeat(*x.size()).normal_() * scale
            x = x + sampled_noise
        return x


def freeze_network(model):
    for name, p in model.named_parameters():
        p.requires_grad = False

def unfreeze_network(model):
    for name, p in model.named_parameters():
        p.requires_grad = True

def normalize(a):
    AA = a.clone()
    b, c, h, w = AA.shape
    AA = AA.view(b, -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(b, c, h, w)
    return AA

def normalize_(AA):
    b, c, h, w = AA.shape
    AA = AA.view(b, -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(b, c, h, w)
    return AA


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "D:\\X\\2019S2\\3912\\MILN_models\\"
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, directory + 'model_best.pth.tar')
    print("model saved")

def load_checkpoint(model, mi_encoder, optimizer, scheduler, losslogger, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        mi_encoder.load_state_dict(checkpoint['mi_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # best_auc = checkpoint['best_prec1']
        scheduler.load_state_dict(checkpoint['scheduler'])
        # scheduler = checkpoint['scheduler']
        best_auc = checkpoint['best_auc']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, mi_encoder, optimizer, start_epoch, best_auc, scheduler


def adjust_learning_rate_(optimizer, epoch, logger, par_set):
    print("/Users/LULU/3912/tb/" + par_set)
    for param_group in optimizer.param_groups:
        # print(param_group['lr'])
        param_group['lr'] = param_group['lr']*0.1
        lr = param_group['lr']
        logger.log_value('learning_rate', lr, epoch)


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
