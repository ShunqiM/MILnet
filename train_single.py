import os
import time

import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torchvision.datasets as datasets
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from sklearn.metrics import *
from prefetch_generator import BackgroundGenerator
from tensorboard_logger import configure, log_value, Logger
import matplotlib.pyplot as plt

from lib.mi_loss import *
from lib.utils import *
from lib.evaluation_funtions import *

par_set = "h7"
alpha = 1
beta = 1
gamma = 0.05
THRESHOLD = 0.8
network_threshold = 0.2
mi_units = 64
x_units = 64
load_model = False
size = 224
lr = 0.01
mi_lr = 0.01
Lambda = 0.2
L2 = 3e-5
YWeight = 0.1
zt = 0
z_factor = 1
Compress = 1
bat = 16
validate_log_freq = 1600/bat
log_freq = 16000/bat
print_freq = 2

# TODO: 1. Dropout MI network  2. Dropout fe netowork 3. noisy network

def training(train_loader, model, mi_encoder, criterion, optimizer, mi_opt, epoch, logger, alpha, beta, measure = 'JSD'):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    predict_losses = AverageMeter()
    zx_losses = AverageMeter()
    zy_losses = AverageMeter()
    consistency_losses = AverageMeter()
    model.train()
    mi_encoder.train()
    end = time.time()
    bat = train_loader.batch_size

    # gml = GradientMultiplier(YWeight)
    # grad_multi = gml.apply

    # seg_crit = nn.BCELoss()
    for i, (input, target, name, bboxes) in enumerate(BackgroundGenerator(train_loader)):
        target_var = target.float().cuda(non_blocking=True)
        input_var = input.cuda(non_blocking=True)

        # flip and concat the input
        x_ = torch.flip(input_var, [3])
        x_in = torch.cat((input_var, x_), 0)
        # target_var = torch.cat((target_var, target_var), 0)


        x, z, output, m = model(x_in)

        # Attention Consistency Loss
        z1, z2 = torch.split(z, input.size(0))
        z_ = torch.flip(z2, [3])
        # x, z1, z_, output, m = model(x_in)
        consistency_loss = F.mse_loss(z1, z_)

        # xc, zx, zy, yc = mi_encoder(x, z, (target_var + grad_multi(torch.sigmoid((output)))/2))
        # xc, zx, zy, yc = mi_encoder(x, z, (target_var + (torch.sigmoid((output)))/2).detach())
        # xc, zx, zy, yc = mi_encoder(x, z, ((torch.sigmoid((output)))))
        x1, x2 = torch.split(x, input.size(0))
        output, _ = torch.split(output, input.size(0))
        xc, zx, zy, yc = mi_encoder(x1, z1, target_var)
        # xc, zx, zy, yc = mi_encoder(x, z, target_var)




        optimizer.zero_grad()

        act_loss = (z1 ** 2).sum(1).mean()

        # loss = total_loss(criterion, output, target_var, xc, zx, zy, yc, measure, alpha, beta)
        predict_loss = criterion(output, target_var)
        zx_nloss, zx_ploss = vector_loss(xc, zx, measure, True)
        # zy_loss = scalar_loss(zy, yc, measure)
        zy_loss = criterion(zy, target_var)
        # zy_loss = vector_loss(z, yc, measure, False)
        # print(zy.shape)
        """ zx_ploss is the disimilarity between z x and should be minimized in mi network """
        loss = predict_loss - alpha * zx_ploss + beta * zy_loss + act_loss * L2 + gamma * consistency_loss
        loss.backward(retain_graph = True)

        """ Step does not need model to be unfreezed, backward does """
        freeze_network(model)
        neg_loss = alpha * zx_nloss
        neg_loss.backward()
        unfreeze_network(model)
        optimizer.step()

        consistency_losses.update(consistency_loss.data, input.size(0))
        losses.update(loss.data + neg_loss.data, input.size(0))
        predict_losses.update(predict_loss.data, input.size(0))
        zx_losses.update(zx_nloss.data - zx_ploss.data, input.size(0))
        zy_losses.update(zy_loss.data, input.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and i != 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PLoss {ploss.val:.4f} ({ploss.avg:.4f})\t'
                  'XLoss {xloss.val:.4f} ({xloss.avg:.4f})\t'
                  'YLoss {yloss.val:.4f} ({yloss.avg:.4f})\t'
                  'Consistency {closs.val:.4f} ({closs.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, ploss = predict_losses, xloss = zx_losses, yloss = zy_losses, closs = consistency_losses))
        if i % log_freq == 0 and i != 0:
            logger.log_value('train_loss', losses.avg, int(epoch*len(train_loader)*bat/16)+int(i*bat/16))
            logger.log_value('train_ploss', predict_losses.avg, int(epoch*len(train_loader)*bat/16)+int(i*bat/16))
            logger.log_value('train_xloss', zx_losses.avg, int(epoch*len(train_loader)*bat/16)+int(i*bat/16))
            logger.log_value('train_yloss', zy_losses.avg, int(epoch*len(train_loader)*bat/16)+int(i*bat/16))
            logger.log_value('consistency_loss', consistency_losses.avg, int(epoch*len(train_loader)*bat/16)+int(i*bat/16))
    return losses.avg


# NOTE to enable bbox evalutation, the result returned by nanmean function might still contains nan
# which needs to be mannually removed to get the true evaluation result (caused by few numbers of annotations)
def validate(val_loader, model, mi_encoder, criterion, epoch, logger, threshold = 0.5, measure = 'JSD'):
    """Perform validation on the validation set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    predict_losses = AverageMeter()
    zx_losses = AverageMeter()
    zy_losses = AverageMeter()
    auc = AverageMeter()
    inv_dic = {0: "Atelectasis", 1:"Cardiomegaly",2:"Effusion", 3:"Infiltration", 4:"Mass", 5:"Nodule",
                6:"Pneumonia", 7:"Pneumothorax", 8:"Consolidation" , 9:"Edema", 10:"Emphysema", 11:"Fibrosis", 12:"Pleural_Thickening", 13:"Hernia"}

    model.eval()
    mi_encoder.eval()
    bat = val_loader.batch_size
    end = time.time()
    y_true = []
    y_pre = []
    with torch.no_grad():
        for i, (input, target, name, bboxes) in enumerate(BackgroundGenerator(val_loader)):
            target = target.float().cuda(non_blocking=True)
            input = input.cuda(non_blocking=True)
            torch.unsqueeze(input,0)
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            x, z, output, m = model(input_var)
            xc, zx, zy, yc = mi_encoder(x, z, target_var)
            # xc, zx, zy, yc = mi_encoder(x, z, (target_var + (torch.sigmoid((output)))/2).detach())
            # xc, zx, zy, yc = mi_encoder(x, z, ((torch.sigmoid((output)))))
            act_loss = (z ** 2).sum(1).mean()
            # loss = total_loss(criterion, output, target_var, xc, zx, zy, yc, measure, alpha, beta)
            predict_loss = criterion(output, target_var)
            zx_loss = vector_loss(xc, zx, measure)
            # zy_loss = scalar_loss(zy, yc, measure)
            zy_loss = criterion(zy, target_var)
            loss = predict_loss + alpha * zx_loss + beta * zy_loss + act_loss * L2

            y_true.extend(target_var.tolist())
            y_pre.extend(output.tolist())

            losses.update(loss.data, input.size(0))
            predict_losses.update(predict_loss, input.size(0))
            zx_losses.update(zx_loss, input.size(0))
            zy_losses.update(zy_loss, input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if i % print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'PLoss {ploss.val:.4f} ({ploss.avg:.4f})\t'
                      'XLoss {xloss.val:.4f} ({xloss.avg:.4f})\t'
                      'YLoss {yloss.val:.4f} ({yloss.avg:.4f})\t'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses,
                          ploss = predict_losses, xloss = zx_losses, yloss = zy_losses))
            if i == len(val_loader) - 1:
                auc.reset()
                y_t = np.asarray(y_true)
                y_p = np.asarray(y_pre)
                val = roc_auc_score(y_t, y_p)
                auc.update(val)
                print("AUC: ", val)
                logger.log_value('avg_auc', auc.avg, int(epoch*len(val_loader)*bat/16)+int(i*bat/16))
                logger.log_value('val_loss', losses.avg,int(epoch*len(val_loader)*bat/16)+int(i*bat/16))
                logger.log_value('val_ploss', predict_losses.avg, int(epoch*len(val_loader)*bat/16)+int(i*bat/16))
                logger.log_value('val_xloss', zx_losses.avg, int(epoch*len(val_loader)*bat/16)+int(i*bat/16))
                logger.log_value('val_yloss', zy_losses.avg, int(epoch*len(val_loader)*bat/16)+int(i*bat/16))
                print("average AUC: ", auc.avg)

    print(' * AUC@1 {auc.avg:.3f}'.format(auc=auc))
    print(' * LOSS@1 {auc.avg:.5f}'.format(auc=losses))
    return auc.avg, losses.avg


def localize(loc_loader, model, mi_encoder, epoch, logger, threshold = 0.5):
    average_iop = AverageMeter()
    average_fpr = AverageMeter()
    average_fnr = AverageMeter()
    model.eval()
    bat = loc_loader.batch_size
    printed = False
    with torch.no_grad():
        for i, (input, name, bboxes) in enumerate(BackgroundGenerator(loc_loader)):
            input = input.cuda(non_blocking=True)
            torch.unsqueeze(input,0)
            input_var = torch.autograd.Variable(input)
            bboxes = bboxes[:, 1:]

            x, z, output, m = model(input_var)
            m = normalize(m)
            # m = 1.0 - m

            if not printed:
                printed = True
                print(m[0])
                np.savetxt("tensor.csv", m[0][0].detach().cpu().numpy(), delimiter=",")
                np.savetxt("tensorz.csv", z[0][0].detach().cpu().numpy(), delimiter=",")
                for i in range(1):
                    torchvision.utils.save_image(normalize(m)[i][0], '/Users/LULU/MILNet/vis/' + str(par_set) + '_' + str(epoch) + '_' + str(name[i]))
                    # torchvision.utils.save_image(m[1][0], '/Users/LULU/MILNet/vis/' + str(par_set) + '_' + str(epoch) + '_' + str(name[1]))

            evals, non_zero_cnt = evaluations(m, threshold, bboxes)
            # print(evals)  #NOTE What if there are no bounding boxes?
            result = np.nanmean(evals, axis=0)

            average_iop.update(result[0], non_zero_cnt)
            average_fpr.update(result[1], non_zero_cnt)
            average_fnr.update(result[2], non_zero_cnt)
            if i == len(loc_loader) - 1:
                print("IOP: ", average_iop.avg)
                print("FPR: ", average_fpr.avg)
                print("FNR: ", average_fnr.avg)
                logger.log_value('IOP', average_iop.avg, epoch)
                logger.log_value('FPR', average_fpr.avg, epoch)
                logger.log_value('FNR', average_fnr.avg, epoch)

    return average_iop.avg, average_fpr.avg, average_fnr.avg
