import os
import time

import torch
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

alpha = 0.3
beta = 3
THRESHOLD = 0.7
network_threshold = 0.2
mi_units = 256
load_model = False
size = 224
lr = 0.01
bat = 16
validate_log_freq = 1600/bat
log_freq = 16000/bat
print_freq = 2
par_set = "non_m"

def training(train_loader, model, mi_encoder, criterion, optimizer, epoch, logger, alpha, beta, measure = 'JSD'):
    """Train for one epoch on the training set"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    predict_losses = AverageMeter()
    zx_losses = AverageMeter()
    zy_losses = AverageMeter()
    model.train()
    end = time.time()
    bat = train_loader.batch_size
    for i, (input, target, name, bboxes) in enumerate(BackgroundGenerator(train_loader)):
        target_var = target.float().cuda(non_blocking=True)
        input_var = input.cuda(non_blocking=True)

        x, z, output, m = model(input_var)
        xc, zx, zy, yc = mi_encoder(x, z, target_var)

        optimizer.zero_grad()

        # loss = total_loss(criterion, output, target_var, xc, zx, zy, yc, measure, alpha, beta)
        predict_loss = criterion(output, target_var)
        zx_loss = vector_loss(xc, zx, measure)
        zy_loss = scalar_loss(zy, yc, measure)
        loss = predict_loss - alpha * zx_loss + beta * zy_loss


        losses.update(loss.data, input.size(0))
        predict_losses.update(predict_loss, input.size(0))
        zx_losses.update(zx_loss, input.size(0))
        zy_losses.update(zy_loss, input.size(0))


        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0 and i != 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PLoss {ploss.val:.4f} ({ploss.avg:.4f})\t'
                  'XLoss {xloss.val:.4f} ({xloss.avg:.4f})\t'
                  'YLoss {yloss.val:.4f} ({yloss.avg:.4f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      loss=losses, ploss = predict_losses, xloss = zx_losses, yloss = zy_losses))
        if i % log_freq == 0 and i != 0:
            logger.log_value('train_loss', losses.avg, int(epoch*len(train_loader)*bat/16)+int(i*bat/16))
            logger.log_value('train_ploss', predict_losses.avg, int(epoch*len(train_loader)*bat/16)+int(i*bat/16))
            logger.log_value('train_xloss', zx_losses.avg, int(epoch*len(train_loader)*bat/16)+int(i*bat/16))
            logger.log_value('train_yloss', zy_losses.avg, int(epoch*len(train_loader)*bat/16)+int(i*bat/16))
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

            # loss = total_loss(criterion, output, target_var, xc, zx, zy, yc, measure, alpha, beta)
            predict_loss = criterion(output, target_var)
            zx_loss = vector_loss(xc, zx, measure)
            zy_loss = scalar_loss(zy, yc, measure)
            loss = predict_loss - alpha * zx_loss + beta * zy_loss

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
    with torch.no_grad():
        for i, (input, name, bboxes) in enumerate(BackgroundGenerator(loc_loader)):
            input = input.cuda(non_blocking=True)
            torch.unsqueeze(input,0)
            input_var = torch.autograd.Variable(input)
            bboxes = bboxes[:, 1:]

            x, z, output, m = model(input_var)

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
