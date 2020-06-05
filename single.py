import sys
import os
sys.path.insert(0, os.getcwd())

from torchvision import models
import signal
import numpy as np
import pandas as pd
from lib.cxr_dataset import *
import time
from lib.utils import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lib.MILNet import *
from train_single import *


def run():
    best_auc = 0
    best_loss = 10

    global bat
    global lr
    logger = Logger(logdir="/Users/LULU/MILNet/logger/" + par_set, flush_secs=6)


    np.random.seed(0)
    torch.manual_seed(3)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    csv_path = "D:\\X\\2019S2\\3912\\CXR8\\Data_Entry_2017.csv"
    root_dir = "D:\\X\\2019S2\\3912\\CXR8\\images\\images"
    train_path = "D:\\X\\2019S2\\3912\\CXR8\\train.csv"
    tun_path = "D:\\X\\2019S2\\3912\\CXR8\\tuning.csv"
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    PATH_TO_IMAGES = "D:\\X\\2019S2\\3912\\CXR8\\images\\images"
    PATH_TO_IMAGES = "C:\\Users\\LULU\\3912\\CXR8\\images\\images"

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    df = pd.read_csv("D:\\X\\2019S2\\3912\\CXR8\\nih_labels.csv", index_col=0)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
        'val': transforms.Compose([
            transforms.Resize((size, size)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    }
    transformed_datasets = {}
    transformed_datasets['train'] = PneumoniaData(
        path_to_images=PATH_TO_IMAGES,
        fold='train',
        transform=data_transforms['train'],
        bbox = False)
    transformed_datasets['val'] = PneumoniaData(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'],
        bbox = True)
    transformed_datasets['loc'] = BBoxDataset(
        path_to_images=PATH_TO_IMAGES,
        transform=data_transforms['val'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=bat,
        shuffle=True,
        num_workers=8) # best number
    dataloaders['val'] = torch.utils.data.DataLoader(
        transformed_datasets['val'],
        batch_size=bat,
        shuffle=True,
        num_workers=8)
    dataloaders['loc'] = torch.utils.data.DataLoader(
        transformed_datasets['loc'],
        batch_size=bat,
        shuffle=True,
        num_workers=8)

    fe = get_feature_extractor()
    classifier = models.resnet50(pretrained=True)
    classifier.fc = nn.Linear(fe.get_channel_num(), 1)
    model = MILNet(fe, classifier, t = network_threshold, zt = zt)
    mi_encoder = MIEncoder(7, 7, model.fe.get_local_channel_num(), mi_units, Lambda, Compress)

    # NOTE The main reason we need two optimizer is that they need different learning rate
    # optimizer = torch.optim.SGD(
    #     filter(
    #         lambda p: p.requires_grad,
    #         (list(model.parameters()))),
    #     lr=lr,
    #     momentum=0.9,
    #     weight_decay=1e-4)
    #
    # mi_opti = torch.optim.SGD(
    #     filter(
    #         lambda p: p.requires_grad,
    #         (list(mi_encoder.parameters()))),
    #     lr=mi_lr,
    #     momentum=0.9,
    #     weight_decay=1e-4)
    mi_opt = None

    optimizer = torch.optim.SGD(
        filter(
            lambda p: p.requires_grad,
            (list(model.parameters()) + list(mi_encoder.parameters()))),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4)


    criterion = nn.BCEWithLogitsLoss().cuda()
    scheduler = ReduceLROnPlateau(optimizer, factor =  0.1, patience = 2)
    start_epoch = 0

    if load_model:
        model, mi_encoder, optimizer, start_epoch, best_auc, scheduler = load_checkpoint(
                        model, mi_encoder, optimizer, scheduler, None, "D:\\X\\2019S2\\3912\\MILN_models\\c47_epoch0")
        # adjust_learning_rate_(optimizer, start_epoch, logger, par_set)
        model = model.cuda()

    for epoch in range(start_epoch, 100):
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            logger.log_value('learning_rate', lr, epoch)

        ep = epoch
        training(dataloaders['train'], model, mi_encoder, criterion, optimizer, mi_opt, epoch, logger, alpha, beta)

        # evaluate on validation set
        new_auc, new_loss = validate(dataloaders['val'], model, mi_encoder, criterion, epoch, logger, THRESHOLD)
        iop, fpr, fnr = localize(dataloaders['loc'], model, mi_encoder, epoch, logger, THRESHOLD)
        # exit()
        scheduler.step(new_loss)
        mi_encoder.update_GRL(0.05)
        best_auc = max(new_auc, best_auc)
        # remember best prec@1 and save checkpoint
        is_best = new_loss < best_loss
        best_loss = min(new_loss, best_loss)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'mi_dict': mi_encoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_auc': best_auc,
            'scheduler': scheduler.state_dict()
        }, is_best, str(par_set)+"_epoch"+str(epoch))
        # exit()
    print('Best accuracy: ', best_auc)



if __name__ == '__main__':
    run()
