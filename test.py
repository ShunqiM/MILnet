import sys
import os
sys.path.insert(0, os.getcwd())

from torchvision import models
from lib.cxr_dataset import *
import signal
import numpy as np
import pandas as pd
from lib.cxr_dataset import *
import time

from prefetch_generator import BackgroundGenerator
from lib.utils import *
from lib.evaluation_funtions import *

print_freq = 2
load_model = False
size = 224
block_configure = (6,12,24,16)
lr = 0.01
grow_rate = 32
bat = 16
validate_log_freq = 1600/bat
log_freq = 16000/bat
par_set = "pretrained_19"
bbox = True

# TODO github test

def run():

    # box = torch.zeros([2, 4])
    # box[1] = box[0] = torch.tensor([1,1,200,200])
    # maps = torch.FloatTensor(2, 224, 224).uniform_(0, 1)
    # ret = evaluations(maps, 0.1, box)
    # print(ret)
    #
    # exit()


    # logger = Logger(logdir="/Users/LULU/3912/tb/" + par_set, flush_secs=6)
    best_auc = 0

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
        bbox = bbox)
    transformed_datasets['val'] = PneumoniaData(
        path_to_images=PATH_TO_IMAGES,
        fold='val',
        transform=data_transforms['val'],
        bbox = bbox)

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


    input, target, _, bbox_data = transformed_datasets['train'].__getitem__(1964)
    print(_, bbox_data)

    end = time.time()
    for i, (input, target, _, bbox_data) in enumerate(BackgroundGenerator(dataloaders['train'])):
        # print(input.shape, target, _, bbox_data)
        print(time.time() - end)
        end = time.time()
        print(target.shape)
        print((bbox_data.shape))
        # print(type(bbox_data[0]))
        # print((bbox_data))
        if (i == 2):
            exit()
        # if len(bbox_data) > 0:
        #     break


# if __name__ == '__main__':
#     run()

bbox = torch.tensor([0, 0, 10, 10])
heatmap = torch.full((20, 20), 1).int()
heatmap[1,1] = 0
print(eval(bbox, heatmap, 20))
