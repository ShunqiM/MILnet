import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from lib.utils import BBoxInfo


class CXRDataset(Dataset):

    def __init__(
            self,
            path_to_images,
            fold,
            transform=None,
            sample=0,
            finding="any",
            starter_images=False,
            bbox = False):

        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv("D:\\X\\2019S2\\3912\\CXR8\\nih_labels.csv")
        self.df = self.df[self.df['fold'] == fold]

        if bbox:
            self.bbdf = pd.read_csv("D:\\X\\2019S2\\3912\\CXR8\\BBox_List_2017.csv")
        else:
            self.bbdf = None

        if(starter_images):
            starter_images = pd.read_csv("starter_images.csv")
            self.df=pd.merge(left=self.df,right=starter_images, how="inner",on="Image Index")

        # can limit to sample, useful for testing
        # if fold == "train" or fold =="val": sample=500
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    self.df = self.df[self.df[finding] == 1]
                else:
                    print("No positive cases exist for "+LABEL+", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")

        self.df = self.df.set_index("Image Index")
        self.PRED_LABEL = [
            'Atelectasis',
            'Cardiomegaly',
            'Effusion',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pneumonia',
            'Pneumothorax',
            'Consolidation',
            'Edema',
            'Emphysema',
            'Fibrosis',
            'Pleural_Thickening',
            'Hernia']
        RESULT_PATH = "results/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
             # can leave zero if zero, else make one
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                label[i] = self.df[self.PRED_LABEL[i].strip()
                                   ].iloc[idx].astype('int')

        if self.transform:
            image = self.transform(image)

        index = self.df.index[idx]
        all_bboxes = torch.full((14, 4), 0)
        if self.bbdf is not None:
            bboxes = self.bbdf.loc[self.bbdf['Image Index'] == index]
            if bboxes.empty:
                return (image, label, index, all_bboxes)
            # # print(bboxes.shape)
            for i in range(0, len(self.PRED_LABEL)):
                bb_single = bboxes.loc[bboxes['Finding Label'] == self.PRED_LABEL[i].strip()]
                if not bb_single.empty:
                    # Using dictionary would cause a multithreading error???
                    all_bboxes[i] = torch.tensor([bb_single['Bbox [x'].iloc[0],
                                            bb_single['y'].iloc[0],
                                            bb_single['w'].iloc[0],
                                            bb_single['h]'].iloc[0]]).long()

        return (image, label, index, all_bboxes)
        # return (image, label,self.df.index[idx])

class PneumoniaData(Dataset):
    def __init__(
            self,
            path_to_images,
            fold,
            transform=None,
            sample=0,
            finding="any",
            starter_images=False,
            bbox = False):

        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv("D:\\X\\2019S2\\3912\\CXR8\\nih_labels.csv")
        self.df = self.df[self.df['fold'] == fold]

        if bbox:
            self.bbdf = pd.read_csv("D:\\X\\2019S2\\3912\\CXR8\\BBox_List_2017.csv")
        else:
            self.bbdf = None

        if(starter_images):
            starter_images = pd.read_csv("starter_images.csv")
            self.df=pd.merge(left=self.df,right=starter_images, how="inner",on="Image Index")

        # can limit to sample, useful for testing
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(sample)

        if not finding == "any":  # can filter for positive findings of the kind described; useful for evaluation
            if finding in self.df.columns:
                if len(self.df[self.df[finding] == 1]) > 0:
                    self.df = self.df[self.df[finding] == 1]
                else:
                    print("No positive cases exist for "+LABEL+", returning all unfiltered cases")
            else:
                print("cannot filter on finding " + finding +
                      " as not in data - please check spelling")

        self.df = self.df.set_index("Image Index")
        RESULT_PATH = "results/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image = Image.open(
            os.path.join(
                self.path_to_images,
                self.df.index[idx]))
        image = image.convert('RGB')

        label = 0
        if(self.df['Pneumonia'].iloc[idx].astype('int') > 0):
            label = self.df['Pneumonia'].iloc[idx].astype('int')
        label = np.reshape(label, [1])

        if self.transform:
            image = self.transform(image)

        if self.bbdf is not None:
            bboxes = self.bbdf.loc[self.bbdf['Image Index'] == self.df.index[idx]]
            bboxes = bboxes.loc[bboxes['Finding Label'] == 'Pneumonia']
            if bboxes.empty:
                return (image, label, self.df.index[idx], torch.tensor([0, 0, 0, 0]))
            else:
                return (image, label, self.df.index[idx], torch.tensor([bboxes['Bbox [x'].iloc[0],
                                                            bboxes['y'].iloc[0],
                                                            bboxes['w'].iloc[0],
                                                            bboxes['h]'].iloc[0]]).long())

        return (image, label, self.df.index[idx], torch.tensor([0, 0, 0, 0])) # dictionary of the bbox of disease


class BBoxDataset(Dataset):
    def __init__(self, path_to_images, transform, single_disease = True):
        self.bbdf = pd.read_csv("D:\\X\\2019S2\\3912\\CXR8\\BBox_List_2017.csv")
        self.transform = transform
        self.path_to_images = path_to_images
        # NOTE that the disease name has changed Infiltration -> Infiltrate
        if single_disease:
            self.bbdf = self.bbdf.loc[self.bbdf['Finding Label'] == 'Infiltrate']
        self.dic = {"Atelectasis":0, "Cardiomegaly":1,"Effusion":2, "Infiltrate":3, "Mass":4, "Nodule":5,
                    "Pneumonia":6, "Pneumothorax":7, "Consolidation":8 , "Edema":9, "Emphysema":10, "Fibrosis":11, "Pleural_Thickening":12, "Hernia":13}

    def __len__(self):
        return len(self.bbdf)

    def __getitem__(self, idx):
        img_name = self.bbdf['Image Index'].iloc[idx]
        image = Image.open(
            os.path.join(
                self.path_to_images,
                img_name))
        image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        label = torch.tensor([self.dic[self.bbdf['Finding Label'].iloc[idx]],
                        self.bbdf['Bbox [x'].iloc[idx],
                        self.bbdf['y'].iloc[idx],
                        self.bbdf['w'].iloc[idx],
                        self.bbdf['h]'].iloc[idx]]).long()

        return (image, img_name, label)
