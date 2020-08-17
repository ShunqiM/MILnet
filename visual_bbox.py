import pandas as pd
import torch
import torchvision
img_name = '00020113_030.png'
df = pd.read_csv("D:\\X\\2019S2\\3912\\CXR8\\BBox_List_2017.csv")
box = df.loc[df['Image Index'] == img_name].values[0]
box = box[2:6].astype(int)
print(box.shape)
print(box)
# print(box)
# x = box[0]
# y = box[1]
# w = box[2]
# h = box[3]
boxed = torch.zeros(1024, 1024)
boxed[box[1] : (box[1] + box[3]), box[0] : (box[0] + box[2])] = 1
torchvision.utils.save_image(boxed, '/Users/LULU/MILNet/vis/' + img_name)
