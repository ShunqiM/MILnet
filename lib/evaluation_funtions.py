import torch
import torch.nn as nn
import numpy as np


def resize(bbox, img_size):
    ratio = float(img_size) / 1024.
    bbox = (bbox.float() * ratio).int()
    return bbox


def generate_bbox(map, t):
    return

def is_box(bbox):
    if bbox[0] == 0:
        return False
    return True

def evaluations(heatmap, t, bbox):
    """ heatmap: tensors of heatmaps
        t: threshold to process the heatmap data
        bbox: tensors containing BBox data: Batch, Disease, (x, y, w, h)
        size: input image length """
    bbox = bbox.int()
    binary = (heatmap > t).int()
    img_size = heatmap.size()[-1]
    bbox = resize(bbox, img_size)
    if (len(bbox.shape) == 3): # batch, disease, data
        non_zero_cnt = np.zeros((14))
        b = heatmap.shape[0]
        ret = np.full((b, 14, 3), np.nan)
        for i in range(b):
            for j in range(14):
                if is_box(bbox[i][j]):
                    non_zero_cnt[j] += 1
                    ret[i][j] = iop(bbox[i][j], binary[i][j], img_size)
    if (len(bbox.shape) == 2): # batch, data (single disease)
        b = heatmap.shape[0]
        ret = np.full((b, 3), np.nan)
        non_zero_cnt = 0
        for i in range(b):
            if is_box(bbox[i]):
                non_zero_cnt += 1
                ret[i] = eval(bbox[i], binary[i], img_size)
        return ret, non_zero_cnt

""" Returns the IOP, FPR, FNR """
def eval(bbox, heatmap, img_size):
    boxed = torch.zeros((img_size, img_size)).to(heatmap.device).int()
    boxed[bbox[1] : (bbox[1] + bbox[3]), bbox[0] : (bbox[0] + bbox[2])] = 1
    box_area = bbox[3] * bbox[2]
    intersect = torch.sum(boxed * heatmap).float()
    if torch.sum(heatmap) == 0:
        iop = 0.
    else:
        iop = (intersect / torch.sum(heatmap)).item()
    fp = torch.sum(((heatmap - boxed) > 0).int()).float()
    fpr = (fp / (img_size * img_size - box_area)).item()
    fn = torch.sum(((boxed - heatmap) > 0).int()).float()
    fnr = (fn / box_area).item()
    return np.array([iop, fpr, fnr])


def iops(heatmap, t, bbox):
    """ heatmap: tensors of heatmaps
        t: threshold to process the heatmap data
        bbox: tensors containing BBox data: Batch, Disease, (x, y, w, h)
        size: input image length """
    bbox = bbox.int()
    binary = (heatmap > t).int()
    img_size = heatmap.size()[-1]
    bbox = resize(bbox, img_size)
    if (len(bbox.shape) == 3): # batch, disease, data
        b = heatmap.shape[0]
        ret = np.zeros((b, 14))
        for i in range(b):
            for j in range(14):
                ret[i][j] = iop(bbox[i][j], binary[i][j], img_size)
    if (len(bbox.shape) == 2): # batch, data (single disease)
        b = heatmap.shape[0]
        ret = np.zeros((b))
        for i in range(b):
            ret[i] = iop(bbox[i], binary[i], img_size)
        return ret

""" Assume the Input are of Right Shape """
def iop(bbox, heatmap, img_size):
    boxed = torch.zeros((img_size, img_size)).to(heatmap.device).int()
    boxed[bbox[1] : (bbox[1] + bbox[3]), bbox[0] : (bbox[0] + bbox[2])] = 1
    intersect = torch.sum(boxed * heatmap).float()
    if intersect == 0:
        return 0
    return (intersect / torch.sum(heatmap)).item()
