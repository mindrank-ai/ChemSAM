import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate
from utils import random_click
import random
from monai.transforms import LoadImaged, Randomizable,LoadImage
class ISIC2016(Dataset):
    def __init__(self, args, data_path , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):
        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
    def __len__(self):
        return len(self.name_list)
    def __getitem__(self, index):
        inout = 1
        point_label = 1
        name = self.name_list[index]
        mask_name = self.label_list[index]
        img_path = os.path.join(self.data_path, name)
        msk_path = os.path.join(self.data_path, mask_name)
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)
        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            if self.transform_msk:
                mask = self.transform_msk(mask)
        name = name.split('/')[-1].split(".jpg")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
        }
class SegChem(Dataset):
    def __init__(self, args, csv_f='seg_chem.csv' , transform = None, transform_msk = None, mode = 'Training',prompt = 'click', plane = False):
        df = pd.read_csv(csv_f, encoding='gbk',header=None)
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.mode = mode
        self.prompt = prompt
        self.img_size = args.image_size
        self.transform = transform
        self.transform_msk = transform_msk
    def __len__(self):
        return len(self.name_list)
    def get_mean_std(self):
        n_pixels = 0
        pixel_sum = 0
        pixel_squared_sum = 0
        for file_name in self.name_list:
            img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
            n_pixels += img.size
            pixel_sum += np.sum(img)
            pixel_squared_sum += np.sum(img ** 2)
        mean_value = pixel_sum / n_pixels
        std_value = np.sqrt((pixel_squared_sum - (pixel_sum**2 * (1-2*n_pixels)/n_pixels**2) ) /(n_pixels-1))
        return(mean_value,std_value)
    def __getitem__(self, index):
        inout = 1
        point_label = 1
        img_path = self.name_list[index]
        msk_path = self.label_list[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')
        newsize = (self.img_size, self.img_size)
        mask = mask.resize(newsize)
        if self.prompt == 'click':
            pt = random_click(np.array(mask) / 255, point_label, inout)
        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            if self.transform_msk:
                mask = self.transform_msk(mask)
        name = img_path.split('/')[-1].split(".png")[0]
        image_meta_dict = {'filename_or_obj':name}
        return {
            'image':img,
            'label': mask,
            'p_label':point_label,
            'pt':pt,
            'image_meta_dict':image_meta_dict,
            'img_path': img_path
        }