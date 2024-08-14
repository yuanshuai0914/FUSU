import datasets.transform as tr
import os
import re
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import osgeo.gdal as gdal
import pdb
#import gdal
#import  cv2
class ChangeDetection(Dataset):
    CLASSES = [
  'background',
 'traffic land',
 'inland water',
 'residential land',
 'cropland',
 'agriculture construction',
 'blank',
 'industrial land',
 'park',
 'orchard',
 'public management and service',
 'commercial land',
'public construction',
 'special',
 'forest',
 'storage',
 'wetland',
 'grass'
 ]


    def __init__(self, root, mode, use_pseudo_label=False):
        super(ChangeDetection, self).__init__()
        self.root = root

        self.mode = mode
        self.use_pseudo_label = use_pseudo_label
        if mode == 'train' or 'pseudo_labeling':
            self.root = os.path.join(self.root, 'train')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'val':
            self.root = os.path.join(self.root, 'val')
            self.ids = os.listdir(os.path.join(self.root, "im1"))
            self.ids.sort()
        elif mode == 'test':
            self.root = os.path.join(self.root, 'test')
            self.ids = os.listdir(os.path.join(self.root, 'im1'))
            self.ids.sort()

        self.transform = transforms.Compose([
            tr.RandomFlipOrRotate()
        ])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((123.675, 116.28, 103.53), (58.395, 57.12, 57.375))
        ])

    def __getitem__(self, index):
        id = self.ids[index]
        img1 = Image.open(os.path.join(self.root, 'im1', id))
        img2 = Image.open(os.path.join(self.root, 'im2', id))
        times_inpt=self.read_tif_files(os.path.join(self.root, 'Times'), os.path.splitext(id)[0])

        if self.mode == "test":
            img1 = self.normalize(img1)
            img2 = self.normalize(img2)
            return img1, img2, id

        if self.mode == "val":
            mask1 = Image.open(os.path.join(self.root, 'label1', id))
            mask2 = Image.open(os.path.join(self.root, 'label2', id))
            mask_bin = Image.open(os.path.join(self.root, 'label_bin', id))
        else:
            if self.mode == 'pseudo_labeling' or (self.mode == 'train' and not self.use_pseudo_label):
                mask1 = Image.open(os.path.join(self.root, 'label1', id))
                mask2 = Image.open(os.path.join(self.root, 'label2', id))
                mask_bin = Image.open(os.path.join(self.root, 'label_bin', id))
            else:
                mask1 = Image.open(os.path.join('outdir/masks/train/im1', id))
                mask2 = Image.open(os.path.join('outdir/masks/train/im2', id))
                mask_bin = Image.open(os.path.join(self.root, 'label_bin', id))
            if self.mode == 'train':
                #gt_mask1 = np.array(Image.open(os.path.join(self.root, 'label1', id)))
                #gt_mask2 = np.array(Image.open(os.path.join(self.root, 'label2', id)))
                #mask_bin = np.zeros_like(gt_mask1)
                #mask_bin[gt_mask1 == 0] = 1
                #mask_bin = Image.fromarray(mask_bin)
                mask_bin = Image.open(os.path.join(self.root, 'label_bin', id))
                sample = self.transform({'img1': img1, 'img2': img2, 'mask1': mask1, 'mask2': mask2,
                                         'mask_bin': mask_bin})
                img1, img2, mask1, mask2, mask_bin = sample['img1'], sample['img2'], sample['mask1'], \
                                                     sample['mask2'], sample['mask_bin']

        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        mask1 = torch.from_numpy(np.array(mask1)).long()
        mask2 = torch.from_numpy(np.array(mask2)).long()
        sstime_list = torch.from_numpy(np.array(times_inpt)).float()
        if self.mode == 'train':
            mask_bin = torch.from_numpy(np.array(mask_bin)).float()
            # sstime_list=torch.from_numpy(np.array( times_inpt)).float()
            return img1, img2, mask1, mask2, mask_bin,sstime_list

        return img1, img2, mask1, mask2, id,sstime_list

    def __len__(self):
        return len(self.ids)

    def read_tif_files(self, folder_path, variable_name):
        tif_files = []
        tif_names = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if re.match(r'^{}.*\.tif$'.format(variable_name+"_"), file):
                    file_path = os.path.join(root, file)
                    img = gdal.Open(file_path, gdal.GA_ReadOnly)
                    tif_array = img.ReadAsArray()
                    tif_files.append(tif_array)
                    tif_names.append(file)
                    #if len(tif_files) >= 9:
                    #    return tif_files
        print("variable_name is {}, len is {}".format(variable_name,len(tif_names)))
        return tif_files

