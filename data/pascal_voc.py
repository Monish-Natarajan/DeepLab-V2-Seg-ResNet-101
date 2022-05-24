import os
import collections
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from .data import transforms as Trs

class VOC_seg(Dataset):
    def __init__(self, data_root, data_mode='train_weak', transforms=None):
        self.train = False
        if data_mode == "train_weak":
            txt_name = "trainaug.txt"
            f_path = os.path.join(data_root, "ImageSets", txt_name)
            self.train = True
        
        self.filenames = [x.split('\n')[0] for x in open(f_path)]
        self.transforms = transforms

        self.img_path  = os.path.join(data_root, "Images", "{}.jpg")
        self.pred_masks_path = os.path.join(data_root,'PredictedMasks','{}.png')
        self.gt_masks_path = os.path.join(data_root,'SegmentationAugmentedMasks','{}.png')
      
        self.len = len(self.filenames)
        print("Number of Files Loaded: ", self.len)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        fn  = self.filenames[index]
        img = np.array(Image.open(self.img_path.format(fn)), dtype=np.float32)
        masks = np.asarray([np.array(Image.open(self.pred_masks_path.format(fn)), dtype=np.float32),
                 np.array(Image.open(self.gt_masks_path.format(fn)), dtype=np.float32)])

        if self.transforms != None:
            img, masks = self.transforms(img, masks)
        
        return img, masks

    if name == '__main__':
        tr_transforms = Compose([
            RandomScale(0.5, 1.5),
            ResizeRandomCrop((321, 321)), 
            RandomHFlip(0.5), 
            ColorJitter(0.5,0.5,0.5,0),
            Normalize_Caffe(),
            ])

    mydataset = VOC_seg(data_root='/content/data/VOC_toy',data_mode='train_weak',transforms=tr_transforms)
    tloader = DataLoader(mydataset, batch_size=2, shuffle=False, num_workers=2)
    for img,mask in tloader:
        print(img.shape)
        print(mask[0].shape)  