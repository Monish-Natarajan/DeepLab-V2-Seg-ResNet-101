import os
import collections
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset

class VOC_seg(Dataset):
    def __init__(self, data_root, data_mode='val', pseudo_label_dir='/kaggle/input/agv-bana-voc/kaggle/working/data/VOCdevkit/VOC2012/Generation', transforms=None):
        self.train = False
        if data_mode == "train_weak":
            txt_name = "train_aug.txt"
            f_path = os.path.join(data_root, "ImageSets/SegmentationAug", txt_name)
            self.train = True
        if data_mode == "val":
            txt_name = "val.txt"
            f_path = os.path.join(data_root, "ImageSets/Segmentation", txt_name)
        if data_mode == "test":
            txt_name = "test.txt"
            f_path = os.path.join(data_root, "ImageSets/Segmentation", txt_name)
        
        self.filenames = [x.split('\n')[0] for x in open(f_path)]
        self.transforms = transforms
        
        self.annot_folders = ["SegmentationClassAug"]
        if data_mode == "train_weak":
            self.annot_folders += pseudo_label_dir
        if data_mode == "test":
            self.annot_folders = None
        
        self.img_path  = os.path.join(data_root, "JPEGImages", "{}.jpg")
        if self.annot_folders is not None:
            self.mask_paths = [os.path.join(data_root, folder, "{}.png") for folder in self.annot_folders]
        self.len = len(self.filenames)
        print("Number of Files Loaded: ", self.len)
    
    def __len__(self):
        return self.len
    
    def __getitem__(self, index):
        fn  = self.filenames[index]
        img = np.array(Image.open(self.img_path.format(fn)), dtype=np.float32) 
        if self.annot_folders is not None:
            masks = [np.array(Image.open(mp.format(fn)), dtype=np.int64) for mp in self.mask_paths]
        else:
            masks = None
            
        if self.transforms != None:
            img, masks = self.transforms(img, masks)
        
        return img, masks