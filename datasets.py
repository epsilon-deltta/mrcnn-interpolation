import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"]    = boxes
        target["labels"]   = labels
        target["masks"]    = masks
        target["image_id"] = image_id
        target["area"]     = area
        target["iscrowd"]  = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)



    

import torchvision
from engine import train_one_epoch, evaluate
import utils
import transforms as T


def get_transform(train):
    transforms = []
    
    transforms.append(T.ToTensor())
    if train:
        
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



#################################################

import os
import numpy as np
import torch
import torch.utils.data
from PIL import Image
import json
from torchvision.transforms import functional as vtf
from PIL import ImageDraw 
import re

class BalloonDataset(torch.utils.data.Dataset):
    def __init__(self, root='./balloon/balloon/train', annFile='./balloon/balloon/train/via_region_data.json',transforms=None):
        self.root = root
        self.transforms = transforms
        
        with open(annFile,'r') as f:
            self.ans = json.load(f)
        self.indices = sorted(self.ans.keys() )
        postfix = re.compile('.jpg\d+')
        self.files   = [re.sub(postfix,'.jpg',fname) for fname in self.indices]
        
        self.hw = []
        for fname in self.files:
            fpath = os.path.join(root,fname)
            fimg = Image.open(fpath).convert("RGB")
            h,w = fimg.height , fimg.width
            self.hw.append((h,w) )
            del fimg
            
    def __getitem__(self, idx):
        key = self.indices[idx]
        it  = self.ans[key]
        h,w = self.hw[idx]
        
        img_path = os.path.join(self.root,self.files[idx])
        img = Image.open(img_path).convert("RGB")
        num_objs = len(it['regions'])

        boxes = []
        masks = []
        for i in range(num_objs):
            x_ = it['regions'][str(i)]['shape_attributes']['all_points_x']
            y_ = it['regions'][str(i)]['shape_attributes']['all_points_y']
            # boxes
            xmin = np.min(x_)
            xmax = np.max(x_)
            ymin = np.min(y_)
            ymax = np.max(y_)
            boxes.append([xmin, ymin, xmax, ymax])
            
            # masks
            zimg = torch.zeros(h,w)
            pimg = vtf.to_pil_image(zimg)
            dr = ImageDraw.Draw(pimg)
            dr.polygon(list(zip(x_,y_)) ,fill=1)
            nimg = np.array(pimg)
            masks.append(nimg)
            del dr
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks,dtype=torch.uint8)
        area  = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])        
        
        target = {}
        
        
        target["masks"]    = masks
        target["boxes"]    = boxes
        
        target["area"]     = area
        
        target["image_id"] = image_id
        target["labels"]   = labels
        target["iscrowd"]  = iscrowd
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.indices)


def get_dataset(name = 'PennFudan'):
    if name == 'PennFudan' :
        # use our dataset and defined transformations
        dataset      = PennFudanDataset('PennFudanPed', get_transform(train=True))
        dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

        # split the dataset in train and test set
        torch.manual_seed(1)
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])
        dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])
        
    elif name == 'balloon' :        
        train_root = './balloon/balloon/train'
        val_root   = './balloon/balloon/val'
        train_ann  = './balloon/balloon/train/via_region_data.json'
        val_ann    = './balloon/balloon/val/via_region_data.json'
        torch.manual_seed(1)

        dataset      = BalloonDataset(train_root,train_ann, get_transform(train=True))
        dataset_test = BalloonDataset(val_root,val_ann, get_transform(train=False))

    return dataset,dataset_test


