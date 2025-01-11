# -*- coding: utf-8 -*-
"""
Created on Sun Feb  7 22:04:28 2021

@author: DELL
"""
####
import torch.utils.data as D
from torchvision import transforms as T
from PIL import ImageFilter, Image, ImageOps
import torchvision.transforms.functional as TF
from tqdm import tqdm
from osgeo import gdal
from sklearn.metrics import f1_score
import random
import numpy as np
import torch
import cv2

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# rgb_mean = (0.4353, 0.4452, 0.4131)
# rgb_std = (0.2044, 0.1924, 0.2013)
rgb_mean = (0.4353, 0.4452, 0.4131)
rgb_std = (0.2044, 0.1924, 0.2013)
# 线性拉伸
def truncated_linear_stretch(image, truncated_value, max_out = 255, min_out = 0):
    def gray_process(gray):
        truncated_down = np.percentile(gray, truncated_value)
        truncated_up = np.percentile(gray, 100 - truncated_value)
        gray = (gray - truncated_down) / (truncated_up - truncated_down) * (max_out - min_out) + min_out 
        gray = np.clip(gray, min_out, max_out)
        gray = np.uint8(gray)
        return gray
    
    image_stretch = []
    for i in range(image.shape[2]):
        # 只拉伸RGB
        if(i<3):
            gray = gray_process(image[:,:,i])
        else:
            gray = image[:,:,i]
        image_stretch.append(gray)
    image_stretch = np.array(image_stretch)
    image_stretch = image_stretch.swapaxes(1, 0).swapaxes(1, 2)
    return image_stretch


def RandomScaleCrop(image, lable):
    base_size = 512
    crop_size = 256
    short_size = random.randint(int(base_size * 0.5), int(base_size * 2.0))
    # print(image.shape)
    w= image.shape[0]
    h = image.shape[1]
    if h > w:
        ow = short_size
        oh = int(1.0 * h * ow / w)
    else:
        oh = short_size
        ow = int(1.0 * w * oh / h)
    # print('ow:', ow)
    # print('oh:', oh)
    # print(image.type)
    img = cv2.resize(image, dsize=(ow, oh), interpolation=cv2.INTER_NEAREST)
    mask = cv2.resize(lable, dsize=(ow, oh), interpolation=cv2.INTER_NEAREST)
    # img = image.resize((ow, oh), Image.BILINEAR)
    # mask = lable.resize((ow, oh), Image.NEAREST)
    # pad crop
    if short_size < crop_size:
        padh = crop_size - oh if oh < crop_size else 0
        padw = crop_size - ow if ow < crop_size else 0
        img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=0)
    # random crop crop_size
    # w, h = img.size
    # x1 = random.randint(0, w - crop_size)
    # y1 = random.randint(0, h - crop_size)
    # img = img.crop((x1, y1, x1 + crop_size, y1 + crop_size))
    # mask = mask.crop((x1, y1, x1 + crop_size, y1 + crop_size))

    return img, mask



def DataAugmentation(image, label, mode):
    if(mode == "train"):
        image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        hor = random.choice([True, False])
        if(hor):

            image = np.flip(image, axis = 1)
            label = np.flip(label, axis = 1)
        ver = random.choice([True, False])
        if(ver):

            image = np.flip(image, axis = 0)
            label = np.flip(label, axis = 0)
        crop = random.choice([True, False])
        if(crop):
            RandomScaleCrop(image=image, lable=label)
        gau = random.choice([True, False])
        if(gau):
            image = cv2.GaussianBlur(image, (9, 9), 0)
        # stretch = random.choice([True, False])
        # if(stretch):
        #     image = truncated_linear_stretch(image, 0.5)

    if(mode == "val"):
        image = cv2.resize(image, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, dsize=(512, 512), interpolation=cv2.INTER_NEAREST)
        # stretch = random.choice([0.8, 1, 2])
    # if(stretch == 'yes'):

        # image = truncated_linear_stretch(image, stretch)
    return image, label


@torch.no_grad()

def cal_val_iou(model, loader):
    val_iou = []
    val_acc = []
    val_f1 = []

    model.eval()
    val_data_loader_num = iter(loader)
    for image, target in tqdm(val_data_loader_num):
        # print(target)

        image, target = image.to(DEVICE), target.to(DEVICE)
        output = model(image)
        # output = output.argmax(1)
        output = output.squeeze(1)
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        # print(output)
        # output = output.type(torch.LongTensor)
        # target = target.type(torch.LongTensor)
        # print('aa:', output.size())
        # print('bb:', target.size())
        iou = cal_binary_iou(output, target)
        val_iou.append(iou)

        f1 = cal_binary_f1(output, target)
        val_f1.append(f1)
        
        acc = Accuracy(output, target)
        val_acc.append(acc)
    return val_iou, val_acc, val_f1

def cal_binary_f1(pred, mask):
    f1_result = []
    p = (mask == 1).int().reshape(-1)
    t = (pred == 1).int().reshape(-1)
    f1 = f1_score(t.cpu().numpy(), p.cpu().numpy())
    f1_result.append(f1)
    return np.stack(f1_result)


def cal_iou(pred, mask, c=1):
    iou_result = []
    for idx in range(c):
        p = (mask == idx).int().reshape(-1)
        t = (pred == idx).int().reshape(-1)
        uion = p.sum() + t.sum()
        # print('p:', p)
        # print('t:', t)
        # print('uion:', uion)
        overlap = (p*t).sum()
        print('over:',overlap)

        iou = 2*overlap/(uion + 0.0001)
        iou_result.append(iou.abs().data.cpu().numpy())
        print('iou:',iou_result)
    return np.stack(iou_result)

def cal_binary_iou(pred, mask, c=1):
    iou_result = []
    p = (mask == 1).int().reshape(-1)
    t = (pred == 1).int().reshape(-1)
    uion = p.sum() + t.sum()
    # print('p:', pred)
    # print('t:', mask)
    # print('uion:', uion)
    overlap = (p * t).sum()
    # print('over:', overlap)

    iou = 2 * overlap / (uion + 0.0001)
    iou_result.append(iou.abs().data.cpu().numpy())
    return np.stack(iou_result)

def Accuracy(pred, mask):
    acc_result = []
    p = pred.int().reshape(-1)
    t = mask.int().reshape(-1)
    valid = (t > 0)
    acc_sum =torch.sum(valid*(p==t).long()).float()
    pixel_sum = torch.sum(valid).float()
    acc = acc_sum / (pixel_sum + 1e-10)
    acc_result.append(acc.abs().data.cpu().numpy())
    return np.stack(acc_result)

class OurDataset(D.Dataset):
    def __init__(self, image_paths, label_paths, mode):
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.mode = mode
        self.len = len(image_paths)
        self.as_tensor = T.Compose([

            T.ToTensor(),
        ])
    # 获取数据操作
    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        label = cv2.imread(self.label_paths[index], cv2.IMREAD_GRAYSCALE)

        if self.mode == "train":
            # print(self.label_paths[index])
            image, label = DataAugmentation(image, label, self.mode)

            image = np.array(image, np.float32) / 255.0
            image_array = np.ascontiguousarray(image)
            image = self.as_tensor(image_array)
            image = TF.normalize(image, mean=rgb_mean, std=rgb_std)
            label = np.array(label, np.float32) / 255.0
            label[label >= 0.5] = 1
            label[label <= 0.5] = 0
            # print(label)
            # print(label.shape)
            return image, self.as_tensor(label)
        elif self.mode == "val":

            image, label = DataAugmentation(image, label, self.mode)
            image = np.array(image, np.float32) / 255.0
            image_array = np.ascontiguousarray(image)
            image = self.as_tensor(image_array)
            image = TF.normalize(image, mean=rgb_mean, std=rgb_std)
            label = np.array(label, np.float32) / 255.0
            label[label >= 0.5] = 1
            label[label <= 0.5] = 0
            # print(label)
            # image = torch.from_numpy(image_array)
            # label = torch.from_numpy(label)
            return image, self.as_tensor(label)
        elif self.mode == "test":   
            image_stretch = truncated_linear_stretch(image, 0.5)
            return self.as_tensor(image), self.as_tensor(image_stretch), self.image_paths[index]

    def __len__(self):
        return self.len


def get_dataloader(image_paths, label_paths, mode, batch_size,
                   shuffle, num_workers):
    dataset = OurDataset(image_paths, label_paths, mode) #Dataset是一个可迭代对象
    dataloader = D.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                              num_workers=num_workers, pin_memory=True)
    return dataloader


def split_train_val(image_paths, label_paths, val_index=0):

    train_image_paths, train_label_paths, val_image_paths, val_label_paths = [], [], [], []
    for i in range(len(image_paths)):

        if i % 4 == val_index:
            val_image_paths.append(image_paths[i])
            val_label_paths.append(label_paths[i])
        else:
            train_image_paths.append(image_paths[i])
            train_label_paths.append(label_paths[i])
    print("Number of training images: ", len(train_image_paths))
    print("Number of val images: ", len(val_image_paths))
    print("val:", val_image_paths)
    
    return train_image_paths, train_label_paths, val_image_paths, val_label_paths

