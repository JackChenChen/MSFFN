from os import listdir
from os.path import join
import os

import cv2.cv2
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomResizedCrop, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, RandomCrop, ToTensor, ToPILImage, CenterCrop, Resize, Grayscale

import random
import math
from torch.autograd import Variable
import torch

import torchvision.transforms as transforms

# gray = transforms.Gray()
import numpy as np

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.bmp', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


def calculate_valid_crop_size(crop_size):
    # return crop_size - (crop_size % blocksize)
    return crop_size

def train_hr_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        # CenterCrop(crop_size),
        # RandomHorizontalFlip(p=0.5),
        # RandomVerticalFlip(p=0.5),
        # Grayscale(),
        ToTensor(),
    ])

class TrainDatasetFromFolder_Multi(Dataset):
    def __init__(self, dataset_dir, crop_size):
        super(TrainDatasetFromFolder_Multi, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]  # for x in ['lesion', 'normal']
        # crop_size = calculate_valid_crop_size(crop_size)
        # self.size = (int(crop_size), int(crop_size))
        self.hr_transform = train_hr_transform(crop_size)

    def __getitem__(self, index):
        try:
            normal_image, PRNU_image, label = self.__get_data(index)

            return normal_image, PRNU_image, label-1
        except:
            normal_image, PRNU_image, label = self.__get_data(index + 1)

            return normal_image, PRNU_image, label-1

    def __get_data(self, index):
        normal_image = Image.open(self.image_filenames[index])

        path = self.image_filenames[index]
        ff_device = os.path.split(path)[1].rsplit('_', 3)[0]
        # print(ff_device)
        PRNU_image = np.load("./SelfData/DiffDevice/PRNU_512x512/" + str(ff_device) + ".npy")

        dict = {'D01': 1, 'D02': 2, 'D03': 3, 'D04': 4, 'D05': 5, 'D06': 6, 'D07': 7, 'D08': 8, 'D09': 9, 'D10': 10,
                'D11': 11, 'D12': 12, 'D13': 13, 'D14': 14, 'D15': 15}

        normal_image = self.hr_transform(normal_image)

        return normal_image, PRNU_image, dict[ff_device]

    def __len__(self):
        return len(self.image_filenames)


class TestDatasetNew(Dataset):
    def __init__(self, dataset_dir, crop_size):
        super(TestDatasetNew, self).__init__()
        self.image_filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image_file(x)]
        # crop_size = calculate_valid_crop_size(crop_size)
        self.hr_transform = train_hr_transform(crop_size)

    def __getitem__(self, index):
        try:
            normal_image, label = self.__get_data(index)

            return normal_image, label-1
        except:
            normal_image, label = self.__get_data(index + 1)

            return normal_image, label-1

    def __get_data(self, index):
        normal_image = Image.open(self.image_filenames[index])

        path = self.image_filenames[index]
        ff_device = os.path.split(path)[1].rsplit('_', 3)[0]

        dict = {'D01': 1, 'D02': 2, 'D03': 3, 'D04': 4, 'D05': 5, 'D06': 6, 'D07': 7, 'D08': 8, 'D09': 9, 'D10': 10,
                'D11': 11, 'D12': 12, 'D13': 13, 'D14': 14, 'D15': 15}

        normal_image = self.hr_transform(normal_image)

        return normal_image, dict[ff_device]

    def __len__(self):
        return len(self.image_filenames)


