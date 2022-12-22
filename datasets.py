import glob
import json
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from osgeo import gdal
from torch.utils.data import Dataset


class EurosatDataset(Dataset):
    def __init__(
        self, data_manifest_path, bands = [1, 2, 3, 7]
    ):
        with open(data_manifest_path) as f:
            data_dict = json.load(f)

        dir_path = data_dict["dir_path"]
        filepaths = glob.glob(dir_path + "/**/*.tif", recursive=True)
        self.filepaths = filepaths
        self.categories = data_dict["categories"]
        self.bands = bands


    def __len__(self):
        return len(self.filepaths)


    @staticmethod
    def get_category_from_filepath(filepath):
        return filepath.replace("\\", "/").split("/")[-2]


    def load(self, filepath):
        ext = filepath.split(".")[-1]
        if ext in ['tif', 'tiff']:
            try:
                image = gdal.Open(str(filepath)).ReadAsArray().astype(np.int16)
            except AttributeError as e:
                print(f"Problem loading {filepath}.")
                raise e
            return image[self.bands]
        else:
            raise NotImplementedError(f"Expects .tif or .tiff files. Received .{ext}.")


    @staticmethod
    def preprocess(image, max_pix_value = 10000):
        return image / max_pix_value


    @staticmethod
    def horizontal_flip(image, p = 0.75):
        if random.random() > p:
            image = TF.hflip(image)
        
        return image


    @staticmethod
    def vertical_flip(image, p = 0.75):
        if random.random() > p:
            image = TF.vflip(image)
        
        return image


    @staticmethod
    def rotate(image, p = 0.75, max_angle = 30):
        if random.random() > p:
            angle = random.randint(- max_angle, max_angle)
            image = TF.rotate(image, angle)
        
        return image


    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        assert os.path.exists(filepath), f"File {filepath} does not exist."

        category = self.get_category_from_filepath(filepath)
        target = self.categories[category]

        image = self.load(filepath)
        image = self.preprocess(image)   

        image = torch.as_tensor(image.copy()).float().contiguous()
        target = torch.as_tensor(target)

        transform_idx = random.randint(0, 2)
        if transform_idx == 0:
            image = self.horizontal_flip(image)
        elif transform_idx == 1:
            image = self.vertical_flip(image)
        else:
            image = self.rotate(image)           

        return {
            'X': image,
            'Y': target
        }
