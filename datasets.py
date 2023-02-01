__author__ = "Richard Correro (richard@richardcorrero.com)"


import argparse
import glob
import json
import logging
import os
import random
from typing import Generator, List, Optional

import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchvision.transforms as T
from osgeo import gdal
from PIL import Image
from torch.utils.data import Dataset

from script_utils import arg_is_true, parse_args


class EurosatDataset(Dataset):
    __name__ = "EurosatDataset"

    DEFAULT_DATA_MANIFEST: str = "eurosat_manifest.json"
    DEFAULT_BANDS: List[int] = [1, 2, 3, 7]
    DEFAULT_USE_DATA_AUG: bool = True


    def __init__(self):
        args = self.parse_args()
        data_manifest_path: str = args["data_manifest"]
        bands: List[str] = args["bands"]
        with open(data_manifest_path) as f:
            data_dict = json.load(f)
        self.args = args

        dir_path = data_dict["dir_path"]
        filepaths = glob.glob(dir_path + "/**/*.tif", recursive=True)
        self.filepaths = filepaths
        self.categories = data_dict["categories"]
        self.bands = bands
        self.use_data_aug = arg_is_true(args["use_data_aug"])


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data-manifest",
            default=self.DEFAULT_DATA_MANIFEST
        )  
        parser.add_argument(
            "--bands",
            nargs="+",
            type=int,
            default=self.DEFAULT_BANDS
        )
        parser.add_argument(
            "--use-data-aug",
            default=self.DEFAULT_USE_DATA_AUG
        )
        args = parse_args(parser=parser)
        return args        


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
            raise NotImplementedError(
                f"Expects .tif or .tiff files. Received .{ext}."
            )


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
        
        if self.use_data_aug:
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


class ConvLSTMCDataset(Dataset):
    __name__ = "ConvLSTMCDataset"

    DEFAULT_DATA_MANIFEST: str = "sits_manifest.json"
    DEFAULT_USE_DATA_AUG: bool = True
    DEFAULT_VERBOSE: bool = False


    def __init__(self):
        args = self.parse_args()
        data_manifest_path = args["data_manifest"]
        with open(data_manifest_path) as f:
            data_dict = json.load(f)
        self.args = args            

        dir_path = data_dict["dir_path"]
        verbose: bool = arg_is_true(args["verbose"])
        if verbose:
            print(f"Loading samples from {dir_path}")
            num_pos: int = 0
            num_neg: int = 0
        samples = list()
        for dirpath, dirnames, filenames in os.walk(dir_path):
            if not dirnames:
                for key, value in data_dict["categories"].items():
                    if key in dirpath:
                        if verbose and value:
                            num_pos += 1
                        elif verbose:
                            num_neg += 1
                        sample_dict = {
                            "dirpath": dirpath,
                            "filenames": filenames,
                            "label": value
                        }
                        if verbose:
                            print(
                                f"""
                                   Sample:
                                Directory: {dirpath}
                                Label: {value}
                                """
                            )
                        samples.append(sample_dict)
        if verbose:
            print(
                f"""
                Done loading samples.
                Number of positive samples: {num_pos}
                Number of negative samples: {num_neg}
                """
            )

        self.transforms = T.Compose([
            T.Resize((224,224)),
            # T.CenterCrop((224,224)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])                

        self.samples = samples
        self.categories = data_dict["categories"]
        self.use_data_aug = arg_is_true(args["use_data_aug"])
        self.verbose = verbose


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--data-manifest",
            default=self.DEFAULT_DATA_MANIFEST
        )
        parser.add_argument(
            "--use-data-aug",
            default=self.DEFAULT_USE_DATA_AUG
        )
        parser.add_argument(
            "--verbose",
            default=self.DEFAULT_VERBOSE
        )
        args = parse_args(parser=parser)
        return args              


    def __len__(self):
        return len(self.samples)    


    def read_png(self, filepath: str) -> np.ndarray:
        img = Image.open(filepath).convert('RGB')
        # arr = np.array(img)
        return img


    def read_png_as_arr(self, filepath: str) -> np.ndarray:
        img = Image.open(filepath).convert('RGB')
        # img = self.transforms(img)
        arr = np.array(img)
        return arr      


    @staticmethod
    def sort_filenames(filenames: List[str]) -> List[str]:
        filenames_sorted = sorted(filenames)
        return filenames_sorted


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


    def spatial_transform(self, image: torch.tensor, idx: int) -> torch.tensor:
        if idx == 0:
            image = self.horizontal_flip(image)
        elif idx == 1:
            image = self.vertical_flip(image)
        else:
            image = self.rotate(image)  
        return image        


    def __getitem__(self, idx):
        sample = self.samples[idx]

        image_arrays: list = list()
        dirpath: str = sample["dirpath"]
        filenames: List[str] = self.sort_filenames(sample["filenames"])
        
        if self.use_data_aug:
            transform_idx = random.randint(0, 2)

        for filename in filenames:
            filepath: str = os.path.join(dirpath, filename).replace("\\", "/")
            assert os.path.exists(filepath), f"File {filepath} does not exist."
            image: torch.tensor = self.read_png(filepath=filepath)
            image: torch.tensor = self.transforms(image)
            image: torch.tensor = image.float().contiguous()
            # image: torch.tensor = torch.as_tensor(arr.copy()).float().contiguous()
            if self.use_data_aug:
                image = self.spatial_transform(image, idx=transform_idx)
            image_arrays.append(image)

        image_arrays = torch.stack(image_arrays, 0)
        # image_arrays = torch.swapaxes(image_arrays, 1, -1) # _ x W x H x C -> _ x C x H x W

        target: torch.tensor = torch.as_tensor(sample["label"])

        return {
            'X': image_arrays,
            'Y': target,
        }


    # @staticmethod
    # def sample_gen(dir_path: str, **kwargs) -> Generator:
    #     def read_png_as_arr(filepath: str) -> np.ndarray:
    #         img = Image.open(filepath).convert('RGB')
    #         arr = np.array(img)
    #         return arr


    #     for dirpath, dirnames, filenames in os.walk(dir_path):
    #         if not dirnames:
    #             image_arrays: list = list()
    #             for filename in filenames:
    #                 filepath: str = os.path.join(dirpath, filename).replace("\\", "/")
    #                 assert os.path.exists(filepath), f"File {filepath} does not exist."
    #                 arr = read_png_as_arr(filepath=filepath)
    #                 image: torch.tensor = torch.as_tensor(arr.copy()).float().contiguous()
    #                 image_arrays.append(image)

    #             image_arrays = torch.stack(image_arrays, 0)
    #             image_arrays = torch.swapaxes(image_arrays, 1, -1) # _ x W x H x C -> _ x C x H x W
    #             image_arrays = image_arrays[None, :, :, :, :]

    #             yield {
    #                 'X': image_arrays,
    #                 'Y': None,
    #                 "dirpath": dirpath,
    #                 # "filenames": filenames                    
    #             }


    # @staticmethod
    # def save_preds(input: dict, output: torch.tensor, **kwargs):
    #     dirpath: str = input["dirpath"]
    #     result: dict = {
    #         "Negative": float(output[0, 0]),
    #         "Positive": float(output[0, 1])
    #     }
    #     logging.info(
    #         f"""
    #                 Sample: {dirpath}
    #                 Positive: {result["Positive"]}
    #                 Negative: {result["Negative"]}
    #         """
    #     )
    #     # logging.info(f"Prediction: \n {output}")                 


# @TODO: IMPLEMENT
class ConvLSTMODDataset(ConvLSTMCDataset):
    __name__ = "ConvLSTMODDataset"


    def __init__(self, data_manifest_path: str):
        raise NotImplementedError("This class is not implemented yet.")
