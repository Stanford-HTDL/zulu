__author__ = "Richard Correro (richard@richardcorrero.com)"


import argparse
import logging
import os
from typing import Generator, Optional

import numpy as np
import torch
from light_pipe import Data, Parallelizer, Transformer
from PIL import Image

from script_utils import arg_is_true, parse_args


class Processor:
    __name__ = "Processor"


class ConvLSTMCProcessor(Processor):
    __name__ = "ConvLSTMCProcessor"

    DEFAULT_PRED_MANIFEST: str = "conv_lstm_c_preds.json"
    DEFAULT_SAVE_MANIFEST: bool = False


    def __init__(self):
        args = self.parse_args()
        save_manifest: bool = arg_is_true(args["save_manifest"])
        if save_manifest:
            pred_manifest_path: str = args["pred_manifest"]
            os.makedirs(os.path.dirname(pred_manifest_path), exist_ok=True)
            self.pred_manifest_path = pred_manifest_path

        self.save_manifest = save_manifest


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--pred-manifest",
            default=self.DEFAULT_PRED_MANIFEST
        )
        parser.add_argument(
            "--save-manifest",
            default=self.DEFAULT_SAVE_MANIFEST
        )        
        args = parse_args(parser=parser)
        return args


    def make_samples(
        self, dir_path: str, parallelizer: Optional[Parallelizer] = Parallelizer(), 
        **kwargs
    ) -> Generator:
        def read_png_as_arr(filepath: str) -> np.ndarray:
            img = Image.open(filepath).convert('RGB')
            arr = np.array(img)
            return arr


        def walk_dir(dir_path: str) -> Generator:
            for dirpath, dirnames, filenames in os.walk(dir_path):
                if not dirnames:
                    yield dirpath, filenames


        def make_sample(input: tuple) -> dict:
            dirpath, filenames = input
            image_arrays: list = list()
            for filename in filenames:
                filepath: str = os.path.join(dirpath, filename).replace("\\", "/")
                assert os.path.exists(filepath), f"File {filepath} does not exist."
                arr = read_png_as_arr(filepath=filepath)
                image: torch.tensor = torch.as_tensor(arr.copy()).float().contiguous()
                image_arrays.append(image)

            image_arrays = torch.stack(image_arrays, 0)
            image_arrays = torch.swapaxes(image_arrays, 1, -1) # _ x W x H x C -> _ x C x H x W
            image_arrays = image_arrays[None, :, :, :, :]

            return {
                'X': image_arrays,
                'Y': None,
                "dirpath": dirpath.replace("\\", "/"),
                # "filenames": filenames                    
            }


        data = Data(walk_dir, dir_path=dir_path)

        data >> Transformer(
            make_sample, parallelizer=parallelizer
        )
        yield from data


    def save_results(self, input: dict, output: torch.tensor, **kwargs):
        dirpath: str = input["dirpath"]
        result: dict = {
            "Negative": float(output[0, 0]),
            "Positive": float(output[0, 1])
        }
        logging.info(
            f"""
                    Sample: {dirpath}
                    Positive: {result["Positive"]}
                    Negative: {result["Negative"]}
            """
        )              
