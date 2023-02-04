__author__ = "Richard Correro (richard@richardcorrero.com)"


import argparse
import csv
import datetime
import io
import json
import logging
import os
from collections import OrderedDict
from typing import Generator, List, Optional

import aiohttp
import torch
import torchvision.transforms as T
from light_pipe import AsyncGatherer, Data, Parallelizer, Transformer
from PIL import Image

import mercantile
from script_utils import (arg_is_true, async_tuple_to_args, parse_args,
                          tuple_to_args)


class Processor:
    __name__ = "Processor"   


    def sort_filenames(self, filenames: List[str]) -> List[str]:
        filenames_sorted = sorted(filenames)
        return filenames_sorted    


    def walk_dir(self, dir_path: str, sort_filenames: Optional[bool] = True) -> Generator:
        for dirpath, dirnames, filenames in os.walk(dir_path):
            if not dirnames:
                if sort_filenames:
                    filenames = self.sort_filenames(filenames)
                yield dirpath, filenames


    def get_filepaths(
        self, dirpath: str, filenames: List[str], 
        as_list: Optional[bool] = False
    ) -> Generator:
        if as_list:
            filepaths: list = list()
        for filename in filenames:
            filepath: str = os.path.join(dirpath, filename).replace("\\", "/")
            if as_list:
                filepaths.append(filepath)
            else:
                yield filepath
        if as_list:
            yield filepaths


    def read_file_as_pil_image(self, filepath: str) -> Image:
        img = Image.open(filepath).convert('RGB')
        return img


    def read_files_as_pil_image(
        self, filepaths: List[str], return_filepaths: Optional[bool] = False
    ) -> List[Image.Image]:
        images: list = list()
        for filepath in filepaths:
            image: Image.Image = self.read_file_as_pil_image(filepath)
            images.append(image)
        if return_filepaths:
            return images, filepaths
        return images


class TimeSeriesProcessor(Processor):
    __name__ = "TimeSeriesProcessor"

    VALID_FC_INDICES = [
        "ndvi", "ndwi", "msavi2", "mtvi2", "vari", "tgi"
    ]   


    def _get_tiles(self, geojson: dict, zooms: List[str], truncate: bool) -> Generator:  
        geo_bounds = mercantile.geojson_bounds(geojson)
        west = geo_bounds.west
        south = geo_bounds.south
        east = geo_bounds.east
        north = geo_bounds.north

        tiles = mercantile.tiles(west, south, east, north, zooms, truncate)
        for tile in tiles:
            yield tile     
  

    def _get_mosaic_time_str_from_start_end(self, start: str, end: str) -> OrderedDict:
        dates = [start, end]
        start, end = [datetime.datetime.strptime(_, "%Y_%m") for _ in dates]
        return OrderedDict(((start + datetime.timedelta(_)).strftime(r"%Y_%m"), None) \
            for _ in range((end - start).days)).keys()


    def _make_papi_time_series_requests(
        self, tiles: List[int], geojson: dict, start: str, end: str, 
        false_color_index: str
    ) -> Generator:
        try:
            geojson_name = geojson["name"]
        except KeyError:
            geojson_name = "geojson"
        for tile in tiles:
            z = tile.z
            x = tile.x
            y = tile.y            
            request_urls = list()
            for year_month in self._get_mosaic_time_str_from_start_end(start, end):
                request_url = \
                    f"https://tiles.planet.com/basemaps/v1/planet-tiles/global_monthly_{year_month}_mosaic/gmap/{z}/{x}/{y}.png?api_key={self.planet_api_key}"
                if false_color_index:
                    request_url += f"&proc={false_color_index}"
                request_urls.append(request_url)
            yield request_urls, z, x, y, geojson_name


    def _make_planet_monthly_time_series_requests(
        self, geojson: dict, start: str, end: str, zooms: List[int], 
        false_color_index: Optional[str] = None, truncate: Optional[bool] = True
    ) -> Generator:
        tiles = self._get_tiles(geojson, zooms=zooms, truncate=truncate)
        requests = self._make_papi_time_series_requests(
            tiles=tiles, geojson=geojson, start=start, end=end, 
            false_color_index=false_color_index
        )
        yield from requests 


    async def _post_monthly_mosaic_request(
        self, request_urls: List[str], z: int, x: int, y: int, geojson_name: str
    ):
        responses = list()
        async with aiohttp.ClientSession() as session:
            for request_url in request_urls:
                async with session.get(request_url) as response:
                    response.raise_for_status()
                    content = await response.read()
                    responses.append(content)
        return responses, z, x, y, geojson_name


    def _get_pil_images_from_response(
        self, responses: List[bytes], *args
    ) -> List[Image.Image]:
        images: list = list()
        for response in responses:
            img_bs = io.BytesIO(response)
            img = Image.open(img_bs).convert('RGB')
            images.append(img)
        return images, args   


    def get_planet_monthly_time_series_as_PIL_Images(
        self, geojson_path: str, start: str, end: str, zooms: List[int],
        false_color_index: Optional[str] = None, truncate: Optional[bool] = True,
        image_parallelizer: Optional[Parallelizer] = Parallelizer()
    ) -> Generator:
        with open(geojson_path) as f:
            geojson: dict = json.load(f)

        data: Data = Data(
            self._make_planet_monthly_time_series_requests, geojson=geojson,
            start=start, end=end, zooms=zooms, false_color_index=false_color_index,
            truncate=truncate
        )
        data >> Transformer(
            async_tuple_to_args(self._post_monthly_mosaic_request), parallelizer=AsyncGatherer()
        )
        data >> Transformer(
            tuple_to_args(self._get_pil_images_from_response), parallelizer=image_parallelizer
        )

        yield from data


class ConvLSTMCProcessor(TimeSeriesProcessor):
    __name__ = "ConvLSTMCProcessor"

    DEFAULT_PRED_MANIFEST: str = "conv_lstm_c_preds.csv"
    DEFAULT_SAVE_MANIFEST: bool = False
    DEFAULT_FROM_LOCAL_FILES: bool = True

    TRANSORMS = T.Compose([
        T.Resize((224,224)),
        # T.CenterCrop((224,224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    DEFAULT_START = "2022_01"
    DEFAULT_END = "2023_01"
    DEFAULT_ZOOMS = [15]
    DEFAULT_DURATION = 250.0
    DEFAULT_EMBED_DATE = True
    DEFAULT_MAKE_GIFS = True

    LOCAL_PRED_CSV_HEADER = ["Directory", "Positive", "Negative", "Predicted Class"]
    PAPI_PRED_CSV_HEADER = [
        "Z", "X", "Y", "Longitude", "Latitude", "Geojson Name", "Positive", "Negative", "Predicted Class"
    ]


    def __init__(self):
        args = self.parse_args()
        save_manifest: bool = arg_is_true(args["save_manifest"])
        from_local_files = arg_is_true(args["from_local_files"])
        if save_manifest:
            pred_manifest_path: str = args["pred_manifest"]
            # os.makedirs(os.path.dirname(pred_manifest_path), exist_ok=True)

            if from_local_files:
                with open(pred_manifest_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.LOCAL_PRED_CSV_HEADER)                
            else:
                with open(pred_manifest_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.PAPI_PRED_CSV_HEADER)   

            self.pred_manifest_path = pred_manifest_path

        self.save_manifest = save_manifest
        self.from_local_files = from_local_files
        self.planet_api_key = args["planet_api_key"]
        self.start = args["start"]
        self.end = args["end"]
        self.zooms = args["zooms"]
        self.duration = args["duration"]
        self.fc_index = args["fc_index"]
        self.embed_date = args["embed_date"]
        self.make_gifs = args["make_gifs"]        


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
        parser.add_argument(
            "--from-local-files",
            default=self.DEFAULT_FROM_LOCAL_FILES
        )
        parser.add_argument(
            "--planet-api-key"
        )
        parser.add_argument(
            "--start",
            default=self.DEFAULT_START
        )
        parser.add_argument(
            "--end",
            default=self.DEFAULT_END
        )
        parser.add_argument(
            "--zooms",
            nargs="+",
            type=int,
            default=self.DEFAULT_ZOOMS
        )
        parser.add_argument(
            "--duration",
            default=self.DEFAULT_DURATION,
            type=float
        )
        parser.add_argument(
            "--fc-index",
            default=None
        )
        parser.add_argument(
            "--embed-date",
            default=self.DEFAULT_EMBED_DATE,
        )
        parser.add_argument(
            "--make-gifs",
            default=self.DEFAULT_MAKE_GIFS,
        )        
        args = parse_args(parser=parser)
        return args 


    def make_sample(self, images: List[Image.Image], *args) -> dict:
        image_tensors: list = list()
        for image in images:
            image: torch.Tensor = self.TRANSORMS(image)
            image: torch.Tensor = image.float().contiguous()
            image_tensors.append(image)
        image_tensors: torch.Tensor = torch.stack(image_tensors, 0)
        # image_arrays = torch.swapaxes(image_arrays, 1, -1) # _ x W x H x C -> _ x C x H x W
        # image_arrays = image_arrays[None, :, :, :, :]
        image_tensors = image_tensors.unsqueeze(0)

        return {
            'X': image_tensors,
            'Y': None,
            "args": args
        }         


    def _make_samples_from_planet_api(
        self, geojson_dir_path: str, start: str, end: str, zooms: List[int], 
        false_color_index: Optional[str] = None, truncate: Optional[bool] = True, 
        image_parallelizer: Optional[Parallelizer] = Parallelizer(),
        parallelizer: Optional[Parallelizer] = Parallelizer()
    ):
        data: Data = Data(self.walk_dir, dir_path=geojson_dir_path)

        data >> Transformer(tuple_to_args(self.get_filepaths)) \
             >> Transformer(
                tuple_to_args(self.get_planet_monthly_time_series_as_PIL_Images),
                start=start, end=end, zooms=zooms, false_color_index=false_color_index,
                truncate=truncate, image_parallelizer=image_parallelizer,
                parallelizer=parallelizer
             ) \
             >> Transformer(tuple_to_args(self.make_sample), parallelizer=parallelizer)

        yield from data


    def _make_samples_from_local_files(
        self, dir_path: str, 
        parallelizer: Optional[Parallelizer] = Parallelizer()
    ) -> Generator:
        data = Data(self.walk_dir, dir_path=dir_path)

        data >> Transformer(tuple_to_args(self.get_filepaths), as_list=True) \
             >> Transformer(
                tuple_to_args(self.read_files_as_pil_image), return_filepaths=True,
                parallelizer=parallelizer
             ) \
             >> Transformer(
                tuple_to_args(self.make_sample), parallelizer=parallelizer
             )
        yield from data


    def make_samples(
        self, from_local_files: Optional[bool] = None, 
        dir_path: Optional[str]  = None,
        parallelizer: Optional[Parallelizer] = Parallelizer(), 
        **kwargs
    ) -> Generator:
        if from_local_files is None:
            from_local_files: bool = self.from_local_files
        if from_local_files:
            yield from self._make_samples_from_local_files(
                dir_path=dir_path, parallelizer=parallelizer, **kwargs
            )
        else:
            yield from self._make_samples_from_planet_api(
                geojson_dir_path=dir_path, start=self.start, end=self.end, 
                zooms=self.zooms, false_color_index=self.fc_index, 
                parallelizer=parallelizer, **kwargs
            )


    def _save_results_from_local_files(self, input: dict, output: torch.Tensor) -> None:
        filepaths: List[str] = input["args"][0]
        dirpath: str = os.path.dirname(os.path.abspath(filepaths[0])).replace("\\", "/")
        result: dict = {
            "Negative": float(output[0, 0]),
            "Positive": float(output[0, 1])
        }
        predicted_class = int(torch.argmax(output[0]))

        logging.info(
            f"""
                    Directory: {dirpath}
                    Positive: {result["Positive"]}
                    Negative: {result["Negative"]}
            """
        )
        results_list: list = [dirpath, result["Positive"], result["Negative"], predicted_class]
        if self.save_manifest:
            with open(self.pred_manifest_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results_list)        


    def _save_results_from_planet_api(self, input: dict, output: torch.Tensor) -> None:
        z: int = input["args"][0][0]
        x: int = input["args"][0][1]
        y: int = input["args"][0][2]
        geojson_name: str = input["args"][0][3]

        tile: mercantile.Tile = mercantile.Tile(x=x, y=y, z=z)
        ul: mercantile.LngLat = mercantile.ul(tile)
        lng = ul.lng
        lat = ul.lat

        result: dict = {
            "Negative": float(output[0, 0]),
            "Positive": float(output[0, 1])
        }
        predicted_class = int(torch.argmax(output[0]))

        logging.info(
            f"""
                    Z/X/Y: {z}/{x}/{y}
                    Geojson Name: {geojson_name}
                    Positive: {result["Positive"]}
                    Negative: {result["Negative"]}
            """
        )
        results_list: list = [z, x, y, lng, lat, geojson_name, result["Positive"], result["Negative"], predicted_class]
        if self.save_manifest:
            with open(self.pred_manifest_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results_list)



    def save_results(
        self, input: dict, output: torch.Tensor, from_local_files: Optional[bool] = None
    ) -> None:
        if from_local_files is None:
            from_local_files: bool = self.from_local_files
        if from_local_files:
            self._save_results_from_local_files(input=input, output=output)
        else:
            self._save_results_from_planet_api(input=input, output=output)
