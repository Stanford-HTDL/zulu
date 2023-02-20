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
import pandas as pd
import torch
import torchvision.transforms as T
from light_pipe import AsyncGatherer, Data, Parallelizer, Transformer
from PIL import Image, ImageDraw

import mercantile
from detection import bbox_to_geojson
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


    def _open_geojson(self, geojson_path: str) -> dict:
        with open(geojson_path) as f:
            geojson: dict = json.load(f)
        return geojson        


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
        self, tiles: List[int], geojson_name: str, start: str, end: str, 
        false_color_index: str
    ) -> Generator:
        # try:
        #     geojson_name = geojson["name"]
        # except KeyError:
        if not geojson_name:
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
        self, tiles: List[mercantile.Tile], geojson_name: str, start: str, 
        end: str, false_color_index: Optional[str] = None
    ) -> Generator:
        # tiles = self._get_tiles(geojson, zooms=zooms, truncate=truncate)
        # tiles: List[mercantile.Tile] = list(set(tiles)) # Remove duplicates
        requests = self._make_papi_time_series_requests(
            tiles=tiles, geojson_name=geojson_name, start=start, end=end, 
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
        self, responses: List[bytes], z: int, x: int, y: int, geojson_name: str
    ) -> List[Image.Image]:
        images: list = list()
        for response in responses:
            img_bs = io.BytesIO(response)
            img = Image.open(img_bs).convert('RGB')
            images.append(img)
        return images, z, x, y, geojson_name


    def get_planet_monthly_time_series_as_PIL_Images(
        self, tiles: List[mercantile.Tile], start: str, end: str,
        geojson_name: Optional[str] = "xyz", false_color_index: Optional[str] = None, 
        image_parallelizer: Optional[Parallelizer] = Parallelizer()
    ) -> Generator:
        # with open(geojson_path) as f:
        #     geojson: dict = json.load(f)

        data: Data = Data(
            self._make_planet_monthly_time_series_requests, tiles=tiles,
            geojson_name=geojson_name, start=start, end=end, 
            false_color_index=false_color_index
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
    DEFAULT_GEOJSON_DIR: str = "pred_bbox_geojson/"
    DEFAULT_SAVE_MANIFEST: bool = False
    DEFAULT_SAVE_GEOJSON: bool = False    
    DEFAULT_FROM_LOCAL_FILES: bool = True

    DEFAULT_FROM_PREDS_CSV: bool = False
    DEFAULT_PREDS_CSV_PATH: str = DEFAULT_PRED_MANIFEST    
    DEFAULT_FILTER_BY_TARGET_VALUE: bool = True
    DEFAULT_TARGET_VALUE: int = 1
    DEFAULT_TARGET_COLUMN_NAME: str = "Predicted Class"
    DEFAULT_COORDINATE_COLUMN_NAMES: List[str] = ["Z", "X", "Y"]
    DEFAULT_BBOX_THRESHOLD: float = 1e-2

    NUM_TILES_PER_SUBLIST = 128
    INPUT_SIZE = (224,224)
    TRANSORMS = T.Compose([
        T.Resize(INPUT_SIZE),
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
    DEFAULT_SAVE_IMAGES = True
    DEFAULT_EMBED_DATE = True
    DEFAULT_MAKE_GIFS = True

    LOCAL_PRED_CSV_HEADER = ["Directory", "Positive", "Negative", "Predicted Class"]
    # PAPI_PRED_CSV_HEADER = [
    #     "Z", "X", "Y", "Longitude", "Latitude", "Geojson Name", "Positive", "Negative", "Predicted Class"
    # ]
    PAPI_PRED_CSV_HEADER = [
        "Z", "X", "Y", "West", "South", "East", "North", "Geojson Name", "Positive", "Negative", "Predicted Class"
    ]    


    def __init__(self, save_dir: str):
        args = self.parse_args()
        save_manifest: bool = arg_is_true(args["save_manifest"])
        from_local_files = arg_is_true(args["from_local_files"])
        save_geojson: bool = arg_is_true(args["save_geojson"])
        self.save_geojson = save_geojson
        if save_geojson:
            bbox_geojson_dir = args["bbox_geojson_dir"]
            bbox_geojson_dir = os.path.join(save_dir, bbox_geojson_dir).replace("\\", "/")
            os.makedirs(bbox_geojson_dir, exist_ok=True)
            self.bbox_geojson_dir = bbox_geojson_dir 
        if save_manifest:
            pred_manifest: str = args["pred_manifest"]
            pred_manifest_path: str = os.path.join(save_dir, pred_manifest).replace("\\", "/")
            # os.makedirs(os.path.dirname(pred_manifest_path), exist_ok=True)

            if from_local_files:
                with open(pred_manifest_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.LOCAL_PRED_CSV_HEADER)                
            else:
                with open(pred_manifest_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.PAPI_PRED_CSV_HEADER)   

            self.save_dir = save_dir
            self.pred_manifest_path = pred_manifest_path

        self.save_manifest = save_manifest
        self.from_local_files = from_local_files
        self.planet_api_key = args["planet_api_key"]
        self.start = args["start"]
        self.end = args["end"]
        self.zooms = args["zooms"]
        self.duration = args["duration"]
        self.fc_index = args["fc_index"]

        self.from_preds_csv = arg_is_true(args["from_preds_csv"])
        self.preds_csv_path = args["preds_csv_path"]
        self.filter_by_target_value = arg_is_true(args["filter_by_target_value"])
        self.target_value = int(args["target_value"])
        self.target_column_name = args["target_column_name"]
        self.coordinate_column_names = args["coordinate_column_names"]
        self.bbox_threshold = float(args["bbox_threshold"])

        self.embed_date = arg_is_true(args["embed_date"])
        self.save_images = arg_is_true(args["save_images"])
        self.make_gifs = arg_is_true(args["make_gifs"])
        self.args = args

    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--pred-manifest",
            default=self.DEFAULT_PRED_MANIFEST
        )
        parser.add_argument(
            "--bbox-geojson-dir",
            default=self.DEFAULT_GEOJSON_DIR
        )
        parser.add_argument(
            "--save-manifest",
            default=self.DEFAULT_SAVE_MANIFEST
        )
        parser.add_argument(
            "--save-geojson",
            default=self.DEFAULT_SAVE_GEOJSON
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
            "--save-images",
            default=self.DEFAULT_SAVE_IMAGES
        )
        parser.add_argument(
            "--embed-date",
            default=self.DEFAULT_EMBED_DATE,
        )
        parser.add_argument(
            "--make-gifs",
            default=self.DEFAULT_MAKE_GIFS,
        )    
        parser.add_argument(
            "--from-preds-csv",
            default=self.DEFAULT_FROM_PREDS_CSV
        )
        parser.add_argument(
            "--preds-csv-path",
            default=self.DEFAULT_PREDS_CSV_PATH
        )        
        parser.add_argument(
            "--filter-by-target-value",
            default=self.DEFAULT_FILTER_BY_TARGET_VALUE
        )
        parser.add_argument(
            "--target-value",
            default=self.DEFAULT_TARGET_VALUE,
            type=int
        )
        parser.add_argument(
            "--target-column-name",
            default=self.DEFAULT_TARGET_COLUMN_NAME
        )
        parser.add_argument(
            "--coordinate-column-names",
            default=self.DEFAULT_COORDINATE_COLUMN_NAMES,
            nargs="+",
            type=str
        )
        parser.add_argument(
            "--bbox-threshold",
            default=self.DEFAULT_BBOX_THRESHOLD,
            type=float,
        )
        args = parse_args(parser=parser)
        return args                    


    def _save_pil_images(
        self, images: List[Image.Image], z: int, x: int, y: int, target_name: str,
        start, end, duration, save_dir: str, make_gifs: Optional[bool] = False,
        save_images: Optional[bool] = True, embed_date = True, 
        timelapse_format: Optional[str] = "gif", image_format: Optional[str] = "png",
        loop: Optional[int] = 0, tile_dir: Optional[str] = "xyz_tiles"
    ) -> Generator:
        dates = self._get_mosaic_time_str_from_start_end(start, end)
        if embed_date:
            for date, image in list(zip(dates, images)):
                year, month = date.split("_")
                draw = ImageDraw.Draw(image)
                draw.text((0, 0), f"{year} {month} {z} {x} {y}",(255,255,255))                   

        if make_gifs:
            gif_filename = f"{start}_{end}/{z}/{target_name}/{z}_{x}_{y}/{start}_{end}.{timelapse_format}"
            gif_filepath = os.path.join(save_dir, tile_dir, f"{timelapse_format}s", gif_filename).replace("\\", "/")
            os.makedirs(os.path.dirname(gif_filepath), exist_ok=True)
            imgs_iter = iter(images)
            first_img = next(imgs_iter)
            first_img.save(fp=gif_filepath, format='GIF', append_images=imgs_iter,
                    save_all=True, duration=duration, loop=loop, interlace=False,
                    include_color_table=True)                     

        if save_images:
            for date, image in list(zip(dates, images)):
                image_filename = f"{start}_{end}/{z}/{target_name}/{z}_{x}_{y}/{date}.{image_format}"
                image_filepath = os.path.join(save_dir, tile_dir, f"{image_format}s", image_filename).replace("\\", "/")
                os.makedirs(os.path.dirname(image_filepath), exist_ok=True)
                image.save(fp=image_filepath, format=image_format)
        return images, z, x, y, target_name


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


    @staticmethod
    def _get_tiles_from_preds_csv_path(
        preds_csv_path: str,
        filter_by_target_value: Optional[bool] = False,
        target_column_name: Optional[str] = "Predicted Class", 
        target_value: Optional[int] = 1,
        coordinate_column_names: Optional[List[str]] = ["Z", "X", "Y"]
    ):
        df = pd.read_csv(preds_csv_path)
        if filter_by_target_value and target_column_name is not None:
            tile_coordinates = df[df[target_column_name] == target_value][coordinate_column_names]
        else:
            tile_coordinates = df[coordinate_column_names]
        for tile_coords in tile_coordinates.values:
            z = tile_coords[0]
            x = tile_coords[1]
            y = tile_coords[2]
            tile = mercantile.Tile(x=x, y=y, z=z)
            yield tile        


    def _make_samples_from_preds_csv_path(
        self, preds_csv_path: str, start: str, end: str, zooms: List[int], 
        false_color_index: Optional[str] = None, truncate: Optional[bool] = True, 
        num_tiles_per_sublist: Optional[int] = 128,
        filter_by_target_value: Optional[bool] = False,
        target_column_name: Optional[str] = "Predicted Class", 
        target_value: Optional[int] = 1,
        coordinate_column_names: Optional[List[str]] = ["Z", "X", "Y"],        
        image_parallelizer: Optional[Parallelizer] = Parallelizer(),
        parallelizer: Optional[Parallelizer] = Parallelizer()
    ):

        # def yield_tiles(tiles: List[mercantile.Tile]) -> Generator:
        #     yield tiles


        data: Data = Data(
            self._get_tiles_from_preds_csv_path, preds_csv_path=preds_csv_path,
            filter_by_target_value=filter_by_target_value, 
            target_value=target_value, target_column_name=target_column_name,
            coordinate_column_names=coordinate_column_names
        )

        # data >> Transformer(tuple_to_args(self.get_filepaths), as_list=False) \
        #      >> Transformer(self._open_geojson) \
        #      >> Transformer(tuple_to_args(self._get_tiles), zooms=zooms, truncate=truncate)

        tiles: List[mercantile.Tile] = data(block=True)
        tiles = list(set(tiles)) # Remove duplicates
        
        # Chunk tiles to prevent order bottlenecks
        tiles = [
            tiles[i:i + num_tiles_per_sublist] for i in range(0, len(tiles), num_tiles_per_sublist)
        ]
        # tiles = [tiles]

        data = Data(tiles)

        data >> Transformer(
                tuple_to_args(self.get_planet_monthly_time_series_as_PIL_Images),
                start=start, end=end, false_color_index=false_color_index, 
                image_parallelizer=image_parallelizer, parallelizer=parallelizer
             )

        if self.save_images or self.make_gifs:
            data >> Transformer(
                tuple_to_args(self._save_pil_images), save_dir=self.save_dir,
                start=start, end=end, duration=self.duration, 
                make_gifs=self.make_gifs, save_images=self.save_images,
                embed_date=self.embed_date, parallelizer=parallelizer
            )

        data >> Transformer(tuple_to_args(self.make_sample), parallelizer=parallelizer)

        yield from data           


    def _make_samples_from_planet_api(
        self, geojson_dir_path: str, start: str, end: str, zooms: List[int], 
        false_color_index: Optional[str] = None, truncate: Optional[bool] = True, 
        num_tiles_per_sublist: Optional[int] = 128,
        image_parallelizer: Optional[Parallelizer] = Parallelizer(),
        parallelizer: Optional[Parallelizer] = Parallelizer()
    ):

        # def yield_tiles(tiles: List[mercantile.Tile]) -> Generator:
        #     yield tiles


        data: Data = Data(self.walk_dir, dir_path=geojson_dir_path)

        data >> Transformer(tuple_to_args(self.get_filepaths), as_list=False) \
             >> Transformer(self._open_geojson) \
             >> Transformer(tuple_to_args(self._get_tiles), zooms=zooms, truncate=truncate)

        tiles: List[mercantile.Tile] = data(block=True)
        tiles = list(set(tiles)) # Remove duplicates
        
        # Chunk tiles to prevent order bottlenecks
        tiles = [
            tiles[i:i + num_tiles_per_sublist] for i in range(0, len(tiles), num_tiles_per_sublist)
        ]
        # tiles = [tiles]

        data = Data(tiles)

        data >> Transformer(
                tuple_to_args(self.get_planet_monthly_time_series_as_PIL_Images),
                start=start, end=end, false_color_index=false_color_index, 
                image_parallelizer=image_parallelizer, parallelizer=parallelizer
             )

        if self.save_images or self.make_gifs:
            data >> Transformer(
                tuple_to_args(self._save_pil_images), save_dir=self.save_dir,
                start=start, end=end, duration=self.duration, 
                make_gifs=self.make_gifs, save_images=self.save_images,
                embed_date=self.embed_date, parallelizer=parallelizer
            )

        data >> Transformer(tuple_to_args(self.make_sample), parallelizer=parallelizer)

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
        elif self.from_preds_csv:
            yield from self._make_samples_from_preds_csv_path(
                preds_csv_path=self.preds_csv_path, start=self.start, end=self.end, 
                zooms=self.zooms, false_color_index=self.fc_index, 
                parallelizer=parallelizer, 
                num_tiles_per_sublist=self.NUM_TILES_PER_SUBLIST, 
                filter_by_target_value=self.filter_by_target_value,
                target_column_name=self.target_column_name, 
                target_value=self.target_value,
                coordinate_column_names=self.coordinate_column_names,                 
                **kwargs                
            )
        else:
            yield from self._make_samples_from_planet_api(
                geojson_dir_path=dir_path, start=self.start, end=self.end, 
                zooms=self.zooms, false_color_index=self.fc_index, 
                parallelizer=parallelizer, 
                num_tiles_per_sublist=self.NUM_TILES_PER_SUBLIST, **kwargs
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
        z: int = input["args"][0]
        x: int = input["args"][1]
        y: int = input["args"][2]
        geojson_name: str = input["args"][3]

        tile: mercantile.Tile = mercantile.Tile(x=x, y=y, z=z)
        # ul: mercantile.LngLat = mercantile.ul(tile)
        # lng = ul.lng
        # lat = ul.lat
        lng_lat_bbox: mercantile.LngLatBbox = mercantile.bounds(tile)
        west: float = lng_lat_bbox.west
        south: float = lng_lat_bbox.south
        east: float = lng_lat_bbox.east
        north: float = lng_lat_bbox.north


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
        results_list: list = [z, x, y, west, south, east, north, geojson_name, result["Positive"], result["Negative"], predicted_class]
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


class ResNetProcessor(ConvLSTMCProcessor):
    __name__ = "ResNetProcessor"


    def make_sample(self, images: List[Image.Image], *args) -> dict:
        for image in images:
            image: torch.Tensor = self.TRANSORMS(image)
            image: torch.Tensor = image.float().contiguous()
            image: torch.Tensor = image.unsqueeze(0)

            yield {
                'X': image,
                'Y': None,
                "args": args
            }


class ObjectDetectorProcessor(ResNetProcessor):
    __name__ = "ObjectDetectorProcessor"

    LOCAL_PRED_CSV_HEADER = ["Directory", "Boxes", "Labels", "Scores"]
    PAPI_PRED_CSV_HEADER = [
        "Z", "X", "Y", "West", "South", "East", "North", "Geojson Name", "Boxes", "Labels", "Scores"
    ]


    @staticmethod
    def make_result(output: dict) -> dict:
        result = dict()
        for key, value in output.items():
            result[key] = value.detach().cpu().numpy().tolist()        
        return result


    def _save_results_from_local_files(self, input: dict, output: torch.Tensor) -> None:
        filepaths: List[str] = input["args"][0]
        dirpath: str = os.path.dirname(os.path.abspath(filepaths[0])).replace("\\", "/")
        result: dict = self.make_result(output[0]) # One image at a time

        logging.info(
            f"""
                    Dirpath: {dirpath}
                    Boxes: {result["boxes"]}
                    Labels: {result["labels"]}
                    Scores: {result["scores"]}
            """
        )
        results_list: list = [dirpath, result["boxes"], result["labels"], result["scores"]]
        if self.save_manifest:
            with open(self.pred_manifest_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results_list)      


    def _save_results_from_planet_api(self, input: dict, output: torch.Tensor) -> None:
        z: int = input["args"][0]
        x: int = input["args"][1]
        y: int = input["args"][2]
        geojson_name: str = input["args"][3]

        tile: mercantile.Tile = mercantile.Tile(x=x, y=y, z=z)
        # ul: mercantile.LngLat = mercantile.ul(tile)
        # lng = ul.lng
        # lat = ul.lat
        lng_lat_bbox: mercantile.LngLatBbox = mercantile.bounds(tile)
        west: float = lng_lat_bbox.west
        south: float = lng_lat_bbox.south
        east: float = lng_lat_bbox.east
        north: float = lng_lat_bbox.north      

        result: dict = self.make_result(output[0]) # One image at a time

        logging.info(
            f"""
                    Z/X/Y: {z}/{x}/{y}
                    Geojson Name: {geojson_name}
                    Boxes: {result["boxes"]}
                    Labels: {result["labels"]}
                    Scores: {result["scores"]}
            """
        )
        results_list: list = [z, x, y, west, south, east, north, geojson_name, result["boxes"], result["labels"], result["scores"]]
        if self.save_manifest:
            with open(self.pred_manifest_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(results_list)

        zxy_str = f"{z}_{x}_{y}"

        # Save bounding boxes as geojson files
        if self.save_geojson:
            bboxes_to_save: list = list()
            for score, bbox in list(zip(result["scores"], result["boxes"])):
                if score > self.bbox_threshold:
                    bboxes_to_save.append(bbox)

            if len(bboxes_to_save) > 0:
                bbox_save_dir = os.path.join(self.bbox_geojson_dir, zxy_str).replace("\\", "/")
                os.makedirs(bbox_save_dir, exist_ok=True)
                for i, bbox in enumerate(bboxes_to_save):
                    bbox_geojson: dict = bbox_to_geojson(bbox, lng_lat_bbox, self.INPUT_SIZE)
                    bbox_savepath: str = os.path.join(bbox_save_dir, f"bbox_{i+1}.geojson").replace("\\", "/")
                    with open(bbox_savepath, "w") as f:
                        json.dump(bbox_geojson, f)
            