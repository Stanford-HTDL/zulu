import json
import os
from typing import Optional, List


def find_file(filename, directory):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename).replace("\\", "/")
    return None


def merge_json_files(source_dir, target_file, value_key="filename"):
    merged_dict = {}
    for filename in os.listdir(source_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(source_dir, filename)
            with open(file_path) as f:
                json_dict = json.load(f)
                for _, value in json_dict.items():
                    assert "regions" in value.keys()
                    if len(value["regions"]) > 0:
                        assert value_key in value.keys()
                        filename: str = value[value_key]
                        merged_dict[filename] = value
    with open(target_file, 'w') as f:
        json.dump(merged_dict, f, indent=4)


def prepare_sios_samples(
        annotation_dict: dict, imagery_dir: str, label: Optional[str] = "Positive"
) -> List[dict]:
    samples = list()
    for key, value in annotation_dict.items():
        filename = key
        filepath: str = find_file(filename=filename, directory=imagery_dir)
        assert filepath is not None, f"File {filename} not found in directory {imagery_dir}."
        sample_dict: dict = {
            "filepath": filepath,
            "annotations": value,
            "label": label
        }
        samples.append(sample_dict)
    return samples


def get_filepaths(directory, extension: str = ".png"):
    paths_list = list()
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(extension):
                filepath = os.path.join(root, file).replace("\\", "/")
                paths_list.append(filepath)
    return paths_list


def get_negative_sios_samples(
    imagery_dir: str, negative_dir_name: str = "Negative", 
    label: Optional[str] = "Negative"
) -> list:
    negative_dir_path: str = os.path.join(imagery_dir, negative_dir_name).replace("\\", "/")
    paths_list = get_filepaths(negative_dir_path)
    samples = list()
    for path in paths_list:
        sample_dict: dict = {
            "filepath": path,
            "annotations": None,
            "label": label
        }
        samples.append(sample_dict)
    return samples
