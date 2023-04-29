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
