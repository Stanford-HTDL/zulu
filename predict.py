import argparse
import logging
import os
import time
from typing import Generator

import numpy as np
import torch
from PIL import Image

# from datasets import ConvLSTMCDataset, EurosatDataset
from models import CNNLSTM, SpectrumNet, SqueezeNet
# from optimizers import SGD
from script_utils import get_args, get_random_string

SCRIPT_PATH = os.path.basename(__file__)

# DEFAULT_NUM_CHANNELS = 4
# DEFAULT_NUM_CLASSES = 10
# DEFAULT_DROPOUT = 0.5
# DEFAULT_BATCH_SIZE = 64
# DEFAULT_NUM_EPOCHS = 64
# DEFAULT_OPTIMIZER = SGD.__name__
# DEFAULT_LR = 1e-3
# DEFAULT_MOMENTUM = 0.9
# DEFAULT_NESTEROV = True
# DEFAULT_WEIGHT_DECAY = 5e-4
# DEFAULT_SCHEDULER = "StepLR"
# DEFAULT_STEP_SIZE = 10
# DEFAULT_SCHEDULER_GAMMA = 0.75
DEFAULT_MODEL_NAME = CNNLSTM.__name__
# DEFAULT_DATASET_NAME = EurosatDataset.__name__
# DEFAULT_VALIDATION_PERCENT = 0.15
# DEFAULT_SHUFFLE = True
# DEFAULT_CRITERION_NAME = "CrossEntropyLoss"
DEFAULT_EXPERIMENT_DIR = "experiments/"
# DEFAULT_NUM_WORKERS = os.cpu_count()
# DEFAULT_PIN_MEMORY = True
DEFAULT_DEVICE = "CUDA if available else CPU"
# DEFAULT_MIXED_PRECISION = True
# DEFAULT_SAVE_MODEL = True
# DEFAULT_SAVE_EVERY = 8
DEFAULT_CHANNEL_AXIS = 1
DEFAULT_EXPERIMENT_DIR: str = "experiments/"

DEFAULT_SEED = 8675309 # (___)-867-5309
torch.manual_seed(DEFAULT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(DEFAULT_SEED)

# REMEMBER to update as new models are added!
MODELS = {
    SqueezeNet.__name__: SqueezeNet,
    SpectrumNet.__name__: SpectrumNet,
    CNNLSTM.__name__: CNNLSTM
}

# DATASETS = {
#     EurosatDataset.__name__: EurosatDataset,
#     ConvLSTMCDataset.__name__: ConvLSTMCDataset
# }

# CRITERIA = {
#     "CrossEntropyLoss": torch.nn.CrossEntropyLoss
# }

# OPTIMIZERS = {
#     SGD.__name__: SGD
# }
# SCHEDULERS = {
#     "StepLR": torch.optim.lr_scheduler.StepLR
# }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME
    )
    parser.add_argument(
        "--model-filepath",
        required=True
    )
    # parser.add_argument(
    #     "--dataset",
    #     default=DEFAULT_DATASET_NAME
    # )
    # parser.add_argument(
    #     "--data-manifest",
    #     default=DEFAULT_DATA_MANIFEST
    # )    
    # parser.add_argument(
    #     "--val-percent",
    #     default=DEFAULT_VALIDATION_PERCENT,
    #     type=float
    # )    
    # parser.add_argument(
    #     "--shuffle",
    #     default=DEFAULT_SHUFFLE,
    #     type=bool
    # )
    # parser.add_argument(
    #     "--criterion",
    #     default=DEFAULT_CRITERION_NAME,
    # )
    # parser.add_argument(
    #     "--num-channels",
    #     default=DEFAULT_NUM_CHANNELS,
    #     type=int
    # )
    # parser.add_argument(
    #     "--num-classes",
    #     default=DEFAULT_NUM_CLASSES,
    #     type=int
    # )
    # parser.add_argument(
    #     "--dropout",
    #     default=DEFAULT_DROPOUT,
    #     type=float
    # )    
    # parser.add_argument(
    #     "--batch-size",
    #     default=DEFAULT_BATCH_SIZE,
    #     type=int
    # )  
    # parser.add_argument(
    #     "--num-epochs",
    #     default=DEFAULT_NUM_EPOCHS,
    #     type=int
    # )
    # parser.add_argument(
    #     "--optimizer",
    #     default=DEFAULT_OPTIMIZER,
    # )    
    # parser.add_argument(
    #     "--lr",
    #     default=DEFAULT_LR,
    #     type=float
    # )
    # parser.add_argument(
    #     "--momentum",
    #     default=DEFAULT_MOMENTUM,
    #     type=float
    # )
    # parser.add_argument(
    #     "--nesterov",
    #     default=DEFAULT_NESTEROV,
    #     type=bool
    # )    
    # parser.add_argument(
    #     "--weight-decay",
    #     default=DEFAULT_WEIGHT_DECAY,
    #     type=float
    # )   
    # parser.add_argument(
    #     "--scheduler",
    #     default=DEFAULT_SCHEDULER
    # )
    # parser.add_argument(
    #     "--step-size",
    #     default=DEFAULT_STEP_SIZE,
    #     type=int
    # )
    # parser.add_argument(
    #     "--scheduler-gamma",
    #     default=DEFAULT_SCHEDULER_GAMMA,
    #     type=float
    # )    
    # parser.add_argument(
    #     "--num-workers",
    #     default=DEFAULT_NUM_WORKERS,
    #     type=int
    # )  
    # parser.add_argument(
    #     "--pin-memory",
    #     default=DEFAULT_PIN_MEMORY,
    #     type=bool
    # )       
    parser.add_argument(
        '--device',
        default=DEFAULT_DEVICE
    )
    # parser.add_argument(
    #     "--mixed-precision",
    #     default=DEFAULT_MIXED_PRECISION,
    #     type=bool
    # )
    # parser.add_argument(
    #     "--save-model",
    #     default=DEFAULT_SAVE_MODEL,
    #     type=bool
    # )
    # parser.add_argument(
    #     "--save-every",
    #     default=DEFAULT_SAVE_EVERY,
    #     type=int
    # )
    parser.add_argument(
        "--channel-axis",
        default=DEFAULT_CHANNEL_AXIS,
        type=int
    )
    parser.add_argument(
        "--experiment-dir",
        default=DEFAULT_EXPERIMENT_DIR
    )
    parser.add_argument(
        "--data-dir",
        required=True
    )       
    parser.add_argument(
        "--id"
    )           
    p_args, _ = parser.parse_known_args()
    return p_args


def conv_lstm_c_gen(dir_path: str) -> Generator:
    def read_png_as_arr(filepath: str) -> np.ndarray:
        img = Image.open(filepath).convert('RGB')
        arr = np.array(img)
        return arr


    for dirpath, dirnames, filenames in os.walk(dir_path):
        if not dirnames:
            image_arrays: list = list()
            for filename in filenames:
                filepath: str = os.path.join(dirpath, filename).replace("\\", "/")
                assert os.path.exists(filepath), f"File {filepath} does not exist."
                arr = read_png_as_arr(filepath=filepath)
                image: torch.tensor = torch.as_tensor(arr.copy()).float().contiguous()
                image_arrays.append(image)

            image_arrays = torch.stack(image_arrays, 0)
            image_arrays = torch.swapaxes(image_arrays, 1, -1) # _ x W x H x C -> _ x C x H x W

            yield {
                'X': image_arrays,
                'Y': None
            }


def conv_lstm_c_save_preds(pred: torch.tensor):
    logging.info(f"Prediction: \n {pred}")


def main():
    args = vars(parse_args())
    if not args["id"]:
        experiment_id = get_random_string()
    else:
        experiment_id = args["id"]
    experiment_super_dir = args["experiment_dir"]
    experiment_dir = os.path.join(
        experiment_super_dir, experiment_id + "/"
    ).replace("\\", "/")
    log_dir = os.path.join(experiment_dir, 'logs/').replace("\\", "/")
    save_dir = os.path.join(
        experiment_dir, "model_checkpoints/"
    ).replace("\\", "/")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    time_str = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    log_filepath = os.path.join(
        log_dir, f"{SCRIPT_PATH}_{time_str}_{experiment_id}.log"
    ).replace('\\', '/')
    
    args = vars(parse_args())

    device = args["device"]
    if device == DEFAULT_DEVICE:
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device in ("CUDA", "Cuda", "cuda"):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f'Using device {device}')       

    model_name = args["model"]

    if model_name == CNNLSTM.__name__:
        sample_generator = conv_lstm_c_gen
        save_preds = conv_lstm_c_save_preds
    else:
        raise NotImplementedError

    model: torch.nn.Module = MODELS[model_name]()    

    model.to(device=device)

    model_filepath: str = args["model_filepath"]
    model.load_state_dict(torch.load(model_filepath, map_location=device))
    model_num_channels = model.args["num_channels"] # A constraint on the Model class
    channel_axis: int = args["channel_axis"]    

    logging.info('Model loaded.')

    args = get_args(
        script_path=SCRIPT_PATH, log_filepath=log_filepath, 
        **args, **model.args, experiment_id = experiment_id, time = time_str
    )

    data_dir = args["data_dir"]

    samples = sample_generator(dir_path=data_dir)

    sample_idx: int = 0
    for sample in samples:
        sample_idx += 1
        X: torch.tensor = sample["X"]
        X = X[None, :, :, :, :]
        X: torch.tensor = X.to(device=device, dtype=torch.float32)
        X_num_channels = X.shape[channel_axis]
        assert X_num_channels == model_num_channels, \
            f"Network has been defined with {model_num_channels}" \
            f"input channels, but loaded images have {X_num_channels}" \
            "channels. Please check that the images are loaded correctly."        
        logging.info(f"Generating predictions for sample {sample_idx}...")
        with torch.no_grad():
            pred = model(X)
        logging.info(f"Saving predictions for sample {sample_idx}...")
        save_preds(pred)    
    logging.info(
        """
                ================
                =              =
                =     Done.    =
                =              =
                ================
        """
    )


if __name__ == "__main__":
    main()
