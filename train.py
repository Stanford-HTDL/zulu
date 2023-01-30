import argparse
import logging
import os
import time

import torch

from datasets import ConvLSTMCDataset, EurosatDataset
from models import CNNLSTM, SpectrumNet, SqueezeNet
from script_utils import get_args, get_random_string

SCRIPT_PATH = os.path.basename(__file__)

DEFAULT_NUM_CHANNELS = 4
DEFAULT_NUM_CLASSES = 10
DEFAULT_DROPOUT = 0.5
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_EPOCHS = 64
DEFAULT_OPTIMIZER = "SGD"
DEFAULT_LR = 1e-3
DEFAULT_MOMENTUM = 0.9
DEFAULT_NESTEROV = True
DEFAULT_WEIGHT_DECAY = 5e-4
DEFAULT_SCHEDULER = "StepLR"
DEFAULT_STEP_SIZE = 10
DEFAULT_SCHEDULER_GAMMA = 0.75
DEFAULT_MODEL_NAME = "SqueezeNet"
DEFAULT_DATASET_NAME = "EurosatDataset"
DEFAULT_DATA_MANIFEST = "eurosat_manifest.json"
DEFAULT_VALIDATION_PERCENT = 0.15
DEFAULT_SHUFFLE = True
DEFAULT_CRITERION_NAME = "CrossEntropyLoss"
DEFAULT_EXPERIMENT_DIR = "experiments/"
DEFAULT_NUM_WORKERS = os.cpu_count()
DEFAULT_PIN_MEMORY = True
DEFAULT_DEVICE = "CUDA if available else CPU"
DEFAULT_MIXED_PRECISION = True
DEFAULT_SAVE_MODEL = True
DEFAULT_SAVE_EVERY = 8
DEFAULT_CHANNEL_AXIS = 1

DEFAULT_SEED = 8675309
torch.manual_seed(DEFAULT_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(DEFAULT_SEED)

# REMEMBER to update as new models are added!
MODELS = {
    SqueezeNet.__name__: SqueezeNet,
    SpectrumNet.__name__: SpectrumNet,
    CNNLSTM.__name__: CNNLSTM
}

DATASETS = {
    EurosatDataset.__name__: EurosatDataset,
    ConvLSTMCDataset.__name__: ConvLSTMCDataset
}

CRITERIA = {
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss
}

OPTIMIZERS = {
    "SGD": torch.optim.SGD
}
SCHEDULERS = {
    "StepLR": torch.optim.lr_scheduler.StepLR
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET_NAME
    )
    parser.add_argument(
        "--data-manifest",
        default=DEFAULT_DATA_MANIFEST
    )    
    parser.add_argument(
        "--val-percent",
        default=DEFAULT_VALIDATION_PERCENT,
        type=float
    )    
    parser.add_argument(
        "--shuffle",
        default=DEFAULT_SHUFFLE,
        type=bool
    )
    parser.add_argument(
        "--criterion",
        default=DEFAULT_CRITERION_NAME,
    )
    parser.add_argument(
        "--num-channels",
        default=DEFAULT_NUM_CHANNELS,
        type=int
    )
    parser.add_argument(
        "--num-classes",
        default=DEFAULT_NUM_CLASSES,
        type=int
    )
    parser.add_argument(
        "--dropout",
        default=DEFAULT_DROPOUT,
        type=float
    )    
    parser.add_argument(
        "--batch-size",
        default=DEFAULT_BATCH_SIZE,
        type=int
    )  
    parser.add_argument(
        "--num-epochs",
        default=DEFAULT_NUM_EPOCHS,
        type=int
    )
    parser.add_argument(
        "--optimizer",
        default=DEFAULT_OPTIMIZER,
    )    
    parser.add_argument(
        "--lr",
        default=DEFAULT_LR,
        type=float
    )
    parser.add_argument(
        "--momentum",
        default=DEFAULT_MOMENTUM,
        type=float
    )
    parser.add_argument(
        "--nesterov",
        default=DEFAULT_NESTEROV,
        type=bool
    )    
    parser.add_argument(
        "--weight-decay",
        default=DEFAULT_WEIGHT_DECAY,
        type=float
    )   
    parser.add_argument(
        "--scheduler",
        default=DEFAULT_SCHEDULER
    )
    parser.add_argument(
        "--step-size",
        default=DEFAULT_STEP_SIZE,
        type=int
    )
    parser.add_argument(
        "--scheduler-gamma",
        default=DEFAULT_SCHEDULER_GAMMA,
        type=float
    )    
    parser.add_argument(
        "--num-workers",
        default=DEFAULT_NUM_WORKERS,
        type=int
    )  
    parser.add_argument(
        "--pin-memory",
        default=DEFAULT_PIN_MEMORY,
        type=bool
    )       
    parser.add_argument(
        '--device',
        default=DEFAULT_DEVICE
    )
    parser.add_argument(
        "--mixed-precision",
        default=DEFAULT_MIXED_PRECISION,
        type=bool
    )
    parser.add_argument(
        "--save-model",
        default=DEFAULT_SAVE_MODEL,
        type=bool
    )
    parser.add_argument(
        "--save-every",
        default=DEFAULT_SAVE_EVERY,
        type=int
    )
    parser.add_argument(
        "--channel-axis",
        default=DEFAULT_CHANNEL_AXIS,
        type=int
    )         
    p_args, _ = parser.parse_known_args()
    return p_args    


def main():
    experiment_id = get_random_string()
    experiment_dir = os.path.join(
        DEFAULT_EXPERIMENT_DIR, experiment_id + "/"
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

    args = get_args(
        script_path=SCRIPT_PATH, log_filepath=log_filepath, **args, 
        experiment_id = experiment_id, time = time_str
    )

    num_channels = args["num_channels"]
    num_classes = args["num_classes"]
    dropout = args["dropout"]

    num_epochs = args["num_epochs"]
    batch_size = args["batch_size"]

    device = args["device"]
    if device == DEFAULT_DEVICE:
        device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif device in ("CUDA", "Cuda", "cuda"):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f'Using device {device}')    

    model_name = args["model"]
    model = MODELS[model_name](
        num_channels=num_channels, num_classes=num_classes, dropout=dropout
    )
    model = model.to(device=device)

    data_manifest = args["data_manifest"]
    dataset_name = args["dataset"]
    dataset = DATASETS[dataset_name](data_manifest)

    val_percent = args["val_percent"]
    num_validation = int(len(dataset) * val_percent)
    num_train = len(dataset) - num_validation

    logging.info(
        f"""
                Number of training samples:   {num_train}
                Number of validation samples: {num_validation}
        """
    )
    train_set, validation_set = torch.utils.data.random_split(
            dataset, [num_train, num_validation], 
            generator=torch.Generator().manual_seed(DEFAULT_SEED)
    )

    shuffle = args["shuffle"]
    num_workers = args["num_workers"]
    pin_memory = args["pin_memory"]
    train_loader = torch.utils.data.DataLoader(
        train_set, shuffle=shuffle, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=pin_memory,
    )
    validation__loader = torch.utils.data.DataLoader(
        validation_set, shuffle=shuffle, batch_size=batch_size, 
        num_workers=num_workers, pin_memory=pin_memory
    )    

    optimizer_name = args["optimizer"]
    Optimizer = OPTIMIZERS[optimizer_name]
    if optimizer_name == "SGD":
        nesterov = args["nesterov"]
        lr = args["lr"]
        momentum = args["momentum"]
        weight_decay = args["weight_decay"]        
        optimizer = Optimizer(
            model.parameters(), lr=lr, momentum=momentum, 
            weight_decay=weight_decay, nesterov=nesterov          
        )
    else:
        raise NotImplementedError(f"Optimizer {optimizer_name} not known.")

    scheduler_name = args["scheduler"]
    Scheduler = SCHEDULERS[scheduler_name]
    if scheduler_name == "StepLR":
        step_size = args["step_size"]
        scheduler_gamma = args["scheduler_gamma"]
        scheduler = Scheduler(
            optimizer, step_size=step_size, gamma=scheduler_gamma
        )

    use_mp = args["mixed_precision"]

    criterion_name = args["criterion"]
    criterion = CRITERIA[criterion_name]().to(device=device)

    save_model = args["save_model"]
    save_every = args["save_every"]

    channel_axis = args["channel_axis"]

    ### Train Loop Begins ###
    logging.info("Starting training...")
    for epoch in range(1, num_epochs + 1):
        logging.info(f"Starting epoch {epoch}...")
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            X, Y = batch["X"], batch["Y"] # A constraint on the Dataset class
            X_num_channels = X.shape[channel_axis]
            assert X_num_channels == model.num_channels, \
                f"Network has been defined with {model.num_channels}" \
                f"input channels, but loaded images have {X_num_channels}" \
                "channels. Please check that the images are loaded correctly."
            
            X = X.to(device=device, dtype=torch.float32) # A constraint on the Dataset class
            Y = Y.to(device=device, dtype=torch.long) # A constraint on the Dataset class
            optimizer.zero_grad()
            with torch.autocast(
                device.type if device.type != "mps" else "cpu", enabled=use_mp 
            ):
                Y_hat = model(X)
                loss = criterion(Y_hat, Y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        scheduler.step()

        logging.info(
            f"""
                    Epoch {epoch} training completed.
                    Train loss: {train_loss:.5f}.\
        
                    Starting validation...
            """
        )
        model.eval()
        validation_loss = 0.0
        for batch in validation__loader:
            X, Y = batch["X"], batch["Y"] # A constraint on the Dataset class
            X_num_channels = X.shape[channel_axis]
            assert X_num_channels == model.num_channels, \
                f"Network has been defined with {model.num_channels}" \
                f"input channels, but loaded images have {X_num_channels}" \
                "channels. Please check that the images are loaded correctly."
            
            X = X.to(device=device, dtype=torch.float32) # A constraint on the Dataset class
            Y = Y.to(device=device, dtype=torch.long) # A constraint on the Dataset class
            with torch.autocast(
                device.type if device.type != "mps" else "cpu", enabled=use_mp 
            ):
                Y_hat = model(X)
                loss = criterion(Y_hat, Y)
            validation_loss += loss.item()

        logging.info(
            f"""
                    Epoch {epoch} validation completed.
                    Validation loss: {validation_loss:.5f}.
        
                    Done with epoch {epoch}.
            """
        )

        if (save_model and epoch % save_every == 0) or epoch == num_epochs:
            state_dict = model.state_dict()
            savepath = os.path.join(save_dir, f"checkpoint_epoch_{epoch:04}.pth")
            torch.save(state_dict, savepath)
            logging.info(f"Checkpoint {epoch} saved.")
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
