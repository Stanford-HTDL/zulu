__author__ = "Richard Correro (richard@richardcorrero.com)"


import argparse

import torch

from script_utils import arg_is_true, parse_args


class StepLR(torch.optim.lr_scheduler.StepLR):
    __name__ = "StepLR"

    requires_metrics = False

    DEFAULT_STEP_SIZE = 16
    DEFAULT_GAMMA = 0.9


    def __init__(self, optimizer: torch.optim.Optimizer):
        args = self.parse_args()
        step_size: int = args["step_size"]
        gamma: float = args["gamma"]
        self.args = args
        super().__init__(
            optimizer=optimizer, step_size=step_size, 
            gamma=gamma
        )


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--step-size",
            default=self.DEFAULT_STEP_SIZE,
            type=int
        )
        parser.add_argument(
            "--gamma",
            default=self.DEFAULT_GAMMA,
            type=float
        )    
        args = parse_args(parser=parser)
        return args


class ReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    __name__ = "ReduceLROnPlateau"

    requires_metrics = True

    DEFAULT_MODE = "min"
    DEFAULT_FACTOR = 0.1
    DEFAULT_PATIENCE = 10
    DEFAULT_THRESHOLD = 0.0001
    DEFAULT_THRESHOLD_MODE = "rel"
    DEFAULT_COOLDOWN = 0
    DEFAULT_MIN_LR = 0
    DEFAULT_EPS = 1e-08
    DEFAULT_VERBOSE = False


    def __init__(self, optimizer: torch.optim.Optimizer):
        args = self.parse_args()
        mode: str = args["plateau_mode"]
        factor: float = args["factor"]
        patience: float = args["patience"]
        threshold: float = args["threshold"]
        threshold_mode: str = args["threshold_mode"]
        cooldown: float = args["cooldown"]
        min_lr: float = args["min_lr"]
        eps: float = args["plateau_eps"]
        verbose: bool = arg_is_true(args["plateau_verbose"])
        self.args = args
        super().__init__(
            optimizer=optimizer, mode=mode, factor=factor, patience=patience,
            verbose=verbose, threshold=threshold, threshold_mode=threshold_mode,
            cooldown=cooldown, min_lr=min_lr, eps=eps
        )


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--plateau-mode",
            default=self.DEFAULT_MODE
        )
        parser.add_argument(
            "--factor",
            default=self.DEFAULT_FACTOR,
            type=float
        )
        parser.add_argument(
            "--patience",
            default=self.DEFAULT_PATIENCE,
            type=float
        )
        parser.add_argument(
            "--threshold",
            default=self.DEFAULT_THRESHOLD,
            type=float
        )
        parser.add_argument(
            "--threshold-mode",
            default=self.DEFAULT_THRESHOLD_MODE
        )
        parser.add_argument(
            "--cooldown",
            default=self.DEFAULT_COOLDOWN,
            type=float
        )
        parser.add_argument(
            "--min-lr",
            default=self.DEFAULT_MIN_LR,
            type=float
        )
        parser.add_argument(
            "--plateau-eps",
            default=self.DEFAULT_EPS,
            type=float
        )
        parser.add_argument(
            "--plateau-verbose",
            default=self.DEFAULT_VERBOSE
        )
        args = parse_args(parser=parser)
        return args         
