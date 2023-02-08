__author__ = "Richard Correro (richard@richardcorrero.com)"


import argparse

import torch

from script_utils import parse_args, arg_is_true


class SGD(torch.optim.SGD):
    __name__: str = "SGD"

    DEFAULT_LR = 1e-3
    DEFAULT_MOMENTUM = 0.9
    DEFAULT_NESTEROV = True
    DEFAULT_WEIGHT_DECAY = 5e-4    


    def __init__(self, parameters):
        args = self.parse_args()
        lr = args["lr"]
        momentum = args["momentum"]
        nesterov = arg_is_true(args["nesterov"])
        weight_decay = args["weight_decay"]           
        self.args = args
        super().__init__(
            params=parameters, lr=lr, momentum=momentum, 
            weight_decay=weight_decay, nesterov=nesterov
        )


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--lr",
            default=self.DEFAULT_LR,
            type=float
        )
        parser.add_argument(
            "--momentum",
            default=self.DEFAULT_MOMENTUM,
            type=float
        )
        parser.add_argument(
            "--nesterov",
            default=self.DEFAULT_NESTEROV
        )    
        parser.add_argument(
            "--weight-decay",
            default=self.DEFAULT_WEIGHT_DECAY,
            type=float
        )   
        args = parse_args(parser=parser)
        return args


class Adam(torch.optim.Adam):
    __name__: str = "Adam"

    DEFAULT_LR = 1e-3
    DEFAULT_BETAS = (0.9, 0.999)
    DEFAULT_EPS = 1e-8
    DEFAULT_WEIGHT_DECAY = 0
    DEFAULT_AMSGRAD = False


    def __init__(self, parameters):
        args = self.parse_args()
        lr = args["lr"]
        betas = tuple(args["betas"])
        assert len(betas) == 2, \
            f"Passed {len(betas)} arguments for `betas`, but only two needed."
        eps = args["eps"]
        weight_decay = args["weight_decay"]
        amsgrad = arg_is_true(args["amsgrad"])
        self.args = args
        super().__init__(
            params=parameters, lr=lr, betas=betas, eps=eps, 
            weight_decay=weight_decay, amsgrad=amsgrad
        )


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--lr",
            default=self.DEFAULT_LR,
            type=float
        )
        parser.add_argument(
            "--betas",
            default=self.DEFAULT_BETAS,
            nargs="+",
            type=float
        )
        parser.add_argument(
            "--eps",
            default=self.DEFAULT_EPS,
            type=float
        )    
        parser.add_argument(
            "--weight-decay",
            default=self.DEFAULT_WEIGHT_DECAY,
            type=float
        )
        parser.add_argument(
            "--amsgrad",
            default=self.DEFAULT_AMSGRAD
        )
        args = parse_args(parser=parser)
        return args            
