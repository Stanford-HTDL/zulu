import argparse

import torch

from script_utils import parse_args


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
        nesterov = args["nesterov"]        
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
            default=self.DEFAULT_NESTEROV,
            type=bool
        )    
        parser.add_argument(
            "--weight-decay",
            default=self.DEFAULT_WEIGHT_DECAY,
            type=float
        )   
        args = parse_args(parser=parser)
        return args        
