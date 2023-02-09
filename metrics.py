__author__ = "Richard Correro (richard@richardcorrero.com)"


from typing import Dict, Optional

import torch
from collections import namedtuple


# class Metrics(
#     namedtuple(
#         "Metrics", ["tp", "fp", "tn", "fn", "precision", "recall", "accuracy", "F"]
#         )
#     ):
#     def __new__(cls, tp, fp, tn, fn, precision, recall, accuracy, F):
#         return tuple.__new__(cls, [tp, fp, tn, fn, precision, recall, accuracy, F])


def calc_metrics(
    Y: torch.Tensor, Y_hat: torch.Tensor, beta: Optional[int] = 1,
    eps: Optional[float] = 1e-16
) -> Dict:
    assert Y.ndim == 1 and Y_hat.ndim == 2, \
        f"Invalid number of dimensions for Y: {Y.ndim} and Y_hat: {Y_hat.ndim}."

    Y_pred: torch.Tensor = torch.argmax(Y_hat, dim=1)

    tp: int = int(Y_pred @ Y)
    fp: int = int(Y_pred @ (1 - Y))
    tn: int = int((1 - Y_pred) @ (1 - Y))
    fn: int = int((1 - Y_pred) @ Y)

    precision: float = (tp + eps) / (tp + fp + eps)
    recall: float = (tp + eps) / (tp + fn + eps)
    accuracy: float = (tp + tn) / (tp + tn + fp + fn)
    
    beta_sq = beta ** 2

    F_beta = (1 + beta_sq) * ((precision * recall + eps) / (beta_sq * precision + recall + eps))

    metrics: Dict = {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        f"F_{beta}": F_beta
    }
    # metrics = Metrics(tp, fp, tn, fn, precision, recall, accuracy, F_beta)
    return metrics
