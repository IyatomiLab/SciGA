import random
import numpy as np
import torch
from tabulate import tabulate


def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across various libraries.

    Args:
        seed (int): The seed value to set for random number generation.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def print_epoch_scores(
    phases: list[str],
    losses: list[float],
    scores: list[dict[str, float | tuple[float, float]]],
) -> None:
    """
    Print the scores of each phase in a formatted table.

    Args:
        phases (List[str]): List of phase names (e.g., 'train', 'valid').
        losses (List[float]): List of loss values for each phase.
        scores (List[Dict[str, float]]): List of score dictionaries for each phase.
            - float: Displayed with 4 decimal places (e.g., '0.1234').
            - str: Displayed as-is.
            - Tuple[float, float]: Interpreted as (mean, std) and displayed as 'mean±std' (e.g., '0.8123±0.0345').
    """
    def _format_value(v: float | tuple[float, float]) -> str:
        if isinstance(v, float):
            return f'{v:.4f}'
        elif isinstance(v, tuple) and len(v) == 2 and all(isinstance(x, float) for x in v):
            return f'{v[0]:.4f}±{v[1]:.4f}'
        return str(v)

    def _format_row(loss: float, score: dict[str, float | tuple[float, float]], metric_keys: list[str]) -> list[str]:
        return [_format_value(loss)] + [_format_value(score[k]) for k in metric_keys]

    metric_keys = list(scores[0].keys())
    headers = ['', 'Loss'] + metric_keys
    records = [[phase] + _format_row(loss, score, metric_keys) for phase, loss, score in zip(phases, losses, scores)]
    table = tabulate(records, headers, tablefmt='fancy_grid')

    print('\n'.join('   ' + line for line in table.splitlines()))
