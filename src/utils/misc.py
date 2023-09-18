import numpy as np
import time
import torch

from numpy.typing import ArrayLike
from tqdm import tqdm
from typing import Optional, Tuple, Iterable, Callable

def rand_split(arrays: Iterable[ArrayLike], p: float=0.5) -> Tuple[Tuple[ArrayLike], Tuple[ArrayLike]]:
    length = len(arrays[0])
    if not all(len(array) == length for array in arrays):
        raise ValueError("All arrays must have the same length")
    
    rng = np.random.default_rng()
    indices = np.arange(start=0, stop=length)
    rng.shuffle(indices)

    split_index = int(length * p)

    return tuple(array[indices[:split_index]] for array in arrays), tuple(array[indices[split_index:]] for array in arrays)

def interpoint_distances(X1: ArrayLike, X2: Optional[ArrayLike]=None) -> torch.Tensor:
    if X2 is None:
        X2 = X1

    return torch.cdist(X1, X2, p=2)

def median_interpoint_distances(X1: ArrayLike, X2: Optional[ArrayLike]=None) -> float:
    distances = interpoint_distances(X1, X2)
    return torch.median(distances).item()

def auto_lengthscales(X: ArrayLike) -> torch.Tensor:
    """ Compute median interpoint distances for each dimension of X """
    return torch.tensor([median_interpoint_distances(X[:, i].reshape(-1,1)) for i in range(X.shape[1])])

def minimize(func: Callable, test_points: ArrayLike) -> Tuple[torch.Tensor, torch.Tensor]:
    tic = time.time()
    if isinstance(test_points, torch.Tensor):
        test_points = test_points.numpy()

    fs = []

    if test_points.shape == ():
        test_points = np.array([test_points])
    for x in tqdm(test_points):
        fs.append(func(x))

    opt_idx = np.nanargmin(fs)
    x_star = test_points[opt_idx]
    f_star = fs[opt_idx]

    toc = time.time()
    print(f'Minimization took {toc - tic:.2f} seconds')

    return torch.tensor(x_star), torch.tensor(f_star), fs

def evaluate_mse(predictions: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return (predictions - truth).norm(dim=1).square().mean()

def evaluate_log10_mse(predictions: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    mse = evaluate_mse(predictions, truth)
    return mse.log10()