import abc
import numpy as np
import torch

from scipy.stats import norm
from typing import Tuple

from src.structures.stage_data import StageData, TestData
from src.utils.misc import evaluate_mse


class _SimpleDesign(abc.ABC):
    @abc.abstractstaticmethod
    def structural_function(x: torch.Tensor) -> torch.Tensor:
        pass

    @staticmethod
    def generate_X_Z_eps(N: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = torch.distributions.MultivariateNormal(
            loc = torch.tensor([0.0, 0.0, 0.0]),
            covariance_matrix = torch.tensor([
                [1.0, 0.5, 0.0],
                [0.5, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ])
        )
        samples = dist.sample((N,))
        eps, V, W = samples[:, 0], samples[:, 1], samples[:, 2]

        X = norm.cdf((W+V)/np.sqrt(2))
        Z = norm.cdf(W)

        X, Z, eps = map(
            lambda x: torch.tensor(x).reshape(-1,1).float(),
            zip(*sorted(zip(X, Z, eps)))
        )
        return X, Z, eps
    
    @classmethod
    def generate_KIV_data(cls, no_points: int) -> StageData:
        X, Z, eps = cls.generate_X_Z_eps(no_points)
        truth = cls.structural_function(X)
        Y = truth + eps

        return StageData.from_all_data(X, Y, Z, p=0.5)
    
    @abc.abstractclassmethod
    def generate_MEKIV_data(cls, N: int) -> torch.Tensor:
        raise NotImplementedError
    
    @classmethod
    def generate_test_data(cls, no_points: int) -> TestData:
        X = torch.linspace(0, 1, no_points).reshape(-1,1)
        truth = cls.structural_function(X)
        return TestData(X, truth, evaluate_mse)    

class Linear(_SimpleDesign):
    @staticmethod
    def structural_function(x: torch.Tensor) -> torch.Tensor:
        return 4 * x - 2


class Sigmoid(_SimpleDesign):
    @staticmethod
    def structural_function(x: torch.Tensor) -> torch.Tensor:
        return torch.log(torch.abs(16*x-8)+1) * torch.sign(x-0.5)