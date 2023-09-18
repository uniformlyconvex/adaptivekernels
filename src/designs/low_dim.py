import abc
import numpy as np
import torch

from scipy.stats import norm
from typing import Tuple

from src.structures.stage_data import StageData, TestData
from src.utils.misc import evaluate_mse

class _MultiDimDesign(abc.ABC):
    @abc.abstractstaticmethod
    def structural_function(x: torch.Tensor) -> torch.Tensor:
        pass

    @classmethod
    def generate_X_Z_eps(cls, no_points: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dist = torch.distributions.MultivariateNormal(
            loc = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0]),
            covariance_matrix = torch.tensor([
                [1.0, 0.5, 0.5, 0.0, 0.0],
                [0.5, 1.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0]
            ])
        )
        samples = dist.sample((no_points,))
        eps = samples[:, 0].reshape(no_points, 1)

        V = samples[:,1:3]
        W = samples[:,3:5]

        X = 10 * norm.cdf((V+W)/np.sqrt(2))
        Z = 10 * norm.cdf(W)

        X, Z = map(torch.tensor, (X, Z))

        return X, Z, eps
    
    @classmethod
    def generate_X_Y_Z(cls, no_points: int) -> Tuple[torch.Tensor]:
        X, Z, eps = cls.generate_X_Z_eps(no_points)
        Y = cls.structural_function(X) + eps

        X, Y, Z = map(lambda x: x.float(), (X, Y, Z))
        return X, Y, Z

    @classmethod
    def generate_KIV_data(cls, no_points: int) -> StageData:
        X, Y, Z = cls.generate_X_Y_Z(no_points)
        return StageData.from_all_data(X, Y, Z, p=0.5)
    
    @classmethod
    def generate_MEKIV_data(cls, no_points: int, merror_type: str, merror_scale: float) -> Tuple[torch.Tensor]:
        X, Y, Z = cls.generate_X_Y_Z(no_points)

        X_std = X.std(dim=0)
        dim = X.shape[1]
        error_dist = torch.distributions.MultivariateNormal(
            loc = torch.zeros(dim),
            covariance_matrix = torch.diag(X_std * merror_scale)**2
        )
        
        M: torch.Tensor = X + error_dist.sample((X.shape[0],))
        N: torch.Tensor = X + error_dist.sample((X.shape[0],))

        return X, M, N, Y, Z
    
    @classmethod
    def generate_test_data(cls, no_points: int) -> TestData:
        X = cls.generate_KIV_data(no_points).all_X
        truth = cls.structural_function(X)
        return TestData(X, truth, evaluate_mse)
    

class SineDesign(_MultiDimDesign):
    @staticmethod
    def structural_function(x: torch.Tensor) -> torch.Tensor:
        return 2*(torch.sin(x[:,0]) + torch.cos(x[:,1] + 0.5*x[:,0])).reshape(-1,1)


