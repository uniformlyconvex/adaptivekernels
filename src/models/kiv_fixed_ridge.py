import torch

from dataclasses import dataclass

import src.utils.misc as utils

from src.kernels.gaussian import MultiDimGaussianKernel
from src.structures.stage_data import StageData

@dataclass
class KIVFixedRidge:
    data: StageData
    lmbda: float | torch.Tensor
    xi: float | torch.Tensor

    def __post_init__(self):
        # Median heuristic for lengthscale
        self.X_kernel = MultiDimGaussianKernel(
            lengthscales = utils.auto_lengthscales(self.data.all_X)
        )
        self.Z_kernel = MultiDimGaussianKernel(
            lengthscales = utils.auto_lengthscales(self.data.all_Z)
        )

        # Precompute kernels
        self.K_X1X1 = self.X_kernel(self.data.stage_1.X, self.data.stage_1.X)
        self.K_X2X1 = self.X_kernel(self.data.stage_2.X, self.data.stage_1.X)
        self.K_X2X2 = self.X_kernel(self.data.stage_2.X, self.data.stage_2.X)

        self.K_Z1Z1 = self.Z_kernel(self.data.stage_1.Z, self.data.stage_1.Z)
        self.K_Z1Z2 = self.Z_kernel(self.data.stage_1.Z, self.data.stage_2.Z)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        K_X1Xtest = self.X_kernel(self.data.stage_1.X, x)

        W = self.K_X1X1 @ torch.linalg.solve(
            self.K_Z1Z1 + len(self.data.stage_1) * self.lmbda * torch.eye(len(self.data.stage_1)),
            self.K_Z1Z2
        )
        alpha = torch.linalg.solve(
            W @ W.T + len(self.data.stage_2) * self.xi * self.K_X1X1,
            (W @ self.data.stage_2.Y)
        )

        preds = K_X1Xtest.T @ alpha
        return preds