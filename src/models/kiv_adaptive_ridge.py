import torch

from dataclasses import dataclass

import src.utils.misc as utils

from src.kernels.gaussian import MultiDimGaussianKernel
from src.structures.stage_data import StageData

@dataclass
class KIVAdaptiveRidge:
    data: StageData
    lmbda: torch.Tensor = None
    xi: torch.Tensor = None
    kernel = None

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

        self._is_trained: bool = False

    def stage_1_tuning(self, search_space: torch.Tensor) -> torch.Tensor:
        def KIV1_loss(lmbda: float) -> float:
            """ Compute OOS loss for a given lambda """
            lmbda = torch.tensor(lmbda).float()
            gamma = torch.linalg.solve(
                self.K_Z1Z1 + len(self.data.stage_1) * lmbda * torch.eye(len(self.data.stage_1)),
                self.K_Z1Z2
            )

            loss = torch.trace(
                self.K_X2X2 - 2 * self.K_X2X1 @ gamma + gamma.T @ self.K_X1X1 @ gamma
            ) / len(self.data.stage_2)

            return loss.float().item()
        
        lmbda, _, _ = utils.minimize(KIV1_loss, search_space)
        return lmbda
    
    def stage_2_tuning(self, lmbda: torch.Tensor, search_space: torch.Tensor) -> torch.Tensor:
        W = self.K_X1X1 @ torch.linalg.solve(
            self.K_Z1Z1 + len(self.data.stage_1) * lmbda * torch.eye(len(self.data.stage_1)),
            self.K_Z1Z2
        )
        def KIV2_loss(xi: float) -> float:
            xi = torch.tensor(xi).float()
            alpha = torch.linalg.solve(
                W @ W.T + len(self.data.stage_2) * xi * self.K_X1X1,
                W @ self.data.stage_2.Y
            )
            preds = self.K_X1X1.T @ alpha  # shape (no_test, dim)
            distances = torch.norm(preds - self.data.stage_1.Y, dim=1)
            loss = torch.mean(distances ** 2).float().item()

            return loss
        
        xi, _, _ = utils.minimize(KIV2_loss, search_space)
        return xi
    
    def train(self):
        if len(self.lmbda) != 1:
            self.lmbda = self.stage_1_tuning(self.lmbda)
        if len(self.xi) != 1:
            self.xi = self.stage_2_tuning(self.lmbda, self.xi)
        
        self._is_trained = True

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            self.train()

        K_X1Xtest = self.X_kernel(self.data.stage_1.X, x)

        W = self.K_X1X1 @ torch.linalg.solve(
            self.K_Z1Z1 + len(self.data.stage_1) * self.lmbda * torch.eye(len(self.data.stage_1)),
            self.K_Z1Z2
        )
        alpha = torch.linalg.solve(
            W @ W.T + len(self.data.stage_2) * self.xi * self.K_X1X1,
            W @ self.data.stage_2.Y
        )

        preds = K_X1Xtest.T @ alpha
        return preds

