import torch

from dataclasses import dataclass
from typing import Tuple

from zmq import device


@dataclass
class GaussianKernel:
    lengthscale: float | torch.Tensor

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        distances = torch.cdist(X, Y, p=2)
        return torch.exp(-distances**2 / (2*self.lengthscale**2))
    
@dataclass
class MultiDimGaussianKernel:
    lengthscales: Tuple[float] | torch.Tensor

    def __post_init__(self):
        self.dim = len(self.lengthscales)

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        assert X.shape[1] == Y.shape[1] == len(self.lengthscales)
        componentwise_distances = (X.unsqueeze(1) - Y.unsqueeze(0)) ** 2  # Shape (N_X, N_Y, DIM)
        componentwise_distances /= 2 * self.lengthscales ** 2
        return torch.exp(-componentwise_distances.sum(dim=2))
    
    def sample_from_bochner(self, no_samples) -> torch.Tensor:
        # Sample from the Fourier transform of the kernel
        dist = torch.distributions.MultivariateNormal(
            loc=torch.zeros(len(self.lengthscales), device=self.lengthscales.device),
            covariance_matrix=torch.diag(1/self.lengthscales**2).to(self.lengthscales.device)
        )
        samples = dist.sample((no_samples,))
        return samples
    
    @staticmethod
    def evaluate_from_samples(X1: torch.Tensor, X2: torch.Tensor, samples: torch.Tensor) -> torch.Tensor:
        differences = (X1.unsqueeze(1) - X2.unsqueeze(0)).to(samples.device)  # Shape (X1.shape[0], X2.shape[0], X.shape[1])
        return torch.mean(
            torch.cos(
                torch.einsum(
                    'xyi,si->xys',
                    differences,
                    samples
                )
            ),
            dim=-1
        )