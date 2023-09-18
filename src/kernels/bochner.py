import torch
import torch.nn as nn
from typing import Optional
from src.kernels.gaussian import MultiDimGaussianKernel

class BochnerKernel(nn.Module):
    def __init__(
        self,
        dimension: int,
        lengthscales: torch.Tensor,
        network: Optional[nn.Module] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__()
        self.dimension = dimension
        self.device = device
        
        if network:
            self.net = network
        else:
            self.net = nn.Linear(
                in_features = self.dimension,
                out_features = self.dimension,
                bias = True,
            )
        self.net.to(self.device)

        self.eps = 0.1
        self.gaussian = MultiDimGaussianKernel(lengthscales=lengthscales.to(self.device))

    def __call__(self, *args, **kwargs):
        return self.evaluate_analytically(*args, **kwargs)

    def evaluate_analytically(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        if type(self.net) is not nn.Linear:
            raise ValueError('Network not linear, unknown how to evaluate analytically')
        X, Y = X.to(self.device), Y.to(self.device)

        differences = X.unsqueeze(1) - Y.unsqueeze(0) # Shape (x.shape[0], y.shape[0], x.shape[1])
        cos_terms = torch.cos(
            torch.einsum(
                'xyj,j->xy',
                differences,
                self.net.bias
            )
        )
        exp_terms = torch.exp(
            -torch.norm(
                torch.einsum(
                    'ij,xyj->xyi',
                    self.net.weight.T,
                    differences
                ),
                dim=-1
            ) ** 2 / 2
        )
        adaptive = torch.einsum(
            'xy,xy->xy',
            cos_terms,
            exp_terms
        )
        gaussian = self.gaussian(X, Y)
        return (1-self.eps) * adaptive + self.eps * gaussian
    
    def sample(self, no_samples: int) -> torch.Tensor:
        no_gaussian_samples = int(self.eps * no_samples)
        no_bochner_samples = no_samples - no_gaussian_samples

        gaussian_samples = self.gaussian.sample_from_bochner(no_gaussian_samples).to(self.device)
        dist = torch.distributions.MultivariateNormal(
            loc = torch.zeros(
                self.dimension,
                device=self.device
            ),
            covariance_matrix = torch.eye(
                self.dimension,
                device=self.device
            )
        )
        samples = dist.sample((no_bochner_samples,))
        bochner_samples = self.net(samples)
        return torch.vstack((bochner_samples, gaussian_samples))
    
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
    
    def evaluate_precisely_no_grad(self, X1: torch.Tensor, X2: torch.Tensor, no_batch_samples: int, no_batches: int) -> torch.Tensor:
        K_X1X2 = torch.zeros((X1.shape[0], X2.shape[0]), device=X1.device)
        with torch.no_grad():
            for i in range(no_batches):
                samples = self.sample(no_batch_samples).detach()
                k_x1x2 = BochnerKernel.evaluate_from_samples(X1, X2, samples)
                K_X1X2 += k_x1x2
                torch.cuda.empty_cache()
        return K_X1X2 / no_batches
            
    @classmethod
    def from_gaussian_kernel(
        cls,
        dimension: int,
        lengthscale: float,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> 'BochnerKernel':
        """ Returns a Bochner kernel which emulates a Gaussian kernel with the given lengthscale. """
        kernel = cls(dimension, device)
        with torch.no_grad():
            kernel.linear.weight = nn.Parameter(
                torch.eye(dimension, device=kernel.device) / lengthscale,
            )
            kernel.linear.bias = nn.Parameter(
                torch.zeros(dimension, device=kernel.device)
            )
        return kernel
    
    @classmethod
    def from_multidim_gaussian_kernel(
        cls,
        dimension: int,
        lengthscales: torch.Tensor,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> 'BochnerKernel':
        kernel = cls(dimension, device)
        with torch.no_grad():
            kernel.linear.weight = nn.Parameter(
                torch.diag(1 / lengthscales).to(device=kernel.device)
            )
            kernel.linear.bias = nn.Parameter(
                torch.zeros(dimension, device=kernel.device)
            )
        return kernel
