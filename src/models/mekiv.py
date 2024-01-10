import torch

from dataclasses import dataclass
from typing import Optional, Iterable, Tuple
from tqdm import tqdm
from src.structures.stage_data import Stage1Data, Stage2Data, StageData

import src.utils.misc as misc
from src.kernels import gaussian
from src.models.kiv_adaptive_ridge import KIVAdaptiveRidge


class XModel(torch.nn.Module):
    def __init__(
        self,
        M: torch.Tensor,
        N: torch.Tensor,
        lambda_N: float,
        K_Z1Z1: torch.Tensor,
        K_Z1Z2: torch.Tensor,
        gamma_MN: torch.Tensor,
        gamma_N: torch.Tensor,
        alpha_samples: torch.Tensor,
        true_X: torch.Tensor = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()

        self.device = device
        self.K_Z1Z1 = K_Z1Z1.to(self.device)
        self.K_Z1Z2 = K_Z1Z2.to(self.device)

        self.n = K_Z1Z1.shape[0]

        lambda_N = torch.tensor([lambda_N]).to(self.device)

        self.init_X = ((M.clone() + N.clone()) / 2).to(self.device)

        # Initialise X and lambda_X
        self.X = torch.nn.Parameter(self.init_X)
        self.log_lambda_X = torch.nn.Parameter(torch.log(lambda_N))

        # get the MN labels
        self.MN_labels = self.compute_labels(
            alpha_samples=alpha_samples.to(self.device),
            exponent=N.to(self.device),
            gamma_numerator=gamma_MN.to(self.device),
            gamma_denominator=gamma_N.to(self.device),
            multiplier_numerator=M.to(self.device),
        )

        self.M = M.to(self.device)
        self.N = N.to(self.device)

        self.alpha_samples = alpha_samples.to(self.device)

        self.good_indices = self.compute_good_indices(self.MN_labels)

        self.true_X = true_X.to(self.device) if true_X is not None else None

        self.losses = []
        self.distances = []

    def compute_good_indices(self, labels: torch.Tensor) -> torch.Tensor:
        no_stds = 1

        label_real = torch.real(labels)
        label_imag = torch.imag(labels)

        mu_real, mu_imag = map(torch.mean, (label_real, label_imag))
        std_real, std_imag = map(torch.std, (label_real, label_imag))

        indices = (torch.abs(label_real - mu_real) <= no_stds * std_real).logical_and(
            torch.abs(label_imag - mu_imag) <= no_stds * std_imag
        )

        return indices

    def forward(self):
        gamma_X = torch.linalg.solve(
            self.K_Z1Z1
            + self.n
            * torch.exp(self.log_lambda_X)
            * torch.eye(self.n, device=self.device),
            self.K_Z1Z2,
        )

        X_labels = self.compute_labels(
            alpha_samples=self.alpha_samples,
            exponent=self.X,
            gamma_numerator=gamma_X,
            gamma_denominator=gamma_X,
            multiplier_numerator=self.X,  # should be x
        )

        return X_labels

    @staticmethod
    def compute_labels(
        alpha_samples: torch.Tensor,  # Shape (C, dim)
        exponent: torch.Tensor,  # Shape (n, dim)
        gamma_numerator: torch.Tensor,  # Shape (n, m)
        gamma_denominator: torch.Tensor,  # Shape (n, m)
        multiplier_numerator: torch.Tensor,  # Shape (n, dim)
    ) -> torch.Tensor:  # Shape (C, m, dim)
        """
        alpha_samples: shape (a, dim)
        exponent: shape (n, dim)
        gamma_numerator: shape (n, m)
        gamma_denominator: shape (n, m)
        multiplier_numerator: shape (n, dim)

        """
        # Shape (C, n); the (a, j)th entry is exp(i alpha_a . n_j)
        exps = torch.exp(1j * (alpha_samples @ exponent.T).type(torch.complex64))
        n = exponent.shape[0]

        # slow_exps = torch.zeros_like(exps)
        # for a in range(alpha_samples.shape[0]):
        #     for j in range(exponent.shape[0]):
        #         slow_exps[a,j] = torch.exp(1j * (alpha_samples[a] @ exponent[j]))
        # assert torch.allclose(exps, slow_exps)

        # Shape (C, m); the (a,j)th entry is gamma_N(z_j) . exp(i alpha_a . n_j)
        # (C, n), (n, m) -> (C, m)
        denominator = exps @ gamma_denominator.type(torch.complex64)

        numerator = torch.einsum(
            "jd,jz,aj -> azd",
            multiplier_numerator.type(torch.complex64),
            gamma_numerator.type(torch.complex64),
            exps,
        )

        return numerator / denominator.unsqueeze(-1)

    def loss(self, good_indices_only: bool = False):
        preds = self.forward()
        truth = self.MN_labels

        if good_indices_only:
            preds = preds * self.good_indices
            truth = truth * self.good_indices

        mse = torch.mean(torch.norm(preds - truth, dim=-1) ** 2)

        msd = torch.mean(torch.norm(self.X - (self.M + self.N) / 2, dim=-1) ** 2)
        # reg = 1000*msd

        return mse

    def fit(self, no_epochs=1000, good_indices_only: bool = True):
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-2)

        mean_sq_dist = lambda T: torch.norm(self.X - T, dim=-1).square().mean().item()

        for epoch in tqdm(range(no_epochs)):
            if epoch == 0:
                print(f"Before training:")
                ms_X = mean_sq_dist(self.true_X)
                ms_M = mean_sq_dist(self.M)
                ms_N = mean_sq_dist(self.N)
                ms_MN = mean_sq_dist((self.M + self.N) / 2)

                print(
                    f"    Mean square distance to | True X: {ms_X:.4f} | M: {ms_M:.4f} | N: {ms_N:.4f} | MN: {ms_MN:.4f}"
                )

            optimizer.zero_grad()
            loss = self.loss(good_indices_only)
            loss.backward()
            optimizer.step()

            if epoch % (no_epochs / 100) == 0:
                if self.true_X is not None:
                    ms_X = mean_sq_dist(self.true_X)
                    ms_M = mean_sq_dist(self.M)
                    ms_N = mean_sq_dist(self.N)
                    ms_MN = mean_sq_dist((self.M + self.N) / 2)

                    print(
                        f"Epoch {epoch} | Loss: {loss.item():.4f} | Good indices: {good_indices_only}"
                    )
                    print(
                        f"    Mean square distance to | True X: {ms_X:.4f} | M: {ms_M:.4f} | N: {ms_N:.4f} | MN: {ms_MN:.4f}"
                    )
                else:
                    print(f"Epoch {epoch} | LOSS: {loss.item()}")

            self.losses.append(loss.item())
            self.distances.append(mean_sq_dist(self.true_X))


@dataclass
class MEKIV:
    M: torch.Tensor
    N: torch.Tensor
    Y: torch.Tensor
    Z: torch.Tensor

    lmbda_search_space: torch.Tensor
    xi_search_space: torch.Tensor

    no_epochs: int = 1000
    real_X: Optional[torch.Tensor] = None

    def __post_init__(self):
        self._is_trained = False

        self.MN = torch.hstack((self.M, self.N))

        if self.real_X is not None:
            first, second = misc.rand_split(
                (self.M, self.N, self.Y, self.Z, self.real_X), p=0.50
            )
            self.M1, self.N1, self.Y1, self.Z1, self.real_X1 = first
            self.M2, self.N2, self.Y2, self.Z2, self.real_X2 = second
        else:
            first, second = misc.rand_split((self.M, self.N, self.Y, self.Z), p=0.5)
            self.M1, self.N1, self.Y1, self.Z1 = first
            self.M2, self.N2, self.Y2, self.Z2 = second
            self.real_X1, self.real_X2 = None, None

        guess = (self.M1 + self.N1) / 2
        if self.real_X is not None:
            dist = torch.norm(self.real_X1 - guess, dim=-1).pow(2).mean().item()
            print(f"Mean squared distance from (M+N)/2 to X: {dist:.4f}")

        self.MN1 = torch.hstack((self.M1, self.N1))
        self.MN2 = torch.hstack((self.M2, self.N2))

        # Median heuristic for lengthscale, computing from all points
        self.N_lengthscales = misc.auto_lengthscales(self.N)
        self.M_lengthscales = misc.auto_lengthscales(self.M)
        self.MN_lengthscales = misc.auto_lengthscales(self.MN)
        self.Z_lengthscales = misc.auto_lengthscales(self.Z)

        self.N_kernel = gaussian.MultiDimGaussianKernel(self.N_lengthscales)
        self.MN_kernel = gaussian.MultiDimGaussianKernel(self.MN_lengthscales)
        self.Z_kernel = gaussian.MultiDimGaussianKernel(self.Z_lengthscales)

        self.fitted_X1: None | torch.Tensor = None

    @property
    def n(self) -> int:
        return self.M1.shape[0]

    @property
    def m(self) -> int:
        return self.M2.shape[0]

    def stage_1_tuning(
        self,
        K_X1X1: torch.FloatTensor,
        K_X2X1: torch.FloatTensor,
        K_X2X2: torch.FloatTensor,
        K_Z1Z1: torch.FloatTensor,
        K_Z1Z2: torch.FloatTensor,
        search_space: Iterable[float],
    ) -> Tuple[float, torch.FloatTensor]:
        n = K_X1X1.shape[0]
        m = K_X2X2.shape[0]

        def get_gamma_Z2(lmbda: float) -> torch.FloatTensor:
            gamma_Z2 = torch.linalg.solve(
                K_Z1Z1 + lmbda * n * torch.eye(n, device=K_Z1Z1.device), K_Z1Z2
            )
            return gamma_Z2  # Shape (n,m), n number of first stage samples

        def objective(lmbda: float) -> float:
            gamma_Z2 = get_gamma_Z2(lmbda)
            loss = (
                torch.trace(
                    K_X2X2 - 2 * K_X2X1 @ gamma_Z2 + gamma_Z2.T @ K_X1X1 @ gamma_Z2
                )
                / m
            )

            return loss.item()

        lmbda, _, fs = misc.minimize(objective, search_space)
        return lmbda.item(), get_gamma_Z2(lmbda)

    def stage_2_tuning(
        self,
        W: torch.FloatTensor,
        K_X1X1: torch.FloatTensor,
        Y1: torch.FloatTensor,
        Y2: torch.FloatTensor,
        search_space: Iterable[float],
    ) -> Tuple[float, torch.FloatTensor]:
        def get_alpha(xi: float) -> torch.FloatTensor:
            alpha = torch.linalg.solve(W @ W.T + self.m * xi * K_X1X1, W @ Y2)
            return alpha

        def KIV2_loss(xi: float) -> float:
            alpha = get_alpha(xi)
            preds = (alpha.T @ K_X1X1).T
            return torch.mean((Y1 - preds) ** 2).float().item()

        xi, _, fs = misc.minimize(KIV2_loss, search_space)

        return xi.item(), get_alpha(xi)

    def train(self):
        # Compute kernels
        self.K_N1N1 = self.N_kernel(self.N1, self.N1)
        self.K_N2N1 = self.N_kernel(self.N2, self.N1)
        self.K_N2N2 = self.N_kernel(self.N2, self.N2)

        self.K_MN1MN1 = self.MN_kernel(self.MN1, self.MN1)
        self.K_MN2MN1 = self.MN_kernel(self.MN2, self.MN1)
        self.K_MN2MN2 = self.MN_kernel(self.MN2, self.MN2)

        self.K_Z1Z1 = self.Z_kernel(self.Z1, self.Z1)
        self.K_Z1Z2 = self.Z_kernel(self.Z1, self.Z2)

        # Get lambda_N, lambda_MN
        lambda_N, gamma_N_Z2 = self.stage_1_tuning(
            self.K_N1N1,
            self.K_N2N1,
            self.K_N2N2,
            self.K_Z1Z1,
            self.K_Z1Z2,
            self.lmbda_search_space,
        )
        lambda_MN, gamma_MN_Z2 = self.stage_1_tuning(
            self.K_MN1MN1,
            self.K_MN2MN1,
            self.K_MN2MN2,
            self.K_Z1Z1,
            self.K_Z1Z2,
            self.lmbda_search_space,
        )

        print(f"{lambda_MN = }")
        print(f"{lambda_N = }")

        alpha_samples = self.N_kernel.sample_from_bochner(1000)

        self._X1_fitter = XModel(
            M=self.M1,
            N=self.N1,
            lambda_N=lambda_N,
            K_Z1Z1=self.K_Z1Z1,
            K_Z1Z2=self.K_Z1Z2,
            gamma_MN=gamma_MN_Z2,
            gamma_N=gamma_N_Z2,
            alpha_samples=alpha_samples,
            true_X=self.real_X1,
        )

        self._X1_fitter.fit(no_epochs=self.no_epochs)
        self.fitted_X1 = self._X1_fitter.X.to("cpu").detach()
        dist = torch.norm(self.real_X1 - self.fitted_X1, dim=-1).square().mean().item()
        print(f"Distance to true X1: {dist}")
        self.lambda_X = torch.exp(self._X1_fitter.log_lambda_X.to("cpu").detach())
        print(f"{self.lambda_X = }")

        self.X_kernel = gaussian.MultiDimGaussianKernel(
            misc.auto_lengthscales(self.fitted_X1)
        )
        K_X1X1 = self.X_kernel(self.fitted_X1, self.fitted_X1)

        W = K_X1X1 @ torch.linalg.solve(
            self.K_Z1Z1 + self.n * self.lambda_X * torch.eye(self.n), self.K_Z1Z2
        )

        xi, alpha = self.stage_2_tuning(
            W, K_X1X1, self.Y1, self.Y2, self.xi_search_space
        )
        self._alpha = alpha

        self._is_trained = True

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        K_Xxtest = self.X_kernel(self.fitted_X1, x)
        return K_Xxtest.T @ self._alpha

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            self.train()

        return self._predict(x)

    def losses_distances(self):
        if not self._is_trained:
            self.train()
        return self._X1_fitter.losses, self._X1_fitter.distances
