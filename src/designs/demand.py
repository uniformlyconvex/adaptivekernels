import torch
import matplotlib.pyplot as plt

from dataclasses import dataclass
from typing import Tuple

from src.structures.stage_data import StageData, TestData
from src.utils.misc import evaluate_log10_mse

@dataclass
class Demand:
    rho: float
    seed: int = 0

    @staticmethod
    def psi(t: torch.Tensor) -> torch.Tensor:
        term_1 = (t - 5).pow(4) / 600
        term_2 = torch.exp(-4 * (t - 5).pow(2))
        term_3 = (t / 10) - 2
        return 2 * (term_1 + term_2 + term_3)

    @staticmethod
    def structural_function(X: torch.Tensor) -> torch.Tensor:
        # X is a tensor of shape (N, 3), where X[:,0] = P, X[:,1] = T, X[:,2] = S
        P = X[:,0]
        T = X[:,1]
        S = X[:,2]
        
        return 100 + (10+P) * S * Demand.psi(T) - 2 * P
    
    def _gen_data(self, no_points: int) -> Tuple[torch.Tensor]:
        torch.manual_seed(self.seed)
        # S ~ Unif{1,...,7}
        S_dist = torch.distributions.Categorical(
            logits=torch.ones(7)
        )
        S = S_dist.sample((no_points,)) + 1  # To get values between 1 and 7
        S = S.reshape(-1,1)

        # T ~ Unif[0,10]
        T_dist = torch.distributions.Uniform(0, 10)
        T = T_dist.sample((no_points,))
        T = T.reshape(-1,1)

        # (C,V) ~ N(0,I_2)
        CV_dist = torch.distributions.MultivariateNormal(
            loc = torch.tensor([0.0, 0.0]),
            covariance_matrix = torch.tensor([
                [1.0, 0.0],
                [0.0, 1.0]
            ])
        )
        CV = CV_dist.sample((no_points,))
        C, V = CV[:,0], CV[:,1]
        C = C.reshape(-1,1)
        V = V.reshape(-1,1)

        # e ~ N(rho * V, 1-rho^2)
        e_dist = torch.distributions.Normal(0, (1-self.rho**2)**0.5)
        e = e_dist.sample((no_points,))
        e = e.reshape(-1,1)
        e += self.rho * V

        # P = 25 + (C+3)*psi(T) + V
        P = 25 + (C+3) * Demand.psi(T) + V

        # X = (P,T,S), Z = (C,T,S)
        X = torch.hstack((P,T,S))
        Z = torch.hstack((C,T,S))

        assert X.shape == Z.shape == (no_points,3)

        # Y = h(X) + e
        Y = Demand.structural_function(X).reshape(-1,1) + e
        return X, Y, Z

    def generate_KIV_data(self, no_points: int) -> StageData:
        X, Y, Z = self._gen_data(no_points)
        return StageData.from_all_data(X, Y, Z)
    
    def generate_MEKIV_data(self, no_points: int, merror_type: str, merror_scale: float):
        X, Y, Z = self._gen_data(no_points)
        if merror_type == 'gaussian':
            # Add Gaussian noise
            X_std = X.std(dim=0)

            err_dist = torch.distributions.MultivariateNormal(
                loc=torch.zeros(3),
                covariance_matrix=torch.diag(X_std * merror_scale)**2
            )
            print(err_dist.covariance_matrix)
            delta_M = err_dist.sample((no_points,))
            delta_N = err_dist.sample((no_points,))
            print(delta_M.std(dim=0))
            print(delta_N.std(dim=0))

            M = X + delta_M
            N = X + delta_N

        if merror_type == 'mog':
            P, T, S = X[:,0], X[:,1], X[:,2]
            P_dist = torch.distributions.Normal(2 * P.std(), merror_scale * P.std())
            P_dist_neg = torch.distributions.Normal(-2 * P.std(), merror_scale * P.std())
            T_dist = torch.distributions.Normal(2 * T.std(), merror_scale * T.std())
            T_dist_neg = torch.distributions.Normal(-2 * T.std(), merror_scale * T.std())
            S_dist = torch.distributions.Normal(2 * S.std(), merror_scale * S.std())
            S_dist_neg = torch.distributions.Normal(-2 * S.std(), merror_scale * S.std())

            p_samples = torch.zeros((no_points,))
            for i in range(no_points):
                if torch.rand(1) > 0.5:
                    p_samples[i] = P_dist.sample()
                else:
                    p_samples[i] = P_dist_neg.sample()

            t_samples = torch.zeros((no_points,))
            for i in range(no_points):
                if torch.rand(1) > 0.5:
                    t_samples[i] = T_dist.sample()
                else:
                    t_samples[i] = T_dist_neg.sample()

            s_samples = torch.zeros((no_points,))
            for i in range(no_points):
                if torch.rand(1) > 0.5:
                    s_samples[i] = S_dist.sample()
                else:
                    s_samples[i] = S_dist_neg.sample()

            delta = torch.vstack((p_samples, t_samples, s_samples)).T
            M = X + delta

            p_samples = torch.zeros((no_points,))
            for i in range(no_points):
                if torch.rand(1) > 0.5:
                    p_samples[i] = P_dist.sample()
                else:
                    p_samples[i] = P_dist_neg.sample()

            t_samples = torch.zeros((no_points,))
            for i in range(no_points):
                if torch.rand(1) > 0.5:
                    t_samples[i] = T_dist.sample()
                else:
                    t_samples[i] = T_dist_neg.sample()

            s_samples = torch.zeros((no_points,))
            for i in range(no_points):
                if torch.rand(1) > 0.5:
                    s_samples[i] = S_dist.sample()
                else:
                    s_samples[i] = S_dist_neg.sample()

            delta = torch.vstack((p_samples, t_samples, s_samples)).T
            N = X + delta
        
        return X, M, N, Y, Z

    def generate_test_data(self, no_points: int=2800) -> TestData:
        # We always use 2800 points for test data but keep the argument for consistency with other designs
        P = torch.linspace(10, 25, 20)
        T = torch.linspace(0, 10, 20)
        S = torch.tensor(range(1,8)).float()

        X = torch.cartesian_prod(P, T, S)
        truth = Demand.structural_function(X).reshape(-1,1)

        metric = evaluate_log10_mse

        return TestData(X, truth, metric)
