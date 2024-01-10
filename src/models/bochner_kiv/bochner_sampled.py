import copy
import time
import torch
import torch.nn as nn
import wandb

from dataclasses import dataclass
from typing import Optional, Tuple

import src.utils.misc as utils
from src.kernels.bochner import BochnerKernel
from src.structures.stage_data import StageData, TestData, StageLosses

@dataclass
class BochnerKIVSampled:
    data: StageData
    lmbda_search_space: torch.Tensor
    xi_search_space: torch.Tensor

    X_net: torch.nn.Module
    Z_net: torch.nn.Module

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_data: Optional[TestData] = None
    target: Optional[float] = None

    def __post_init__(self):
        self.data.to(self.device)
        if self.test_data is not None:
            self.test_data.to(self.device)

        X_lengthscales = utils.auto_lengthscales(self.data.all_X)
        Z_lengthscales = utils.auto_lengthscales(self.data.all_Z)
        self.X_kernel = BochnerKernel(dimension=self.data.all_X.shape[1], lengthscales=X_lengthscales, network=self.X_net)
        self.Z_kernel = BochnerKernel(dimension=self.data.all_Z.shape[1], lengthscales=Z_lengthscales, network=self.Z_net)

        # Get optimizers
        # Below seems to work well?
        self.Z_opt = torch.optim.AdamW(self.Z_kernel.parameters(), lr=1e-4)
        self.X_opt = torch.optim.AdamW(self.X_kernel.parameters(), lr=4e-5)

        self.best_performance: float = float('inf')
        self.best_parameters: Tuple[dict, dict] = (
            copy.deepcopy(self.X_kernel.state_dict()),
            copy.deepcopy(self.Z_kernel.state_dict())
        )

        self._is_trained: bool = False

    def stage_1(self, iterations: int=20, minibatched: bool=True) -> StageLosses:
        # Regress treatment feature from 1st stage on instrument feature from 1st stage
        # Training Z kernel
        self.X_kernel.train(False)
        self.Z_kernel.train(True)

        # If the minibatch size is sufficiently large, we are just using the whole dataset
        MINIBATCH_SIZE = 200 if minibatched else max(len(self.data.stage_1), len(self.data.stage_2))
        NO_SAMPLES = 20_000
        MINIBATCH_SIZE = 10000

        # We can draw these once and detach them since we're training the Z kernel
        X_samples = self.X_kernel.sample(NO_SAMPLES).detach()

        for i in range(iterations):
            stage_1_regularized_loss_overall = 0.0
            stage_1_unregularized_loss_overall = 0.0
            stage_2_unregularized_loss_overall = 0.0

            for stage_1, stage_2 in self.data.minibatches(MINIBATCH_SIZE):
                self.Z_opt.zero_grad()
                Z_samples = self.Z_kernel.sample(NO_SAMPLES)

                # In-sample loss
                with torch.no_grad():
                    K_X1X1 = self.X_kernel.evaluate_from_samples(stage_1.X, stage_1.X, X_samples)
                
                K_Z1Z1 = self.Z_kernel.evaluate_from_samples(stage_1.Z, stage_1.Z, Z_samples)
                gamma_11 = torch.linalg.solve(
                    K_Z1Z1 + len(stage_1) * self.lmbda * torch.eye(len(stage_1), device=K_Z1Z1.device),
                    K_Z1Z1
                )
                stage_1_unregularized_loss = torch.trace(
                    K_X1X1 - 2 * K_X1X1 @ gamma_11 + gamma_11.T @ K_X1X1 @ gamma_11
                ) / len(stage_1)
                ridge = K_Z1Z1 + len(stage_1) * self.lmbda * torch.eye(len(stage_1), device=K_Z1Z1.device)
                regularizer = self.lmbda * torch.trace(
                    torch.linalg.solve(ridge, K_Z1Z1) @
                    torch.linalg.solve(ridge, K_X1X1)
                )
                stage_1_regularized_loss = stage_1_unregularized_loss + regularizer

                stage_1_regularized_loss.backward()
                self.Z_opt.step()

                # Out-of-sample loss and book keeping
                with torch.no_grad():
                    K_X2X1 = self.X_kernel.evaluate_from_samples(stage_2.X, stage_1.X, X_samples)
                    K_X2X2 = self.X_kernel.evaluate_from_samples(stage_2.X, stage_2.X, X_samples)

                K_Z1Z2 = self.Z_kernel.evaluate_from_samples(stage_1.Z, stage_2.Z, Z_samples)
                gamma_12 = torch.linalg.solve(
                    K_Z1Z1 + len(stage_1) * self.lmbda * torch.eye(len(stage_1), device=K_Z1Z1.device),
                    K_Z1Z2
                )
                stage_2_unregularized_loss = torch.trace(
                    K_X2X2 - 2 * K_X2X1 @ gamma_12 + gamma_12.T @ K_X1X1 @ gamma_12
                ) / len(stage_2)

                stage_1_regularized_loss_overall += stage_1_regularized_loss.item() * len(stage_1) / len(self.data.stage_1)
                stage_1_unregularized_loss_overall += stage_1_unregularized_loss.item() * len(stage_1) / len(self.data.stage_1)
                stage_2_unregularized_loss_overall += stage_2_unregularized_loss.item() * len(stage_2) / len(self.data.stage_2)
        
        return StageLosses(
            name="Z kernel",
            metrics=[
                (stage_1_regularized_loss_overall, "Stage 1 data (regularized)"),
                (stage_1_unregularized_loss_overall, "Stage 1 data (unregularized)"),
                (stage_2_unregularized_loss_overall, "Stage 2 data (unregularized)")
            ]
        )


    def stage_2(self, iterations: int=1, minibatched: bool=True) -> StageLosses:
        """
        Regress outcome from 2nd stage on predicted treatment from 2nd stage
        """
        self.X_kernel.train(True)
        self.Z_kernel.train(False)

        # If the minibatch size is sufficiently large, we are just using the whole dataset
        MINIBATCH_SIZE = 100 if minibatched else max(len(self.data.stage_1), len(self.data.stage_2))
        NO_SAMPLES = 10_000

        # We can draw these once and detach them since we're training the X kernel
        Z_samples = self.Z_kernel.sample(NO_SAMPLES).detach()

        for i in range(iterations):
            stage_2_unregularized_loss_overall = 0.0
            stage_2_regularized_loss_overall = 0.0
            stage_1_unregularized_loss_overall = 0.0

            for stage_1, stage_2 in self.data.minibatches(MINIBATCH_SIZE):
                self.X_opt.zero_grad()
                X_samples = self.X_kernel.sample(NO_SAMPLES)

                # In-sample loss
                with torch.no_grad():
                    K_Z1Z1 = self.Z_kernel.evaluate_from_samples(stage_1.Z, stage_1.Z, Z_samples)
                    K_Z1Z2 = self.Z_kernel.evaluate_from_samples(stage_1.Z, stage_2.Z, Z_samples)
                
                K_X1X1 = self.X_kernel.evaluate_from_samples(stage_1.X, stage_1.X, X_samples)
                W = K_X1X1 @ torch.linalg.solve(
                    K_Z1Z1 + len(stage_1) * self.lmbda * torch.eye(len(stage_1), device=K_Z1Z1.device),
                    K_Z1Z2
                )
                alpha = torch.linalg.solve(
                    W @ W.T + len(stage_2) * self.xi * K_X1X1,
                    (W @ stage_2.Y)
                )
                stage_2_unregularized_loss = torch.norm(stage_2.Y - W.T @ alpha, dim=1).square().mean()
                regularizer = self.xi * alpha.T @ K_X1X1 @ alpha
                stage_2_regularized_loss = stage_2_unregularized_loss + regularizer

                stage_2_regularized_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.X_kernel.parameters(), 0.1)
                self.X_opt.step()

                # Out-of-sample loss and book keeping
                stage_1_unregularized_loss = torch.norm(stage_1.Y - K_X1X1.T @ alpha, dim=1).square().mean()

                stage_2_unregularized_loss_overall += stage_2_unregularized_loss.item() * len(stage_2) / len(self.data.stage_2)
                stage_2_regularized_loss_overall += stage_2_regularized_loss.item() * len(stage_2) / len(self.data.stage_2)
                stage_1_unregularized_loss_overall += stage_1_unregularized_loss.item() * len(stage_1) / len(self.data.stage_1)
        
        return StageLosses(
            name="X kernel",
            metrics=[
                (stage_2_unregularized_loss_overall, "Stage 2 data (unregularized)"),
                (stage_2_regularized_loss_overall, "Stage 2 data (regularized)"),
                (stage_1_unregularized_loss_overall, "Stage 1 data (unregularized)")
            ]
        )
    
    def tune_lmbda(self, search_space: torch.Tensor, no_samples: int=50_000) -> torch.Tensor:
        self.X_kernel.train(False)
        self.Z_kernel.train(False)

        K_Z1Z1 = self.Z_kernel.evaluate_precisely_no_grad(self.data.stage_1.Z, self.data.stage_1.Z, 1000, int(no_samples/1000))
        K_Z1Z2 = self.Z_kernel.evaluate_precisely_no_grad(self.data.stage_1.Z, self.data.stage_2.Z, 1000, int(no_samples/1000))
        
        K_X1X1 = self.X_kernel.evaluate_precisely_no_grad(self.data.stage_1.X, self.data.stage_1.X, 1000, int(no_samples/1000))
        K_X2X1 = self.X_kernel.evaluate_precisely_no_grad(self.data.stage_2.X, self.data.stage_1.X, 1000, int(no_samples/1000))
        K_X2X2 = self.X_kernel.evaluate_precisely_no_grad(self.data.stage_2.X, self.data.stage_2.X, 1000, int(no_samples/1000))

        def KIV1_loss(lmbda: float) -> float:            
            lmbda = torch.tensor(lmbda, device=K_Z1Z1.device)
            gamma = torch.linalg.solve(
                K_Z1Z1 + lmbda * len(self.data.stage_1) * torch.eye(len(self.data.stage_1), device=K_Z1Z1.device),
                K_Z1Z2
            )

            loss = torch.trace(
                K_X2X2 - 2 * K_X2X1 @ gamma + gamma.T @ K_X1X1 @ gamma
            ) / len(self.data.stage_2)

            return loss.item()

        lmbda, _, _ = utils.minimize(KIV1_loss, search_space)
        return lmbda

    def tune_xi(self, lmbda: torch.Tensor, search_space: torch.Tensor, no_samples: int=50_000) -> torch.Tensor:
        self.X_kernel.train(False)
        self.Z_kernel.train(False)

        K_Z1Z1 = self.Z_kernel.evaluate_precisely_no_grad(self.data.stage_1.Z, self.data.stage_1.Z, 1000, int(no_samples/1000))
        K_Z1Z2 = self.Z_kernel.evaluate_precisely_no_grad(self.data.stage_1.Z, self.data.stage_2.Z, 1000, int(no_samples/1000))
        
        K_X1X1 = self.X_kernel.evaluate_precisely_no_grad(self.data.stage_1.X, self.data.stage_1.X, 1000, int(no_samples/1000))

        
        W = K_X1X1 @ torch.linalg.solve(
            K_Z1Z1 + lmbda.to(K_Z1Z1.device) * len(self.data.stage_1) * torch.eye(len(self.data.stage_1), device=K_Z1Z1.device),
            K_Z1Z2
        )

        def KIV2_loss(xi: float) -> float:
            xi = torch.tensor(xi, device=W.device)
            alpha = torch.linalg.solve(
                W @ W.T + xi * len(self.data.stage_2) * K_X1X1,
                (W @ self.data.stage_2.Y)
            )
            preds = K_X1X1.T @ alpha
            loss = torch.norm(self.data.stage_1.Y - preds, dim=1).square().mean()
            return loss.item()

        xi, _, _ = utils.minimize(KIV2_loss, search_space)

        return xi
    
    def tune_regularization(self):
        self.lmbda = self.tune_lmbda(self.lmbda_search_space)
        self.xi = self.tune_xi(self.lmbda, self.xi_search_space)

    def _predict(self, x: torch.Tensor, no_samples: int=50_000) -> torch.Tensor:
        self.X_kernel.train(False)
        self.Z_kernel.train(False)

        K_X1X1 = self.X_kernel.evaluate_precisely_no_grad(self.data.stage_1.X, self.data.stage_1.X, 1000, int(no_samples/1000))
        K_X1x = self.X_kernel.evaluate_precisely_no_grad(self.data.stage_1.X, x, 100, int(no_samples/100))
        K_Z1Z1 = self.Z_kernel.evaluate_precisely_no_grad(self.data.stage_1.Z, self.data.stage_1.Z, 1000, int(no_samples/1000))
        K_Z1Z2 = self.Z_kernel.evaluate_precisely_no_grad(self.data.stage_1.Z, self.data.stage_2.Z, 1000, int(no_samples/1000))

        W = K_X1X1 @ torch.linalg.solve(
            K_Z1Z1 + len(self.data.stage_1) * self.lmbda * torch.eye(len(self.data.stage_1), device=K_Z1Z1.device),
            K_Z1Z2
        )
        alpha = torch.linalg.solve(
            W @ W.T + len(self.data.stage_2) * self.xi * K_X1X1,
            (W @ self.data.stage_2.Y)
        )

        preds = K_X1x.T @ alpha

        return preds


    def _predict(self, x: torch.Tensor, no_samples: int=50_000) -> torch.Tensor:
            self.X_kernel.train(False)
            self.Z_kernel.train(False)

            K_X1X1 = self.X_kernel.evaluate_precisely_no_grad(self.data.stage_1.X, self.data.stage_1.X, 1000, int(no_samples/1000))
            K_X1x = self.X_kernel.evaluate_precisely_no_grad(self.data.stage_1.X, x, 100, int(no_samples/100))
            K_Z1Z1 = self.Z_kernel.evaluate_precisely_no_grad(self.data.stage_1.Z, self.data.stage_1.Z, 1000, int(no_samples/1000))
            K_Z1Z2 = self.Z_kernel.evaluate_precisely_no_grad(self.data.stage_1.Z, self.data.stage_2.Z, 1000, int(no_samples/1000))

            W = K_X1X1 @ torch.linalg.solve(
                K_Z1Z1 + len(self.data.stage_1) * self.lmbda * torch.eye(len(self.data.stage_1), device=K_Z1Z1.device),
                K_Z1Z2
            )
            alpha = torch.linalg.solve(
                W @ W.T + len(self.data.stage_2) * self.xi * K_X1X1,
                (W @ self.data.stage_2.Y)
            )

            preds = K_X1x.T @ alpha

            return preds
    
    def test_MSE(self) -> float:
        preds = self._predict(self.test_data.X)
        return self.test_data.evaluate_preds(preds)
    
    def copy_model_parameters(self) -> Tuple[dict, dict]:
        return (
            copy.deepcopy(self.X_kernel.state_dict()),
            copy.deepcopy(self.Z_kernel.state_dict())
        )
    
    def update_best_params(self, metric: float) -> None:
        if metric <= self.best_performance:
            print(f'New best performance: {metric:.4f}')
            self.best_performance = metric
            self.best_params = self.copy_model_parameters()

    def restore_best_params(self) -> None:
        self.X_kernel.load_state_dict(self.best_params[0])
        self.Z_kernel.load_state_dict(self.best_params[1])
        print(f'Best parameters restored')
    
    def train(self):
        wandb.init(project='KIV')
        self.tune_regularization()
        print(f'Before training | Test MSE: {self.test_MSE():.4f}')
        try:
            for epoch in range(250):
                tic = time.time()
                stage_1_losses = self.stage_1(iterations=20)
                stage_2_losses = self.stage_2(iterations=10)

                # self.update_best_params(stage_2_losses.oos[0])
                test_MSE = self.test_MSE()
                toc = time.time()

                results = {
                    'Test MSE': test_MSE,
                    'Target MSE': self.target,
                    'Time taken': toc - tic,
                    **stage_1_losses.wandb_dict(),
                    **stage_2_losses.wandb_dict()
                }
                wandb.log(results)
                print(f'Epoch {epoch} | Test MSE: {test_MSE:.4f} | Target MSE: {self.target:.4f} | Time taken: {toc-tic:.2f} seconds')
        except KeyboardInterrupt:
            print('Training interrupted')
        # self.restore_best_params()
        self.tune_regularization()
        final_mse = self.test_MSE()
        print(f'Final | Test MSE: {final_mse:.4f} | Target MSE: {self.target:.4f}')
        wandb.log({'Test MSE': final_mse})
        wandb.finish()
        self._is_trained = True

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            self.train()
        
        return self._predict(x).detach().cpu()
        
