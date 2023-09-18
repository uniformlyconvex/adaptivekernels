import copy
import time
import torch
import torch.nn as nn
import wandb

from dataclasses import dataclass
from typing import Optional, Tuple

import src.utils.misc as utils
from src.structures.stage_data import StageData, TestData, StageLosses


TREATMENT_NET = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.Tanh()
)


INSTRUMENT_NET = nn.Sequential(
    nn.Linear(3, 128),
    nn.ReLU(),
    nn.Linear(128,64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.BatchNorm1d(32)
)



@dataclass
class DFIV:
    data: StageData
    lmbda_search_space: torch.Tensor
    xi_search_space: torch.Tensor

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    test_data: Optional[TestData] = None
    target: Optional[float] = None

    def __post_init__(self):
        self.data.to(self.device)
        if self.test_data is not None:
            self.test_data.to(self.device)

        self.treatment_net = TREATMENT_NET.to(self.device)
        self.instrument_net = INSTRUMENT_NET.to(self.device)

        self.treatment_opt = torch.optim.AdamW(self.treatment_net.parameters(), lr=1e-5)
        self.instrument_opt = torch.optim.AdamW(self.instrument_net.parameters(), lr=1e-4)

        self._is_trained = False

    def stage_1(self, iterations: int=20) -> StageLosses:
        self.treatment_net.train(False)
        self.instrument_net.train(True)
        
        # Target is the treatment_feature
        target = self.treatment_net(self.data.stage_1.X).detach()
        for i in range(iterations):
            self.instrument_opt.zero_grad()
            instrument_feature = self.augment_feature(self.instrument_net(self.data.stage_1.Z))

            # weight is (K+lambda)^{-1} X^t y
            weight = torch.linalg.solve(
                instrument_feature.T @ instrument_feature + len(self.data.stage_1) * self.lmbda * torch.eye(instrument_feature.shape[1], device=instrument_feature.device),
                instrument_feature.T @ target
            )
            predictions = instrument_feature @ weight

            loss = torch.norm(target - predictions) ** 2 + len(self.data.stage_1) * self.lmbda * torch.norm(weight) ** 2
            loss.backward()

            self.instrument_opt.step()
            # print(loss.item())

        return loss.item()

    def stage_2(self, iterations: int=1):
        self.treatment_net.train(True)
        self.instrument_net.train(False)

        # Instrument features are augmented
        instrument_feature_1 = self.augment_feature(self.instrument_net(self.data.stage_1.Z).detach())
        instrument_feature_2 = self.augment_feature(self.instrument_net(self.data.stage_2.Z).detach())

        for i in range(iterations):
            self.treatment_opt.zero_grad()
            treatment_feature_1 = self.treatment_net(self.data.stage_1.X)

            # Fit and predict for stage 2
            first_weight = torch.linalg.solve(
                instrument_feature_1.T @ instrument_feature_1 + len(self.data.stage_1) * self.lmbda * torch.eye(instrument_feature_1.shape[1], device=instrument_feature_1.device),
                instrument_feature_1.T @ treatment_feature_1
            )
            predicted_treatment_feature_2 = self.augment_feature(instrument_feature_2 @ first_weight)

            # Now regress y on predicted stage 2
            second_weight = torch.linalg.solve(
                predicted_treatment_feature_2.T @ predicted_treatment_feature_2 + len(self.data.stage_2) * self.xi * torch.eye(predicted_treatment_feature_2.shape[1], device=predicted_treatment_feature_2.device),
                predicted_treatment_feature_2.T @ self.data.stage_2.Y
            )
            predicted_outcome_2 = predicted_treatment_feature_2 @ second_weight

            loss = torch.norm(self.data.stage_2.Y - predicted_outcome_2) ** 2 + len(self.data.stage_2) * self.xi * torch.norm(second_weight) ** 2
            
            loss.backward()
            self.treatment_opt.step()

        return loss.item(), second_weight

    def train(self):
        wandb.init(project='DFIV')
        self.tune_regularization()
        print(f'Chose {self.lmbda} for lambda and {self.xi} for xi')
        self.best_MSE = float('inf')
        for epoch in range(2000):
            tic = time.time()
            stage_1_loss = self.stage_1(iterations=20)
            stage_2_loss, self._stage_2_weight = self.stage_2(iterations=1)


            toc = time.time()
            test_MSE = self.test_MSE()
            self.best_MSE = min(self.best_MSE, test_MSE)
            results = {
                'Test MSE': test_MSE,
                'Target MSE': self.target,
                'Time taken': toc-tic,
                'Stage 1 loss': stage_1_loss,
                'Stage 2 loss': stage_2_loss
            }
            wandb.log(results)
            print(f'Epoch {epoch} | Test MSE: {test_MSE} | Target MSE: {self.target} | Time taken: {toc-tic} | Stage 1 loss: {stage_1_loss} | Stage 2 loss: {stage_2_loss}')

        self.tune_regularization()
        final_mse = self.test_MSE()
        print(f'Final | Test MSE: {final_mse} | Target MSE: {self.target}')
        wandb.finish()
        self._is_trained = True

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        self.treatment_net.train(False)
        self.instrument_net.train(False)

        treatment_feature = self.augment_feature(self.treatment_net(x))
        predictions = treatment_feature @ self._stage_2_weight

        return predictions
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if not self._is_trained:
            self.train()
        return self._predict(x)
    
    def test_MSE(self):
        preds = self._predict(self.test_data.X)
        return self.test_data.evaluate_preds(preds)
    
    @staticmethod
    def augment_feature(feature: torch.Tensor) -> torch.Tensor:
        return torch.cat([feature, torch.ones((feature.shape[0], 1), device=feature.device)], dim=1)


    def tune_lmbda(self, search_space: torch.Tensor) -> torch.Tensor:
        target = self.treatment_net(self.data.stage_1.X).detach()
        instrument = self.augment_feature(self.instrument_net(self.data.stage_1.Z).detach())
        oos_target = self.treatment_net(self.data.stage_2.X).detach()
        oos_instrument = self.augment_feature(self.instrument_net(self.data.stage_2.Z).detach())


        def KIV1_loss(lmbda: float) -> float:
            lmbda = torch.tensor(lmbda, device=target.device)
            weight = torch.linalg.solve(
                instrument.T @ instrument + len(self.data.stage_1) * lmbda * torch.eye(instrument.shape[1], device=instrument.device),
                instrument.T @ target
            )
            # Compute loss on stage 2
            oos_predictions = oos_instrument @ weight
            loss = torch.norm(oos_target - oos_predictions) ** 2
            return loss.item()
        
        lmbda, _, _ = utils.minimize(KIV1_loss, search_space)
        return lmbda
    
    def tune_xi(self, lmbda: torch.Tensor, search_space: torch.Tensor) -> torch.Tensor:
        instrument_feature_1 = self.augment_feature(self.instrument_net(self.data.stage_1.Z).detach())
        instrument_feature_2 = self.augment_feature(self.instrument_net(self.data.stage_2.Z).detach())

        treatment_feature_1 = self.treatment_net(self.data.stage_1.X).detach()
        treatment_feature_2 = self.treatment_net(self.data.stage_2.X).detach()

        first_weight = torch.linalg.solve(
            instrument_feature_1.T @ instrument_feature_1 + len(self.data.stage_1) * lmbda * torch.eye(instrument_feature_1.shape[1], device=instrument_feature_1.device),
            instrument_feature_1.T @ treatment_feature_1
        )
        predicted_treatment_feature_2 = self.augment_feature(instrument_feature_2 @ first_weight)

        def KIV2_loss(xi: float) -> float:
            xi = torch.tensor(xi, device=treatment_feature_1.device)
            second_weight = torch.linalg.solve(
                predicted_treatment_feature_2.T @ predicted_treatment_feature_2 + len(self.data.stage_2) * xi * torch.eye(predicted_treatment_feature_2.shape[1], device=predicted_treatment_feature_2.device),
                predicted_treatment_feature_2.T @ self.data.stage_2.Y
            )

            # Now use the second weight to predict on stage 1
            predicted_outcome_1 = self.augment_feature(treatment_feature_1) @ second_weight
            loss = torch.norm(self.data.stage_1.Y - predicted_outcome_1) ** 2
            return loss.item()
        
        xi, _, _ = utils.minimize(KIV2_loss, search_space)
        return xi
    
    def tune_regularization(self):
        self.lmbda = self.tune_lmbda(self.lmbda_search_space)
        self.xi = self.tune_xi(self.lmbda, self.xi_search_space)
    