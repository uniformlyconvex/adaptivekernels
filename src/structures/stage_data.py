import torch

from dataclasses import dataclass, fields, Field
from typing import Optional, Callable, Any, Tuple, Iterable

import src.utils.misc as utils


def _tensor_fields(obj: Any) -> Tuple[Field[torch.Tensor]]:
    return (
        field for field in fields(obj) 
        if isinstance(getattr(obj, field.name), torch.Tensor)
    )

class _TensorDataclass:
    def __post_init__(self):
        self._length = None
        for field in _tensor_fields(self):
            field_attr = getattr(self, field.name)
            setattr(self, field.name, field_attr.float())
            if self._length is None:
                self._length = len(field_attr)
            else:
                assert self._length == len(field_attr), "All tensors must have the same length"

    def to(self, device: str | torch.device) -> None:
        for field in _tensor_fields(self):
            setattr(self, field.name, getattr(self, field.name).to(device))

    def __getitem__(self, indices):
        tensors = {
            field.name: getattr(self, field.name)[indices]
            for field in _tensor_fields(self)
        }
        others = {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in tensors
        }
        return type(self)(**tensors, **others)
    
    def __len__(self):
        return self._length
    
    def minibatches(self, batch_size: int, shuffle: bool=True):
        indices = torch.randperm(len(self)) if shuffle else torch.arange(len(self))
        for minibatch in indices.split(batch_size):
            yield self[minibatch]


@dataclass
class Stage1Data(_TensorDataclass):
    X: torch.Tensor
    Z: torch.Tensor
    Y: Optional[torch.Tensor]


@dataclass
class MEKIVStage1Data(_TensorDataclass):
    M: torch.Tensor
    N: torch.Tensor
    Z: torch.Tensor
    Y: Optional[torch.Tensor]
    X: Optional[torch.Tensor]

    @property
    def MN(self) -> torch.Tensor:
        return torch.hstack((self.M, self.N))


@dataclass
class MEKIVStage2Data(_TensorDataclass):
    Y: torch.Tensor
    Z: torch.Tensor
    M: Optional[torch.Tensor]
    N: Optional[torch.Tensor]
    X: Optional[torch.Tensor]

    @property
    def MN(self) -> torch.Tensor:
        return torch.hstack((self.M, self.N))


@dataclass
class Stage2Data(_TensorDataclass):
    Y: torch.Tensor
    Z: torch.Tensor
    X: Optional[torch.Tensor]
    

@dataclass
class StageData:
    stage_1: Stage1Data
    stage_2: Stage2Data

    @property
    def all_X(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.X, self.stage_2.X))
    
    @property
    def all_Y(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.Y, self.stage_2.Y))
    
    @property
    def all_Z(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.Z, self.stage_2.Z))

    def to(self, device: str | torch.device) -> None:
        self.stage_1.to(device)
        self.stage_2.to(device)
    
    @classmethod
    def from_all_data(cls, X: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, p: float=0.5) -> 'StageData':
        first, second = utils.rand_split((X, Y, Z), p=p)
        X1, Y1, Z1 = first
        X2, Y2, Z2 = second
        return cls(
            Stage1Data(X=X1, Z=Z1, Y=Y1),
            Stage2Data(X=X2, Z=Z2, Y=Y2)
        )
    
    def __len__(self):
        return len(self.all_X)
    
    def minibatches(self, batch_size: int, shuffle: bool=True):
        return zip(
            self.stage_1.minibatches(batch_size, shuffle),
            self.stage_2.minibatches(batch_size, shuffle)
        )


@dataclass
class MEKIVStageData:
    stage_1: MEKIVStage1Data
    stage_2: MEKIVStage2Data

    @property
    def all_M(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.M, self.stage_2.M))
    
    @property
    def all_N(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.N, self.stage_2.N))
    
    @property
    def all_MN(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.MN, self.stage_2.MN))

    @property
    def all_Y(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.Y, self.stage_2.Y))
    
    @property
    def all_Z(self) -> torch.Tensor:
        return torch.vstack((self.stage_1.Z, self.stage_2.Z))
    
    def to(self, device: str | torch.device) -> None:
        self.stage_1.to(device)
        self.stage_2.to(device)

    @classmethod
    def from_all_data(cls, M: torch.Tensor, N: torch.Tensor, Y: torch.Tensor, Z: torch.Tensor, p: float=0.5):
        first, second = utils.rand_split((M, N, Y, Z), p=p)
        M1, N1, Y1, Z1 = first
        M2, N2, Y2, Z2 = second
        return cls(
            MEKIVStage1Data(M=M1, N=N1, Z=Z1, Y=Y1),
            MEKIVStage2Data(Y=Y2, Z=Z2, M=M2, N=N2)
        )

@dataclass
class TestData(_TensorDataclass):
    X: torch.Tensor
    truth: torch.Tensor
    metric: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def evaluate_preds(self, preds: torch.Tensor) -> float:
        preds = preds.to(self.truth.device)
        return self.metric(preds, self.truth).to('cpu').item()


@dataclass
class StageLosses:
    name: str
    metrics: Iterable[Tuple[float, str]]

    def wandb_dict(self) -> dict:
        return {
            f"[{self.name}] {metric[1]}": metric[0]
            for metric in self.metrics
        }