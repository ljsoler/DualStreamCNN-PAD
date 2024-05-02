from typing import Any
from archs.types_ import *
from torch import Tensor, nn
from abc import abstractmethod

class BaseCNN(nn.Module):
    
    def __init__(self) -> None:
        super(BaseCNN, self).__init__()

    def predict(self, input: Tensor) -> Any:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass

    @abstractmethod
    def accuracy(self, *inputs: Any, **kwargs) -> Tensor:
        pass

