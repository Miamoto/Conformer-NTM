from abc import ABC
from abc import abstractmethod

import torch

class AbsNTM(torch.nn.Module, ABC):
    """
    Abstract base class for Neural Turing Machines (NTMs).
    """
    @abstractmethod
    def forward(
        self,
        xb: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Abstract method for the forward pass of the NTM.

        Args:
            xb (torch.Tensor): input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError

    @abstractmethod
    def init_params(self):
        """
        Abstract method for initializing the parameters of the NTM.
        """
        raise NotImplementedError

    @abstractmethod
    def init_state(
            self, 
            batch_size: int,
            device: str,
    ):
        """
        Abstract method for initializing the state of the NTM.

        Args:
            batch_size (int): Size of the input batch.
            device (str): Device on which the computation will be performed.
        """
        raise NotImplementedError
