import torch

from torch import nn
from espnet.nets.pytorch_backend.nets_utils import to_device

class Memory(nn.Module):
    """
    Class representing the memory component of a Neural Turing Machine (NTM).
    """
    def __init__(self, num_rows, num_cols):
        super(Memory, self).__init__()

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.data     = None

        #self.mem_bias = torch.Tensor().new_full((num_rows, num_cols), 1e-6)

    def init_state(self, batch_size, device):
        """
        Initializes the state of the memory.

        Args:
            batch_size (int): Size of the input batch.
            device (torch.device): Device on which the computation will be performed.
        """
        self.data = torch.zeros(batch_size, self.num_rows, self.num_cols).to(device)
        #self.data = self.mem_bias.clone().repeat(batch_size, 1, 1).to(device)
