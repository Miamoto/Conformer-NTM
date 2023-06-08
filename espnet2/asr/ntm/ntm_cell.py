import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from torch.nn.utils import clip_grad_norm_

from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet2.asr.ntm.memory import Memory
from espnet2.asr.ntm.heads import ReadHead
from espnet2.asr.ntm.heads import WriteHead

from espnet2.asr.ntm.abs_ntm import AbsNTM

class NTMCell(AbsNTM):
    """NTMCell class.

    Args:
        input_size: The number of expected features in the input
        output_size: The number of output features
        memory_num_rows: The number of rows in memory
        memory_num_cols: The number of columns in memory
        hidden_output: The output of the encoder network
        num_heads: Number of heads
        max_shift: Maximum shift

    """
    def __init__(
            self, 
            input_size: int,
            output_size: int,
            memory_num_rows: int, 
            memory_num_cols: int,
            num_heads: int =1,
            max_shift: int =1,
    ):
        
        super().__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        
        self.num_heads = num_heads
        self.max_shift = max_shift
        
        self.memory_num_rows = memory_num_rows
        self.memory_num_cols = memory_num_cols
        
        self.state_container = namedtuple('state', [
            'read_vectors', 'read_focus_weights', 'write_focus_weights'
        ])
         
        # module instantiations
        self.memory = Memory(memory_num_rows, memory_num_cols)
     
        # create heads
        self.read_heads = nn.ModuleList([
            ReadHead(self.memory, input_size, max_shift)
            for _ in range(num_heads)
        ])
        self.write_heads = nn.ModuleList([
            WriteHead(self.memory, input_size, max_shift)
            for _ in range(num_heads)
        ])
 
        self.init_params()

        self.fc = torch.nn.Linear(input_size + num_heads * memory_num_cols, output_size)

    def forward(
            self, 
            h_pad: torch.Tensor,
    ) -> torch.Tensor:
        
        # unpacking the previous state
        prev_reads      = self.state.read_vectors
        prev_read_foci  = self.state.read_focus_weights
        prev_write_foci = self.state.write_focus_weights
    
        # pad when input contains smaller batch 
        h_pad_size = h_pad.size(0)
        read_size = prev_reads[-1].size(0)

        if h_pad_size < read_size:
            h_pad = F.pad(h_pad,(0,0,read_size-h_pad_size,0))

        h_pad_aux = torch.cat([h_pad, *prev_reads], dim=1)
        
        # read and write
        reads      = []
        read_foci  = []
        write_foci = []
        
        for i, (read_head, write_head) in enumerate(zip(self.read_heads, self.write_heads)):
            read, read_focus = read_head(h_pad_aux,  prev_read_foci[i], self.batch_size)    # read
            write_focus      = write_head(h_pad_aux, prev_write_foci[i], self.batch_size)   # write
         
            reads      += [read]
            read_foci  += [read_focus]
            write_foci += [write_focus]
            
        # pack new state
        self.state = self.state_container(reads, read_foci, write_foci)
        
        # output
        return self.fc(torch.cat([h_pad, *reads], dim=1)) 

    def init_params(self):
        pass

    def init_state(self, device, batch_size):
 
        # Initialize the memory
        self.memory.init_state(batch_size, device)
        self.batch_size = batch_size

        # init heads and collect initial read vectors, read foci and write foci
        reads      = []
        read_foci  = []
        write_foci = []
        
        for rh, wh in zip(self.read_heads, self.write_heads):
            read, read_focus = rh.init_state(batch_size, device)
            write_focus      = wh.init_state(batch_size, device)
            
            reads      += [read]
            read_foci  += [read_focus]
            write_foci += [write_focus]
        
        # pack the initial state
        self.state = self.state_container(reads, read_foci, write_foci)
