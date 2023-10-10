import torch
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from torch.nn.utils import clip_grad_norm_

from espnet.nets.pytorch_backend.nets_utils import to_device

class HeadBase(nn.Module):
    """
    Base class for the heads of a Neural Turing Machine (NTM).
    """
    def __init__(self, memory, hidden_size, max_shift):
        super(HeadBase, self).__init__()

        self.memory = memory
        self.hidden_size = hidden_size
        self.max_shift = max_shift
 
        self.fc = nn.Linear(hidden_size + 1*self.memory.num_cols, 
                            sum([s for s, _ in self.hidden_state_unpacking_scheme()]))

        self.init_params()
    
    def hidden_state_unpacking_scheme():
        """
        Abstract method that defines the hidden state unpacking scheme.
        Subclasses must implement this method to specify the structure of the hidden state.

        Returns:
            List of tuples: Each tuple contains the size of the chunk and the activation function.
        """
        raise NotImplementedError
    
    def unpack_hidden_state(self, h):
        """
        Unpacks the hidden state tensor into chunks based on the hidden state unpacking scheme.

        Args:
            h (torch.Tensor): Hidden state tensor.

        Returns:
            Tuple: Unpacked hidden state tuple.
        """
        chunk_idxs, activations = zip(*self.hidden_state_unpacking_scheme())
        chunks = torch.split(h, chunk_idxs, dim=1)
        return tuple(activation(chunk) for chunk, activation in zip(chunks, activations))

    def focus_head(self, k, beta, prev_w, g, s, gamma, batch_size):
        """
        Computes the attention-based focus of the head.

        Args:
            k (torch.Tensor): Key vector.
            beta (torch.Tensor): Beta weight vector for content addressing.
            prev_w (torch.Tensor): Previous read/write weights.
            g (torch.Tensor): Gating scalar.
            s (torch.Tensor): Shift weighting vector.
            gamma (torch.Tensor): Sharpening scalar.
            batch_size (int): Size of the input batch.

        Returns:
            torch.Tensor: Content weight vector.
        """
        w_c = self._content_weight(k, beta)
        w_g = self._gated_interpolation(w_c, prev_w, g)
        w_s = self._mod_shift(w_g, s, batch_size)
        w   = self._sharpen(w_s, gamma) 
        return w
    
    def _gated_interpolation(self, w, prev_w, g):
        """
        Performs gated interpolation between the previous weight vector and the content weight vector.

        Args:
            w (torch.Tensor): Content weight vector.
            prev_w (torch.Tensor): Previous read/write weights.
            g (torch.Tensor): Gating scalar.

        Returns:
            torch.Tensor: Interpolated weight vector.
        """
        return g*w + (1-g)*prev_w

    def _mod_shift(self, w, s, batch_size):
        """
        Applies modified circular convolution to shift the weight vector.

        Args:
            w (torch.Tensor): Weight vector.
            s (torch.Tensor): Shift weighting vector.
            batch_size (int): Size of the input batch.

        Returns:
            torch.Tensor: Shifted weight vector.
        """
        unrolled = torch.cat([w[:, -self.max_shift:], w, w[:, :self.max_shift]], 1)
        return F.conv1d(unrolled.unsqueeze(1),
                        s.unsqueeze(1))[range(batch_size), range(batch_size)]
    
    def _sharpen(self, w, gamma):
        """
        Performs sharpening of the weight vector using a power and normalization operation.

        Args:
            w (torch.Tensor): Weight vector.
            gamma (torch.Tensor): Sharpening scalar.

        Returns:
            torch.Tensor: Sharpened weight vector.
        """
        w = w.pow(gamma)
        return torch.div(w, w.sum(1).view(-1, 1) + 1e-16)

    def _content_weight(self, k, beta):
        """
        Computes the content weight vector based on the cosine similarity between the key vector and memory.

        Args:
            k (torch.Tensor): Key vector.
            beta (torch.Tensor): Beta weight vector for content addressing.

        Returns:
            torch.Tensor: Content weight vector.
        """
        k = k.unsqueeze(1).expand_as(self.memory.data)
        similarity_scores = F.cosine_similarity(k, self.memory.data, dim=2)
        w = F.softmax(beta * similarity_scores, dim=1)
        return w

    def forward(self, h):
        """
        Abstract method for the forward pass of the head.

        Args:
            h (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError

    def init_state(self, batch_size):
        """
        Abstract method for initializing the state of the head.

        Args:
            batch_size (int): Size of the input batch.
        """
        raise NotImplementedError

    def init_params(self):
        """
        Method for initializing the parameters of the head.
        Subclasses can override this method to customize parameter initialization.
        """
        pass

class ReadHead(HeadBase):

    def __init__(self, memory, hidden_sz, max_shift):
        super(ReadHead, self).__init__(memory, hidden_sz, max_shift)

        #self.read_bias       = nn.Parameter(torch.randn(1, self.memory.num_cols))
        #self.read_focus_bias = nn.Parameter(torch.randn(1, self.memory.num_rows))
    
    def hidden_state_unpacking_scheme(self):
      return [
            # size, activation-function
            (self.memory.num_cols, torch.tanh),                    # k
            (1,                    F.softplus),                    # β
            (1,                    torch.sigmoid),                 # g
            (2*self.max_shift+1,   lambda x: F.softmax(x, dim=1)), # s
            (1,                    lambda x: 1 + F.softplus(x))    # γ
        ]

    def read(self, w):
        return torch.matmul(w.unsqueeze(1), self.memory.data).squeeze(1)
    
    def forward(self, h, prev_w, batch_size):
        k, beta, g, s, gamma = self.unpack_hidden_state(self.fc(h))
        w = self.focus_head(k, beta, prev_w, g, s, gamma, batch_size)
        read = self.read(w)
        return read, w

    def init_state(self, batch_size, device):
        #reads      = self.read_bias.clone().repeat(batch_size, 1).to(device)
        #read_focus = self.read_focus_bias.clone().repeat(batch_size, 1).to(device)
        #return reads, torch.softmax(read_focus, dim=1)
        reads      = torch.zeros(batch_size, self.memory.num_cols).to(device)
        read_focus = torch.zeros(batch_size, self.memory.num_rows).to(device)
        read_focus[:, 0] = 1
        return reads, read_focus

class WriteHead(HeadBase):

    def __init__(self, memory, hidden_sz, max_shift):
        super(WriteHead, self).__init__(memory, hidden_sz, max_shift)

        #self.write_focus_bias = nn.Parameter(torch.rand(1, self.memory.num_rows))

    def hidden_state_unpacking_scheme(self):
        return [
            # size, activation-function
            (self.memory.num_cols, torch.tanh),                    # k
            (1,                    F.softplus),                    # β 
            (1,                    torch.sigmoid),                 # g
            (2*self.max_shift+1,   lambda x: F.softmax(x, dim=1)), # s
            (1,                    lambda x: F.softplus(x) + 1),   # γ
            (self.memory.num_cols, torch.sigmoid),                 # e
            (self.memory.num_cols, torch.tanh)                     # a
        ] 
    
    def erase(self, w, e):
        return self.memory.data * (1 - w.unsqueeze(2) * e.unsqueeze(1))
    
    def write(self, w, e, a):
        memory_erased    = self.erase(w, e)
        self.memory.data = memory_erased + (w.unsqueeze(2) * a.unsqueeze(1))

    def forward(self, h, prev_w, batch_size):
        k, beta, g, s, gamma, e, a = self.unpack_hidden_state(self.fc(h))

        w = self.focus_head(k, beta, prev_w, g, s, gamma, batch_size)
        self.write(w, e, a)

        return w

    def init_state(self, batch_size, device):
        #write_focus = self.write_focus_bias.clone().repeat(batch_size, 1).to(device)
        #return torch.softmax(write_focus, dim=1)
        write_focus = torch.zeros(batch_size, self.memory.num_rows).to(device)
        write_focus[:, 0] = 1.
        return write_focus
