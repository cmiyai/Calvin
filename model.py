import torch
import torch.nn as nn
import math

def xavier_truncated_normal_(x: torch.Tensor, in_features: int, out_features: int):
        '''
        Xavier normal initialization, distribution truncated to +-3 std
        '''
        d_in, d_out = in_features, out_features
        std = math.sqrt(2.0/(d_in + d_out))
        with torch.no_grad():
            return nn.init.trunc_normal_(x, mean=0.0, std=std, a=-3.0*std, b=3.0*std)

class Linear(nn.Module):
    '''
    Linear class to perform linear transformation: y = x *(W)^T
    '''

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weights = nn.Parameter(
            torch.randn(out_features, in_features, **factory_kwargs)
        )
        xavier_truncated_normal_(self.weights, self.in_features, self.out_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T

class Embedding(nn.Module):
    '''
    Custom Embedding class from scratch
    Maps integer token IDs into a vector space of dimension d_model
    '''
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        factory_kwargs = {'device': device, 'dtype': dtype}
        
        # our weight matrix has one row for every token in vocabulary
        self.weights = nn.Parameter(
            torch.randn(num_embeddings, embedding_dim, **factory_kwargs)
        )
        with torch.no_grad():
            nn.init.trunc_normal_(self.weights, mean=0.0, std=1.0, a=-3.0, b=3.0)
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weights[token_ids]

class RMSNorm(nn.Module):
    '''
    Apply the RMS to the activations
    '''
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))


    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32) # up scale to float32 to prevent overflow
        d_m = self.d_model
        eps = self.eps

        a_sum = (x * x).sum(dim=-1, keepdim=True)
        # RMSNorm Denominator: RMS(a)
        rms_denom = torch.sqrt((1/d_m) * a_sum + eps)
        result = (x / rms_denom) * self.weight
        return result.to(in_dtype) # cast back into original

class SwiGLU(nn.Module):
    '''
    Implementation of SwiGLU activation function from scratch
    '''
    def __init__(self, d_model: int, d_ff: int):
        # d_ff is the dims of inner layer of position-wise FFN
        # approx: d_ff = 8/3 * d_model
        super().__init__()
        # initialize W1 and W3 for gating
        self.w1 = Linear(in_features=d_model, out_features=d_ff)
        self.w3 = Linear(in_features=d_model, out_features=d_ff)

        # Initialize W2
        self.w2 = Linear(in_features=d_ff, out_features=d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # FFN(x) = W2(Swish(W1_x) * (W3x))
        w1_x = self.w1(x)
        w3_x = self.w3(x)

        # Swish(x) = x * sigmoid(x)
        swish_w1_x = w1_x * torch.sigmoid(w1_x)
        # GLU on W1 and W3
        gated = swish_w1_x * w3_x

        return self.w2(gated)

class RoPE(nn.Module):
    '''
    Implementation of Rotary Position Embeddings (RoPE)
    RoPE rotates queries and keys by angles determined by token position
    '''
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.theta = theta # float value for the rotation
        self.d_k = d_k# dimension of query and key vectors
        self.max_seq_len = max_seq_len # max sequence input length

        # compute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        m = torch.arange(max_seq_len, dtype=torch.float32, device=device)

        freqs = torch.outer(m, inv_freq) # outer product to calc freq for each position
        freqs = freqs.repeat_interleave(2, dim=-1) # duplicate the shared frequencies

        self.register_buffer("cos", freqs.cos())
        self.register_buffer("sin", freqs.sin())


    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Applies Rotary Position Embeddings to input tensor x
        """
        # lookup for each token position
        cos_pos = self.cos[token_positions]
        sin_pos = self.sin[token_positions]
    
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        cos = cos_pos[..., 0::2]
        sin = sin_pos[..., 0::2]
        
        # rotate the 2d vector by theta
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos
        return torch.stack((x_even_rot, x_odd_rot), dim=-1).flatten(-2)



def softmax(x: torch.Tensor, dim_i: int):
    """
    Apply softmax to dimension i
    """
    max_vals = torch.amax(x, dim=dim_i, keepdim=True)
    norm_logits = torch.exp(x - max_vals)
    norm_sum = norm_logits.sum(dim=dim_i, keepdim=True)
    return norm_logits / norm_sum