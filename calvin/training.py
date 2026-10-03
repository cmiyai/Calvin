import torch
import numpy as np

def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes cross-entropy loss: \ell_i = -log softmax(o_i)[x_{i+1}].
    
    Args:
        inputs (torch.Tensor): Logits of shape (..., vocab_size).
        targets (torch.Tensor): Ground truth labels of shape (...) with integer indices.
        
    Returns:
        torch.Tensor: Scalar tensor representing the mean loss across all batch/sequence elements.
    """
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    shifted_inputs = inputs - max_logits
    
    # Extract the shifted logit corresponding to the target class:
    # targets.unsqueeze(-1) matches the rank of shifted_inputs for gather along dim=-1
    target_logits = torch.gather(shifted_inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    # Compute log-sum-exp with cancellation:
    # -log(softmax(o)[target]) = - (shifted_target - log(sum(exp(shifted_inputs))))
    #                          = log(sum(exp(shifted_inputs))) - shifted_target
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted_inputs), dim=-1))
    
    # Loss per batch/token position
    loss = log_sum_exp - target_logits
    
    # Average across all batch and sequence dimensions
    return loss.mean()

import numpy as np

def get_batch(x: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a batch of inputs and targets from a 1D sequence of token IDs.
    """
    # torch.randint's upper bound is exclusive. 
    # If the array length is n, the maximum starting index we can pick and still 
    # have enough tokens for the target sequence (length context_length + 1) 
    # is n - context_length - 1. 
    # Therefore, the exclusive upper bound is n - context_length.
    ix = torch.randint(len(x) - context_length, (batch_size,))
    
    # Slice the numpy array for each random index.
    # cast to np.int64 before converting to tensor to ensure PyTorch gets 
    # standard long integers, which is required for embedding layers and loss functions.
    inputs = torch.stack([
        torch.from_numpy(x[i : i + context_length].astype(np.int64)) for i in ix
    ])
    
    targets = torch.stack([
        torch.from_numpy(x[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix
    ])
    
    # move the tensors to the requested device (CPU, MPS, or CUDA)
    # but standard .to() is perfectly fine here.
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    return inputs, targets

def save_checkpoint(model, optimizer, iteration, out):
    """
    Saves the model state, optimizer state, iteration
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    torch.save(checkpoint, out)

def load_checkpoint(src, model, optimizer):
    """
    Loads model checkpoint
    """
    checkpoint = torch.load(src)
    state = checkpoint["model_state_dict"]
    state = {
        (k.replace(".weights", ".weight") if k.endswith(".weights") else k): v
        for k, v in state.items()
    }
    model.load_state_dict(state)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']