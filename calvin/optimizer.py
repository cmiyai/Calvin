from collections.abc import Callable, Iterable
from typing import Optional
import torch
import math
from torch.optim import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate.
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or 0.
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place.
                state["t"] = t + 1 # Increment iteration number.
        return loss

class CustomAdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        # Invalid parameter cases
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0 or 1: {betas}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        # Store hyperparameters in the base class's param_groups
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(CustomAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step"""
        loss = None
        if closure is not None:
            loss = closure()

        # Iterate over all parameter groups (handles different learning rates per group if applicable)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamW does not support sparse gradients')

                # self.state is a defaultdict mapping parameters to dictionaries
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values (m)
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values (v)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                t = state['step']

                # Decoupled Weight Decay
                # AdamW applies weight decay directly to the weights, rather than adding it to the gradient
                if group['weight_decay'] != 0:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Update biased first and second moment estimates
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # v_t = beta2 * v_{t-1} + (1 - beta2) * (g_t ** 2)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Compute bias-corrected moments
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                
                # To save memory, rather than allocating new tensors for m_hat and v_hat, 
                # we adjust the step size and denominator directly
                step_size = group['lr'] / bias_correction1
                
                # sqrt(v_t / bias_correction2) + eps
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                # Apply the gradient update
                # w_t = w_{t} - step_size * (m_t / denom)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

def cosine_lr_schedule(t: int, alpha_max: float, alpha_min: float, t_w: int, t_c: int) -> float:
    """
    Computes the learning rate at step t using a cosine annealing schedule with warmup.
    """
    if t < t_w:
        # Linear warm-up
        return (t / t_w) * alpha_max
    elif t <= t_c:
        # Cosine annealing
        progress = (t - t_w) / (t_c - t_w)
        return alpha_min + 0.5 * (1.0 + math.cos(progress * math.pi)) * (alpha_max - alpha_min)
    else:
        # Post-annealing constant minimum
        return alpha_min

def clip_gradients(params, max_norm: float):
    """
    Clips the gradients of the given parameters globally based on their L2-norm.
    """
    eps = 1e-6
    
    # Filter out parameters that don't have gradients
    params_with_grad = [p for p in params if p.grad is not None]
    if not params_with_grad:
        return
    
    # Compute the global L2 norm across all parameter gradients
    # We sum the squared gradients, then take the square root of the total
    total_norm_sq = 0.0
    for p in params_with_grad:
        total_norm_sq += (p.grad.detach() ** 2).sum().item()
        
    total_norm = total_norm_sq ** 0.5
    
    # If the global norm exceeds max_norm, scale all gradients down
    if total_norm > max_norm:
        scale_factor = max_norm / (total_norm + eps)
        for p in params_with_grad:
            # Modify the gradients in place
            p.grad.detach().mul_(scale_factor)