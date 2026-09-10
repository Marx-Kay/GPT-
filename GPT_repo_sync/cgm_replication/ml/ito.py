"""Duarte, Duarte & Silva (2024), Proposition 1, Eq. (8)-(9).
No state Hessian is materialized. f and g are coefficients evaluated at s:
do not recompute them at the epsilon-displaced states.
"""
import math
import torch
from torch.func import jvp

def ito_generator(value, state, drift, diffusion):
    """state, drift: [batch,d]; diffusion: [batch,d,m]; value: [batch,d]->[batch]."""
    batch, dim = state.shape
    m = diffusion.shape[-1]
    if m == 0:
        return jvp(value, (state,), (drift,))[1]
    def auxiliary(epsilon):
        moved = (state[:, None, :] + epsilon * diffusion.transpose(1, 2) / math.sqrt(2)
                 + epsilon.square() * drift[:, None, :] / (2*m))
        return value(moved.reshape(batch*m, dim)).reshape(batch, m).sum(1)
    zero = state.new_zeros(())
    one = state.new_ones(())
    def first(epsilon):
        return jvp(auxiliary, (epsilon,), (one,))[1]
    return jvp(first, (zero,), (one,))[1]
