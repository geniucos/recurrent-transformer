# import torch
# import torch.nn as nn
# import torch.optim as optim

# from itertools import chain

# # Parts of the code are modifications of Pytorch's AdamW optimizer
# # Parts of the code are modifications of code from https://github.com/jiaweizzhao/GaLore/blob/master/galore_torch/galore_projector.py


# class RotatedBasisOptimizer(optim.Optimizer):
#     """
#     Implements SOAP-style fixed random left/right rotations for 2D parameters only.
#     For 2D W: rotate gradients as G_rot = QL^T * G * QR, do Adam in rotated space,
#     then project back: U = QL * (AdamUpdateRot) * QR^T.

#     Parameters:
#         params (`Iterable[nn.parameter.Parameter]`):
#             Iterable of parameters to optimize or dictionaries defining parameter groups.
#         lr (`float`, *optional*, defaults to 0.003):
#             The learning rate to use.
#         betas (`Tuple[float,float]`, *optional*, defaults to `(0.95, 0.95)`):
#             Adam's betas parameters (b1, b2).
#         eps (`float`, *optional*, defaults to 1e-08):
#             Adam's epsilon for numerical stability.
#         weight_decay (`float`, *optional*, defaults to 0.01): weight decay coefficient.
#         max_precond_dim (`int`, *optional*, defaults to 10000):
#             Maximum dimension of the rotation on each side; if a side exceeds this, skip rotation on that side.
#         merge_dims (`bool`, *optional*, defaults to `False`):
#             Kept for API compatibility; not used by rotation path.
#         normalize_grads (`bool`, *optional*, defaults to `False`):
#             Whether or not to normalize gradients per layer (applied after projecting back).
#         data_format (`str`, *optional*, defaults to `channels_first`):
#             Kept for API compatibility.
#         correct_bias (`bool`, *optional*, defaults to `True`):
#             Whether or not to use bias correction in Adam.
#     """

#     def __init__(
#         self,
#         params,
#         lr: float = 3e-3,
#         betas=(0.95, 0.95),
#         eps: float = 1e-8,
#         weight_decay: float = 0.01,
#         max_precond_dim: int = 10000,
#         merge_dims: bool = False,
#         normalize_grads: bool = False,
#         data_format: str = "channels_first",
#         correct_bias: bool = True,
#     ):
#         defaults = {
#             "lr": lr,
#             "betas": betas,
#             "eps": eps,
#             "weight_decay": weight_decay,
#             "max_precond_dim": max_precond_dim,
#             "merge_dims": merge_dims,
#             "normalize_grads": normalize_grads,
#             "correct_bias": correct_bias,
#         }
#         super().__init__(params, defaults)
#         self._data_format = data_format
        
#     def merge_dims(self, grad, max_precond_dim):
#         """
#         Merges dimensions of the gradient tensor till the product of the dimensions is <= max_precond_dim.
#         (Kept from original; not used by the rotation path.)
#         """
#         assert self._data_format in ["channels_first", "channels_last"]
#         if self._data_format == "channels_last" and grad.dim() == 4:
#             grad = grad.permute(0, 3, 1, 2)
#         shape = grad.shape
#         new_shape = []
        
#         curr_shape = 1
#         for sh in shape:
#             temp_shape = curr_shape * sh
#             if temp_shape > max_precond_dim:
#                 if curr_shape > 1:
#                     new_shape.append(curr_shape)
#                     curr_shape = sh
#                 else:
#                     new_shape.append(sh)
#                     curr_shape = 1
#             else:
#                 curr_shape = temp_shape
        
#         if curr_shape > 1 or len(new_shape) == 0:
#             new_shape.append(curr_shape)
        
#         new_grad = grad.reshape(new_shape)
#         return new_grad               

#     @torch.no_grad()
#     def step(self, closure=None):
#         """
#         Performs a single optimization step.
#         """
#         loss = None if closure is None else closure()
        
#         for group in self.param_groups:
#             for p in group["params"]:
#                 if p.grad is None:
#                     continue
#                 grad = p.grad

#                 state = self.state[p]
                
#                 if "step" not in state:
#                     state["step"] = 0 
                    
#                 # Initialize on first sight: build rotations (for 2D only), size EMA in rotated space
#                 if 'Q' not in state:
#                     self.init_preconditioner(
#                         grad,
#                         state,
#                         max_precond_dim=group['max_precond_dim'],
#                         merge_dims=group["merge_dims"],
#                     )
#                     g0 = self.project(grad, state, merge_dims=group["merge_dims"], 
#                                       max_precond_dim=group['max_precond_dim'])
#                     state["exp_avg"] = torch.zeros_like(g0)
#                     state["exp_avg_sq"] = torch.zeros_like(g0)
#                     # Skip this step so we never mix current grad into freshly init buffers
#                     continue 
                
#                 # Project gradients into fixed basis (no-op for non-2D or skipped sides)
#                 grad_projected = self.project(grad, state, merge_dims=group["merge_dims"], 
#                                               max_precond_dim=group['max_precond_dim'])

#                 exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
#                 beta1, beta2 = group["betas"]

#                 state["step"] += 1

#                 # Adam moments in the rotated space (or original if not rotated)
#                 exp_avg.mul_(beta1).add_(grad_projected, alpha=(1.0 - beta1))
#                 exp_avg_sq.mul_(beta2).add_(grad_projected.square(), alpha=(1.0 - beta2))

#                 denom = exp_avg_sq.sqrt().add_(group["eps"])
                
#                 # Already in projected space
#                 exp_avg_projected = exp_avg
                
#                 step_size = group["lr"]
#                 if group["correct_bias"]:
#                     bias_correction1 = 1.0 - beta1 ** (state["step"])
#                     bias_correction2 = 1.0 - beta2 ** (state["step"])
#                     step_size = step_size * (bias_correction2 ** .5) / bias_correction1

#                 # Project back to original space
#                 norm_grad = self.project_back(exp_avg_projected / denom, state, merge_dims=group["merge_dims"],
#                                               max_precond_dim=group['max_precond_dim'])

#                 if group["normalize_grads"]:
#                     norm_grad = norm_grad / (1e-30 + torch.mean(norm_grad**2).sqrt())
                
#                 p.add_(norm_grad, alpha=-step_size)
                
#                 # AdamW decoupled weight decay
#                 if group["weight_decay"] > 0.0:
#                     p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))
        
#         return loss
    
#     def init_preconditioner(self, grad, state, max_precond_dim=10000, merge_dims=False):
#         """
#         Initialize fixed left/right orthonormal rotation matrices per parameter.
#         - Only for 2D tensors (out, in). 
#         - If a side's dim > max_precond_dim, skip rotation on that side (identity).
#         - Ignore 1D and conv/other ranks entirely (no rotation).
#         """
#         device = grad.device
#         dtype = grad.dtype
#         dim = grad.dim()

#         def maybe_Q(n):
#             if n <= 1 or n > max_precond_dim:
#                 return None  # identity (skip rotation on this side)
#             return self.random_orthonormal(n, n, device=device, dtype=dtype)

#         # Defaults: no rotation
#         state['QL'] = None
#         state['QR'] = None
#         state['kind'] = 'none'

#         if dim == 2:
#             out, inn = grad.shape
#             state['QL'] = maybe_Q(out)
#             state['QR'] = maybe_Q(inn)
#             state['kind'] = '2d'
#         # else: kind stays 'none' (ignore 1D, conv, etc.)

#         # Marker so step() knows we've initialized this param
#         state['Q'] = True
        
#     def project(self, grad, state, merge_dims=False, max_precond_dim=10000):
#         """
#         Project gradient to rotated space:
#           - 2D: g' = QL^T g QR  (skipping any side with None)
#           - else: identity (ignored)
#         """
#         kind = state.get('kind', 'none')
#         QL = state.get('QL', None)
#         QR = state.get('QR', None)

#         if kind == '2d':
#             g = grad
#             if QL is not None:
#                 g = torch.tensordot(QL.T, g, dims=[[1], [0]])  # (out, in)
#             if QR is not None:
#                 g = torch.tensordot(g, QR, dims=[[1], [0]])    # (out, in)
#             return g

#         # No rotation
#         return grad

#     def project_back(self, grad, state, merge_dims=False, max_precond_dim=10000):
#         """
#         Project update back to original space:
#           - 2D: u = QL g' QR^T  (skipping any side with None)
#           - else: identity
#         """
#         kind = state.get('kind', 'none')
#         QL = state.get('QL', None)
#         QR = state.get('QR', None)

#         if kind == '2d':
#             u = grad
#             if QL is not None:
#                 u = torch.tensordot(QL, u, dims=[[1], [0]])
#             if QR is not None:
#                 u = torch.tensordot(u, QR.T, dims=[[1], [0]])
#             return u

#         # No rotation
#         return grad
        
#     def random_orthonormal(self, n, k, device=None, dtype=None):
#         X = torch.randn(n, k, device=device, dtype=dtype)
#         Q, _ = torch.linalg.qr(X, mode='reduced')
#         return Q
