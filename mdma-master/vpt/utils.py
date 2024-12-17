from torch.func import jacrev, vmap
import torch
#import torch.nn.functional as F

# D = 16
#
# def model(x):
#     weight = torch.ones(D, D) * 2
#     bias = torch.ones(D)
#     return F.linear(x, weight, bias).tanh()

def log_det_jacobian(model, x):
    jacobian = vmap(jacrev(model))(x)
    signs, log_det_jacobian = torch.linalg.slogdet(jacobian)
    return log_det_jacobian


# x = torch.randn(64, D)
# log_det_jacobian = log_det_jacobian(model, x)
#
# print(log_det_jacobian.shape)