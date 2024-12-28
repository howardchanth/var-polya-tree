from torch.func import jacrev, vmap
import torch.nn as nn
import torch
import sys
import torch.nn.functional as F

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

class sigmoid_projection(nn.Module):
    def __init__(self):
        super(sigmoid_projection, self).__init__()

    def forward(self, x):
        """
        The output tensor and the log-det-Jacobian.
        """
        # Check for NaN values
        nan_check = torch.isnan(x)
        contains_nan = nan_check.any()
        if contains_nan.item():
            print("\nnan value before sigmoid", contains_nan.item())
            sys.exit()

        output = 1 / (1 + torch.exp(-x))
        nan_check_output = torch.isnan(output)
        contains_nan_output = nan_check_output.any()
        if contains_nan_output.item():
            print("\nnan value after sigmoid")
            sys.exit()
        Jacs = output * (1 - output)
        determinant = torch.prod(Jacs, dim=-1)
        log_det_Jac = torch.log(torch.clamp(torch.abs(determinant), min=1e-5))
        return output, log_det_Jac

    def inverse(self, y):
        if torch.any(y <= 0) or torch.any(y > 1):
            raise ValueError("Input to inverse must be in the range (0, 1).")
        return torch.log(y / (1 - y + 1e-8))


