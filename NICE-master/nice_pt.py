"""Utility classes for NICE.
"""

import torch
import torch.nn as nn

from vpt.utils import sigmoid_projection

"""Additive coupling layer.
"""
class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(
            nn.Linear(in_out_dim//2, mid_dim),
            nn.ReLU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim//2)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        if reverse:
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W))

"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(self.scale)
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_J

"""NICE main model.
"""
class NICE_PT(nn.Module):
    def __init__(self, warmup_prior, prior, coupling,
        in_out_dim, mid_dim, hidden, mask_config, warm_up, device):
        """Initialize a NICE.

        Args:
            prior: prior distribution over latent space Z.
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(NICE_PT, self).__init__()
        self.warmup_prior = warmup_prior
        self.prior = prior
        self.in_out_dim = in_out_dim

        self.coupling = nn.ModuleList([
            Coupling(in_out_dim=in_out_dim,
                     mid_dim=mid_dim,
                     hidden=hidden,
                     mask_config=(mask_config+i)%2) \
            for i in range(coupling)])
        self.scaling = Scaling(in_out_dim).to(device)
        self.sigmoid_projection = sigmoid_projection().to(device)
        self.device = device
        self.warm_up = warm_up

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        if self.warm_up:
            x = z
        else:
            x = self.sigmoid_projection.inverse(z)
        x, _ = self.scaling(x, reverse=True)

        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        for i in range(len(self.coupling)):
            # Check for NaN values
            nan_check = torch.isnan(x)
            contains_nan = nan_check.any()
            if contains_nan.item():
                print("\nnan value before {}".format(i))
                sys.exit()
            x = self.coupling[i](x)
        # Check for NaN values
        nan_check = torch.isnan(x)
        contains_nan = nan_check.any()
        if contains_nan.item():
            print("\nnan value before scaling", contains_nan.item())
            sys.exit()
        x, log_scale = self.scaling(x)
        if self.warm_up:
            return x, log_scale
        else:
            x, log_sigmoid = self.sigmoid_projection(x)
            return x, log_scale, log_sigmoid

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        if self.warm_up:
            z, log_scale = self.f(x)
            log_ll = torch.sum(self.warmup_prior.log_prob(z), dim=1)
            return log_ll + log_scale
        else:
            z, log_scale, log_sigmoid = self.f(x)
            log_ll = self.prior(z)
            return log_ll + log_scale + log_sigmoid

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        if self.warm_up:
            z = self.warmup_prior.sample((size, self.in_out_dim)).to(self.device)
        else:
            z = self.prior.sample(size).to(self.device)
        return self.g(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)

    def end_warmup(self):
        self.warm_up = False
        for param in self.coupling.parameters():
            param.requires_grad = False
        for param in self.scaling.parameters():
            param.requires_grad = False