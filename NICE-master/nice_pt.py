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
            nn.Softplus())
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.Softplus()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, in_out_dim//2)

    def reset_nan_parameters(self):
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                # print(f"Resetting {name} due to NaN values.")
                nn.init.uniform_(param)

    def forward(self, inp, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """

        self.reset_nan_parameters()  # Reset nan parameters if any
        inp = inp.clamp(max=torch.max(inp[~torch.isinf(inp)]))

        [B, W] = list(inp.size())
        x = inp.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        off_ = off_.clamp(max=torch.max(off_[~torch.isinf(off_)]))

        assert not torch.isnan(off_).any()
        assert not torch.isinf(off_).any()

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

        assert not torch.isnan(x).any()

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

    def reset_scale(self):
        if torch.isnan(self.scale).any():
            nn.init.constant_(self.scale, torch.nanmean(self.scale))

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """

        self.reset_scale()

        log_det_J = torch.sum(self.scale)
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_J

"""NICE main model.
"""
class NICE_PT(nn.Module):
    def __init__(self, prior, coupling,
        in_out_dim, mid_dim, hidden, mask_config, device):
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

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        z = z.clamp(max=1-1e-5)
        x = self.sigmoid_projection.inverse(z)
        x, _ = self.scaling(x, reverse=True)

        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)

        assert not torch.isnan(x).any()

        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        for i in range(len(self.coupling)):
            x = self.coupling[i](x)
            x = nn.functional.normalize(x, dim=1)
            x = x.clamp(max=torch.max(x[~torch.isinf(x)]))

        x_ = x

        x, log_scale = self.scaling(x_)

        assert not torch.isnan(x).any()
        assert not torch.isinf(x).any()

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
        z, log_scale, log_sigmoid = self.f(x)
        log_ll = self.prior(z)
        #log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        #print(log_ll.mean(), log_scale.mean(), log_sigmoid.mean())
        return log_ll + log_scale + log_sigmoid

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
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