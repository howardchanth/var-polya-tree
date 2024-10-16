import torch
from torch import nn
from torch.distributions import Beta, MultivariateNormal
from torch import digamma, lgamma


class Node:
    def __init__(self, beta, dim, device):
        self.parent = None
        self.left = None
        self.right = None

        self.lower = torch.zeros(dim).to(device)
        self.upper = torch.ones(dim).to(device)

        self.beta = beta
        self.device = device

    @property
    def length(self):
        return self.upper - self.lower


class Tree:
    def __init__(self, L, betas, dim, device):
        self.L = L
        self.dim = dim
        self.device = device

        self.nodes = self.create_nodes(betas)
        self.lowers, self.uppers = self.get_intervals()

    def create_nodes(self, betas):

        nodes = [Node(beta, self.dim, self.device) for beta in betas]

        # Initialize root
        nodes[0].left = nodes[1]
        nodes[0].right = nodes[2]

        for l in range(1, self.L):
            start = 2 ** l - 1
            end = 2 ** (l + 1) - 1

            for n in range(start, end):
                nodes[n].parent = nodes[(n-1) // 2]
                nodes[n].left = nodes[n * 2 + 1]
                nodes[n].right = nodes[n * 2 + 2]

        return nodes

    def get_intervals(self):
        for node in self.nodes:
            if node.parent is not None:
                beta = node.parent.beta
                length = node.parent.length

                if node == node.parent.left:
                    node.lower = node.parent.lower
                    node.upper = node.parent.lower + beta * length
                elif node == node.parent.right:
                    node.lower = node.parent.lower + beta * length
                    node.upper = node.parent.lower + beta * length + (1 - beta) * length

        lowers = [node.lower for node in self.nodes]
        uppers = [node.upper for node in self.nodes]

        return lowers, uppers


class PolyaTree(nn.Module):
    def __init__(self,
         L,
         dim,
         prior_shape=1,
         prior_scale=1
    ):
        super(PolyaTree, self).__init__()

        # size should be (dim, 2 ** L - 1)
        self.shapes = nn.Parameter(dim, 2 ** L - 1)
        self.scales = nn.Parameter(dim, 2 ** L - 1)

        self.prior_shape = prior_shape
        self.prior_scale = prior_scale

        self.n_betas = 2 ** L - 1

        self.L = L
        self.dim = dim

        pass

    def forward(self, x):
        """
        x: (n, dim), each entry normalized to [0,1]
        output: the log-likelihood of the model --- need to combine with object specific loss
        """
        # Sample beta intervals
        samples = Beta(self.shapes, self.scales).rsample()

        tree = Tree(self.L, samples, self.dim, samples.device)
        lowers = torch.stack(tree.lowers, dim=1)
        uppers = torch.stack(tree.uppers, dim=1)

        # Compute indicators (n, dim, 2 ** L - 1)
        within_lower = x >= lowers
        within_upper = x <= uppers

        # Compute likelihood
        a_s = torch.logical_and(within_lower, within_upper).sum(0)
        likelihood = a_s * torch.log(samples).sum

        return likelihood


    def kl(self):
        # TODO Seems the posterior part is a bit duplicated with the likelihood
        kl = 0
        kl += lgamma(self.shapes + self.scales) + lgamma(self.shapes) + lgamma(self.scales)
        kl -= lgamma(self.prior_shapes + self.prior_scale) + lgamma(self.prior_shape) + lgamma(self.prior_scale)
        kl += (self.shapes - self.prior_shape) * (digamma(self.shapes) - digamma(self.shapes + self.scales))
        kl -= (self.scales - self.prior_scale) * (digamma(self.scales) - digamma(self.shapes + self.scales))

        pass


class OptionalPolyaTree(nn.Module):
    def __init__(self, L):
        super(OptionalPolyaTree, self).__init__()
        # TODO May treat as a variant of our method
        pass

    def forward(self):
        pass

    def kl(self):
        pass
