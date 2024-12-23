import sys

import torch
from torch import nn
from torch.distributions import Beta, MultivariateNormal
from torch import digamma, lgamma

def log1p_exp(x):
    return torch.log1p(torch.exp(x))

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
        nodes = []
        for j in range(2 ** self.L - 1):
            nodes.append(Node(betas[:,j], self.dim, self.device))
        #nodes = [Node(beta, self.dim, self.device) for beta in betas]
        n_nodels = len(nodes)
        # Initialize root
        nodes[0].left = nodes[1]
        nodes[0].right = nodes[2]

        for l in range(1, self.L):
            start = 2 ** l - 1
            end = 2 ** (l + 1) - 1

            for n in range(start, end):
                nodes[n].parent = nodes[(n-1) // 2]
                if (n * 2 + 1) < n_nodels:
                    nodes[n].left = nodes[n * 2 + 1]
                    nodes[n].right = nodes[n * 2 + 2]
                else:
                    nodes[n].left = None
                    nodes[n].right = None


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
        #self.shapes = nn.Parameter(torch.ones(2 ** L - 1, dim))
        #self.scales = nn.Parameter(torch.ones(2 ** L - 1, dim))
        self.shapes = nn.Parameter(torch.ones(dim, 2 ** L - 1))
        self.scales = nn.Parameter(torch.ones(dim, 2 ** L - 1))

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
        # positive constraints
        shapes, scales = log1p_exp(self.shapes), log1p_exp(self.scales)

        # Sample beta intervals
        samples = Beta(shapes, scales).rsample()


        tree = Tree(self.L, samples, self.dim, samples.device)
        lowers = torch.stack(tree.lowers, dim=1)
        uppers = torch.stack(tree.uppers, dim=1)


        # Compute indicators (n, dim, 2 ** L - 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        elif x.shape[1] != self.dim:
            x = x.transpose(2,1)

        within_lower = x > lowers.unsqueeze(0)
        within_upper = x <= uppers.unsqueeze(0)

        # Compute log likelihood Eq (13)
        #TODO: sanity check
        a_s = torch.logical_and(within_lower, within_upper)
        B = uppers - lowers

        a_s_true = torch.nonzero(a_s, as_tuple=True)
        #print(a_s_true[2].reshape(len(x), self.dim, self.L)[0,:,:])
        #print(samples.shape)
        #print(samples.unsqueeze(0).repeat(len(x), 1, 1).shape)
        while a_s_true[2].shape[0] % (self.L * self.dim) != 0:
            x_ = x + torch.rand_like(x) * 1e-8
            within_lower = x_ > lowers.unsqueeze(0)
            within_upper = x_ <= uppers.unsqueeze(0)
            a_s = torch.logical_and(within_lower, within_upper)
            a_s_true = torch.nonzero(a_s, as_tuple=True)

        # N * D
        Y_alld_alln = torch.prod(samples.unsqueeze(0).repeat(len(x), 1, 1)[torch.arange(len(x)).unsqueeze(-1).unsqueeze(-1),
            torch.arange(self.dim).unsqueeze(0).unsqueeze(-1), a_s_true[2].reshape(len(x), self.dim, self.L)], dim = -1)
        #print(Y_alld_alln.shape, Y_alld_alln[0,:])

        # N * D
        log_B = torch.log(B.unsqueeze(0).repeat(len(x), 1, 1)[torch.arange(len(x)).unsqueeze(-1),
            torch.arange(self.dim).unsqueeze(0), a_s_true[2].reshape(len(x), self.dim, self.L)[:,:,-1]])

        # N
        log_like_vec = (torch.log(Y_alld_alln) - log_B).mean(-1)
        # log_likes = []
        # for i in range(len(x)):
        #     #log_like = 0.0
        #     a_s_true = torch.nonzero(a_s[i,:,:], as_tuple=True)
        #     #print(a_s_true[1].reshape(self.dim, self.L))
        #     #sys.exit()
        #     #print(samples[torch.arange(self.dim).unsqueeze(1), a_s_true[1].reshape(self.dim, self.L)])
        #
        #     # if a_s_true[1].shape[0] != 32:
        #     #     print(a_s_true[1].shape, a_s_true[0])
        #     #     print(a_s_true[1])
        #     #     print(a_s[i,:,:].shape)
        #     #     print(a_s.shape)
        #     #     print(within_lower.shape, within_upper.shape)
        #     #     print(lowers.shape, uppers.shape)
        #     Y_alldim = torch.prod(samples[torch.arange(self.dim).unsqueeze(1), a_s_true[1].reshape(self.dim, self.L)], dim = -1)
        #     #print(Y_alldim)
        #     #sys.exit()
        #     #print(B.shape, a_s_true[0], a_s_true[1][-1])
        #     #print(a_s_true[0])
        #     #print(a_s_true[1].reshape(self.dim, self.L)[:,-1], B[torch.arange(self.dim), a_s_true[1].reshape(self.dim, self.L)[:,-1]])
        #     log_like_alldim = torch.log(Y_alldim) - torch.log(B[torch.arange(self.dim), a_s_true[1].reshape(self.dim, self.L)[:,-1]])
        #     log_like_mean = torch.mean(log_like_alldim)
        #     #print(log_like_mean)
        #
        #     #
        #     #
        #     # sys.exit()
        #
        #
        #     # for j in range(self.dim):
        #     #     a_s_true= torch.nonzero(a_s[i,j,:], as_tuple=True)[0]
        #     #     #print(a_s_true)
        #     #     #print(samples[j][a_s_true])
        #     #     #sys.exit()
        #     #     #Y = torch.prod(samples[j][a_s_true[-1]])
        #     #     Y = torch.prod(samples[j][a_s_true])
        #     #     #print(a_s_true[-1], B[j][a_s_true[-1]])
        #     #     log_like += torch.log(Y) - torch.log(B[j][a_s_true[-1]])
        #     # print(log_like/self.dim)
        #     # #sys.exit()
        #     # assert log_like_mean == log_like/self.dim, f"Assertion failed: {log_like_mean} is not equal to {log_like/self.dim}"
        #     log_likes.append(log_like_mean)
        #assert torch.stack(log_likes).mean() == log_like_vec.mean(), f"Assertion failed: {torch.stack(log_likes).mean()} is not equal to {log_like_vec.mean()}"


        #print(a_s[0,:,:])
        #print(a_s_true_1)

        #print(B)
        #print(B[0][a_s_true_1[-1]])

        #print('samples shape')
        #print(samples.shape)
            #Y = torch.prod(samples[0][a_s_true_1[-1]])
        #print(Y)

            #log_like += torch.log(Y) - torch.log(B[0][a_s_true_1[-1]])
            #print(log_like)
        return log_like_vec#torch.stack(log_likes)


        #likelihood = torch.sum((a_s * torch.log(samples)), dim=(1,2))
        #return likelihood.mean()


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
