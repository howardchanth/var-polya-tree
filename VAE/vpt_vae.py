import math
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid


# ===================== 1. Basic 1D Pólya tree on [0,1] =====================

class PolyaTree1D(nn.Module):
    """
    Finite-depth 1D Pólya tree on [0, 1] with dyadic partitions.

    - depth L: partitions [0,1] into 2^L equal subintervals.
    - Each internal node has a Beta(alpha, alpha) branching probability for "left" vs "right".
    - Base measure is Lebesgue on [0,1], so the density is piecewise-constant.
    """

    def __init__(self, depth: int, alpha: float = 1.0,
                 dtype=torch.float32, device=None):
        super().__init__()
        assert depth >= 1, "Depth must be at least 1"
        self.depth = depth
        self.num_nodes = 2 ** depth - 1

        if device is None:
            device = torch.device("cpu")

        # Symmetric Beta(alpha, alpha) at each node
        self.register_buffer(
            "alpha_left",
            torch.full((self.num_nodes,), alpha, dtype=dtype, device=device)
        )
        self.register_buffer(
            "alpha_right",
            torch.full((self.num_nodes,), alpha, dtype=dtype, device=device)
        )

    def log_pdf(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Evaluate log density log f(x) for given tree parameters theta.

        Args:
            x: tensor of shape (B,) in [0, 1]
            theta: tensor of shape (num_nodes, 2)
        """
        assert theta.shape == (self.num_nodes, 2)
        x = x.to(theta.device)
        x = x.clamp(0.0, 1.0 - 1e-8)  # avoid x=1 hitting out-of-range bin

        B = x.shape[0]
        log_mass = torch.zeros(B, dtype=theta.dtype, device=theta.device)

        for l in range(self.depth):
            # which parent interval at level l?
            bins_l = torch.floor(x * (2 ** l)).long()   # (B,)
            node_offset = 2 ** l - 1
            node_indices = node_offset + bins_l          # (B,)

            # which child interval at level l+1?
            child_bins = torch.floor(x * (2 ** (l + 1))).long()
            branches = (child_bins - 2 * bins_l).long()  # 0 = left, 1 = right

            log_theta = torch.log(theta[node_indices, branches] + 1e-20)
            log_mass += log_theta

        # leaf length = 2^{-depth}, so density = mass * 2^depth
        log_density = log_mass + self.depth * math.log(2.0)
        return log_density


# ===================== 2. Variational Pólya tree prior =====================

class VPTPrior1D(nn.Module):
    """
    Variational Pólya tree prior over z in [0,1].
    """

    def __init__(self, depth: int = 4, alpha0: float = 1.0, device=None):
        super().__init__()
        self.tree = PolyaTree1D(depth=depth, alpha=alpha0, device=device)
        self.num_nodes = self.tree.num_nodes

        if device is None:
            device = torch.device("cpu")

        # Unconstrained params -> positive via softplus
        # Initialize near the prior: a_i, b_i ≈ alpha0
        init_val = math.log(math.exp(alpha0) - 1.0)  # inverse softplus
        self.unconstrained_a = nn.Parameter(
            torch.full((self.num_nodes,), init_val, device=device)
        )
        self.unconstrained_b = nn.Parameter(
            torch.full((self.num_nodes,), init_val, device=device)
        )

        self.alpha0 = alpha0

    def _ab_q(self):
        a = F.softplus(self.unconstrained_a) + 1e-3
        b = F.softplus(self.unconstrained_b) + 1e-3
        return a, b

    def q_theta(self):
        a, b = self._ab_q()
        return torch.distributions.Beta(a, b)

    def sample_theta(self):
        """
        Sample theta_left ~ q(theta) and form theta = [theta_left, 1 - theta_left].
        Returns:
            theta: (num_nodes, 2)
        """
        beta_q = self.q_theta()
        theta_left = beta_q.rsample()  # (num_nodes,)
        theta = torch.stack([theta_left, 1.0 - theta_left], dim=-1)
        return theta

    def kl_q_p(self):
        """
        KL[q(theta) || p(theta)] where both are product-Beta.
        Returns scalar KL.
        """
        a_q, b_q = self._ab_q()
        a0 = torch.full_like(a_q, self.alpha0)
        b0 = torch.full_like(b_q, self.alpha0)

        # KL(Beta(a_q, b_q) || Beta(a0, b0)) per node:
        # KL = log B(a0,b0) - log B(a_q,b_q)
        #      + (a_q-a0)ψ(a_q) + (b_q-b0)ψ(b_q)
        #      - (a_q+b_q-a0-b0)ψ(a_q+b_q)
        log_B_p = torch.lgamma(a0) + torch.lgamma(b0) - torch.lgamma(a0 + b0)
        log_B_q = torch.lgamma(a_q) + torch.lgamma(b_q) - torch.lgamma(a_q + b_q)

        term1 = log_B_p - log_B_q
        term2 = (a_q - a0) * torch.digamma(a_q)
        term3 = (b_q - b0) * torch.digamma(b_q)
        term4 = (a_q + b_q - a0 - b0) * torch.digamma(a_q + b_q)

        kl = term1 + term2 + term3 - term4
        return kl.sum()

    def log_prob(self, z: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Log prior density log p(z | theta) for z in [0,1].
        z: (B,)
        theta: (num_nodes, 2)
        Returns: (B,)
        """
        return self.tree.log_pdf(z, theta)


# ===================== 3. Encoder/decoder for MNIST VAE =====================

class BetaEncoder(nn.Module):
    """
    Encoder: x -> Beta(alpha(x), beta(x)) over z in [0,1].
    x is (B, 1, 28, 28).
    """

    def __init__(self, hidden_dim: int = 400):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden_dim)
        self.fc_ab = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        B = x.size(0)
        x_flat = x.view(B, -1)
        h = F.relu(self.fc1(x_flat))
        ab_uncon = self.fc_ab(h)  # (B, 2)
        alpha = F.softplus(ab_uncon[:, 0]) + 1e-3
        beta = F.softplus(ab_uncon[:, 1]) + 1e-3
        return alpha, beta


class Decoder(nn.Module):
    """
    Decoder: z in [0,1] -> Bernoulli logits over pixels.
    """

    def __init__(self, hidden_dim: int = 400):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 28 * 28)

    def forward(self, z):
        # z: (B,)  -> treat as (B,1)
        h = F.relu(self.fc1(z.unsqueeze(-1)))
        logits = self.fc_out(h)  # (B, 784)
        return logits.view(-1, 1, 28, 28)


# ===================== 4. VAE with VPT prior =====================

class VAE_VPT(nn.Module):
    """
    Very simple VAE on MNIST:
      - latent z in [0,1] (scalar) with encoder q(z|x) = Beta(alpha(x), beta(x))
      - prior p(z | theta) given by a Pólya tree density over [0,1]
      - variational posterior q(theta) over tree branch probabilities

    We estimate the ELBO by Monte Carlo:
      E_{q(theta) q(z|x)} [ log p(x|z) + log p(z|theta) - log q(z|x) ]
      - KL[q(theta)||p(theta)]
    """

    def __init__(self, depth: int = 4, alpha0: float = 1.0, hidden_dim: int = 400, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cpu")

        self.encoder = BetaEncoder(hidden_dim=hidden_dim).to(device)
        self.decoder = Decoder(hidden_dim=hidden_dim).to(device)
        self.vpt_prior = VPTPrior1D(depth=depth, alpha0=alpha0, device=device).to(device)
        self.device = device

    def forward(self, x):
        """
        One Monte Carlo estimate of ELBO per minibatch.
        Returns:
            loss (scalar), logs (dict)
        """
        x = x.to(self.device)
        B = x.size(0)

        # q(z | x)
        alpha, beta = self.encoder(x)
        qz = torch.distributions.Beta(alpha, beta)
        z = qz.rsample()  # (B,)

        # Sample tree parameters theta ~ q(theta)
        theta = self.vpt_prior.sample_theta()

        # Likelihood p(x | z) using BCE as negative log-likelihood
        logits = self.decoder(z)  # (B, 1, 28, 28)
        recon_loss = F.binary_cross_entropy_with_logits(
            logits, x, reduction="none"
        )  # (B, 1, 28, 28)
        recon_loss = recon_loss.view(B, -1).sum(dim=1)  # (B,)
        log_px_z = -recon_loss  # log p(x|z)

        # Prior term log p(z | theta)
        log_pz_theta = self.vpt_prior.log_prob(z, theta)  # (B,)

        # Entropy term -log q(z|x)
        log_qz_x = qz.log_prob(z)  # (B,)

        # KL for theta (product-Beta)
        kl_theta = self.vpt_prior.kl_q_p()  # scalar

        elbo = log_px_z + log_pz_theta - log_qz_x  # (B,)
        elbo_mean = elbo.mean()

        # Treat KL_theta as global reg; scale by batch size
        loss = -(elbo_mean - kl_theta / B)

        logs = {
            "loss": loss.item(),
            "elbo": elbo_mean.item(),
            "log_px": log_px_z.mean().item(),
            "log_pz": log_pz_theta.mean().item(),
            "log_qz": log_qz_x.mean().item(),
            "kl_theta": kl_theta.item(),
        }
        return loss, logs


# ===================== 5. Training loop on MNIST =====================

def get_mnist_test_loader(batch_size: int = 128):
    transform = transforms.ToTensor()
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    return test_loader


def train_vae_vpt(
    epochs: int = 10,
    batch_size: int = 128,
    lr: float = 1e-3,
    depth: int = 4,
    alpha0: float = 1.0,
    hidden_dim: int = 400,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ----- Data -----
    transform = transforms.ToTensor()
    train_ds = datasets.MNIST(root='~/datasets', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )

    # ----- Model & optimizer -----
    model = VAE_VPT(depth=depth, alpha0=alpha0, hidden_dim=hidden_dim, device=device)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_elbo = 0.0
        total_kl_theta = 0.0
        num_batches = 0

        for x, _ in train_loader:
            optimizer.zero_grad()
            loss, logs = model(x)
            loss.backward()
            optimizer.step()

            total_loss += logs["loss"]
            total_elbo += logs["elbo"]
            total_kl_theta += logs["kl_theta"]
            num_batches += 1

        print(
            f"Epoch {epoch:03d} | "
            f"loss={total_loss / num_batches:.3f} | "
            f"elbo={total_elbo / num_batches:.3f} | "
            f"kl_theta={total_kl_theta / num_batches:.3f}"
        )

    return model


# ===================== 6. Reconstruction & generation =====================

@torch.no_grad()
def reconstruct_test(model: VAE_VPT, device, num_images: int = 8, save_path: str = "recon_test.png"):
    model.eval()
    test_loader = get_mnist_test_loader(batch_size=num_images)
    x, _ = next(iter(test_loader))  # (B, 1, 28, 28)
    B = min(num_images, x.size(0))
    x = x[:B].to(device)

    # Use posterior mean of Beta for a deterministic reconstruction
    alpha, beta = model.encoder(x)
    z_mean = alpha / (alpha + beta)
    logits = model.decoder(z_mean)
    x_recon = torch.sigmoid(logits).cpu()

    # Build grid: originals on top, recon on bottom
    x_cpu = x.cpu()
    comparison = torch.cat([x_cpu, x_recon], dim=0)  # (2B, 1, 28, 28)
    grid = make_grid(comparison, nrow=B)
    save_image(grid, save_path)
    print(f"Saved test reconstructions to {save_path}")


@torch.no_grad()
def generate_from_tree_nodes(
    model: VAE_VPT,
    device,
    num_samples_per_node: int = 1,
    save_path: str = "gen_nodes.png",
):
    """
    Generate images by sampling z from the interval associated with each node in the Pólya tree.
    """
    model.eval()
    depth = model.vpt_prior.tree.depth

    # Build list of intervals for every node in BFS order
    node_intervals = []  # list of (left, right)
    for l in range(depth):
        num_nodes_level = 2 ** l
        for j in range(num_nodes_level):
            left = j / num_nodes_level
            right = (j + 1) / num_nodes_level
            node_intervals.append((left, right))

    zs = []
    for (left, right) in node_intervals:
        width = right - left
        # sample z uniformly in this node's interval
        u = torch.rand(num_samples_per_node, device=device)
        z_node = left + width * u  # (num_samples_per_node,)
        zs.append(z_node)

    z_tensor = torch.cat(zs, dim=0)  # (num_nodes * num_samples_per_node,)
    logits = model.decoder(z_tensor)
    x_gen = torch.sigmoid(logits).cpu()

    # Arrange grid: each row corresponds to one node, each column to a sample
    nrow = num_samples_per_node
    grid = make_grid(x_gen, nrow=nrow)
    save_image(grid, save_path)
    print(f"Saved generations from each tree node to {save_path}")


# ===================== 7. Entry point =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--depth", type=int, default=4, help="Pólya tree depth")
    parser.add_argument("--alpha0", type=float, default=1.0, help="PT Beta prior concentration")
    parser.add_argument("--hidden_dim", type=int, default=400)
    parser.add_argument("--recon_images", type=int, default=8)
    parser.add_argument("--gen_per_node", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = train_vae_vpt(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        depth=args.depth,
        alpha0=args.alpha0,
        hidden_dim=args.hidden_dim,
        device=device,
    )

    torch.save(model.state_dict(), "vae_vpt_mnist.pt")
    print("Saved model to vae_vpt_mnist.pt")

    # Reconstructions on test data
    reconstruct_test(model, device, num_images=args.recon_images, save_path="recon_test.png")

    # Generations from each tree node
    generate_from_tree_nodes(
        model,
        device,
        num_samples_per_node=args.gen_per_node,
        save_path="gen_nodes.png",
    )


if __name__ == "__main__":
    main()
