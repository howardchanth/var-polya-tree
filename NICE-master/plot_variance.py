"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
import numpy as np
from matplotlib import pyplot as plt

import nice_pt, utils
from vpt.polya_tree import PolyaTree
import os

import seaborn as sns

def set_deterministic_mode(seed):
    # Set random seed for numpy
    np.random.seed(seed)
    # Set random seed for PyTorch
    torch.manual_seed(seed)
    # If using CUDA, set the seed for all GPUs
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Make CUDNN deterministic
        torch.backends.cudnn.deterministic = True
        # Disable benchmark to ensure deterministic behavior
        torch.backends.cudnn.benchmark = False

def main(args):
    # device = torch.device("cuda:0")
    device = torch.device(args.device)

    # model hyperparameters
    dataset = args.dataset
    batch_size = args.batch_size

    coupling = 4
    mask_config = 1.



    zca = None
    mean = None
    if dataset == 'mnist':
        mean = torch.load('./statistics/mnist_mean.pt')
        (full_dim, mid_dim, hidden) = (1 * 28 * 28, 1000, 5)
        transform = torchvision.transforms.ToTensor()
        trainset = torchvision.datasets.MNIST(root='~/torch/data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='~/torch/data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    elif dataset == 'fashion-mnist':
        mean = torch.load('./statistics/fashion_mnist_mean.pt')
        (full_dim, mid_dim, hidden) = (1 * 28 * 28, 1000, 5)
        transform = torchvision.transforms.ToTensor()
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size, shuffle=True, num_workers=2)
    elif dataset == 'svhn':
        zca = torch.load('./statistics/svhn_zca_3.pt')
        mean = torch.load('./statistics/svhn_mean.pt')
        (full_dim, mid_dim, hidden) = (3 * 32 * 32, 2000, 4)
        transform = torchvision.transforms.ToTensor()
        trainset = torchvision.datasets.SVHN(root='~/torch/data/SVHN',
                                             split='train', download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size, shuffle=True, num_workers=4)
        testset = torchvision.datasets.MNIST(root='~/torch/data/SVHN',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    elif dataset == 'cifar10':
        zca = torch.load('./statistics/cifar10_zca_3.pt')
        zca_logdet = torch.linalg.slogdet(zca)[1]
        mean = torch.load('./statistics/cifar10_mean.pt')
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomHorizontalFlip(p=0.5),
             torchvision.transforms.ToTensor()])
        (full_dim, mid_dim, hidden) = (3 * 32 * 32, 2000, 4)
        trainset = torchvision.datasets.CIFAR10(root='~/torch/data/CIFAR10',
                                                train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size, shuffle=True, num_workers=4)
        testset = torchvision.datasets.MNIST(root='~/torch/data/CIFAR10',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    prior = PolyaTree(L = args.pt_level, dim = full_dim, device = device)

    result_dir = args.result_dir
    model_path = result_dir + 'model/'
    print('loading from {}'.format(model_path))
    tar_files = [f for f in os.listdir(model_path) if f.endswith('.tar')]
    model_file = model_path + tar_files[0]

    flow = nice_pt.NICE_PT(warmup_prior=None,
                     prior=prior,
                     coupling=coupling,
                     in_out_dim=full_dim,
                     mid_dim=mid_dim,
                     hidden=hidden,
                     mask_config=mask_config, warm_up = True, device=device).to(device)

    checkpoint = torch.load(model_file)
    flow.load_state_dict(checkpoint['model_state_dict'])

    with torch.no_grad():
        var = flow.prior.get_variance()
        var_np = var.cpu().numpy().reshape(28, 28)
        print(var_np)

    plt.figure(figsize=(8, 6))
    plt.axis('off')
    sns.heatmap(var_np, cmap='Reds', cbar=True)
    os.makedirs(result_dir + '/var/', exist_ok=True)
    plt.savefig(result_dir + '/var/' + 'var.png', dpi=300)
    print('img saved')












if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot vpt variance PyTorch implementation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=50)
    parser.add_argument('--pt_level', default=4, type=int)
    parser.add_argument('--device', default='cpu', help='cuda or cpu.')
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()
    main(args)
