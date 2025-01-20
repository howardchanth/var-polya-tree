"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
import numpy as np
import nice_pt, utils
from vpt.polya_tree import PolyaTree
import os


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
    device = torch.device(args.device)

    set_deterministic_mode(args.seed)

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
                                                  batch_size=batch_size, shuffle=True, num_workers=2)
    elif dataset == 'cifar10':
        zca = torch.load('./statistics/cifar10_zca_3.pt')
        mean = torch.load('./statistics/cifar10_mean.pt')
        transform = torchvision.transforms.Compose(
            [torchvision.transforms.RandomHorizontalFlip(p=0.5),
             torchvision.transforms.ToTensor()])
        (full_dim, mid_dim, hidden) = (3 * 32 * 32, 2000, 4)
        trainset = torchvision.datasets.CIFAR10(root='~/torch/data/CIFAR10',
                                                train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=batch_size, shuffle=True, num_workers=2)

    prior = PolyaTree(L = args.pt_level, dim = full_dim, device = device)

    result_dir = args.result_dir
    model_path = result_dir + 'model/'
    print('loading from {}'.format(model_path))
    tar_files = [f for f in os.listdir(model_path) if f.endswith('.tar')]
    model_file = model_path + tar_files[0]

    os.makedirs(result_dir + 'mix/', exist_ok=True)

    flow = nice_pt.NICE_PT(warmup_prior=None,prior=prior,
                     coupling=coupling,
                     in_out_dim=full_dim,
                     mid_dim=mid_dim,
                     hidden=hidden,
                     mask_config=mask_config, device=device, warm_up=False).to(device)

    checkpoint = torch.load(model_file)
    flow.load_state_dict(checkpoint['model_state_dict'])

      # set to inference mode
    data_0 = next(iter(trainloader))
    inputs_0, _ = data_0
    inputs_0 = utils.prepare_data(
        inputs_0, dataset, zca=zca, mean=mean).to(device)

    data_1 = next(iter(trainloader))
    inputs_1, _ = data_1
    inputs_1 = utils.prepare_data(
        inputs_1, dataset, zca=zca, mean=mean).to(device)

    flow.eval()
    with torch.no_grad():
        for j in range(batch_size):
            input_0 = inputs_0[j].unsqueeze(0)
            input_1 = inputs_1[j].unsqueeze(0)
            z_0, _,_ = flow.f(input_0)
            z_0 = z_0.to(device)

            z_1, _,_ = flow.f(input_1)
            z_1 = z_1.to(device)

            reconst_mix =[]
            for i in range(11):
                z_mix = z_1 *(1 - i * 0.1) + z_0 * i * 0.1

                reconst = flow.g(z_mix).cpu().squeeze()
                reconst_mix.append(reconst)
            reconst_stack = torch.stack(reconst_mix, dim=0)
            reconst_plot = utils.prepare_data(
                reconst_stack, dataset, zca=zca, mean=mean, reverse=True)
            torchvision.utils.save_image(torchvision.utils.make_grid(reconst_plot, nrow=11),
                                         result_dir + '/mix/'+ 'test_{}.png'.format(j) )



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot vpt generates PyTorch implementation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=200)
    parser.add_argument('--pt_level', default=4, type=int)
    parser.add_argument('--device', default='cpu', help='cuda or cpu.')
    parser.add_argument('--result_dir', type=str)
    args = parser.parse_args()
    main(args)
