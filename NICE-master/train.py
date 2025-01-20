"""Training procedure for NICE.
"""

import argparse
import torch, torchvision
import numpy as np
import nice, utils
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
    latent = args.latent
    sample_size = args.sample_size
    coupling = 4
    mask_config = 1.

    # optimization hyperparameters
    lr = args.lr
    momentum = args.momentum
    decay = args.decay

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
        max_iter = (len(trainset)// batch_size + 1) * args.epoch
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
         torchvisitransforms.ToTensor()])
        (full_dim, mid_dim, hidden) = (3 * 32 * 32, 2000, 4)
        trainset = torchvision.datasets.CIFAR10(root='~/torch/data/CIFAR10',
            train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
            batch_size=batch_size, shuffle=True, num_workers=2)
     
    if latent == 'normal':
        prior = torch.distributions.Normal(
            torch.tensor(0.).to(device), torch.tensor(1.).to(device))
    elif latent == 'logistic':
        prior = utils.StandardLogistic()

    filename = '%s_' % dataset \
             + 'bs%d_' % batch_size \
             + '%s_' % latent \
             + 'cp%d_' % coupling \
             + 'md%d_' % mid_dim \
             + 'hd%d_' % hidden \
             + 'seed%d' % args.seed

    result_dir = os.path.join('results', filename)
    os.makedirs(result_dir + '/reconstruction/', exist_ok=True)
    os.makedirs(result_dir + '/samples/', exist_ok=True)
    if args.savemodel:
        os.makedirs(result_dir + '/model/', exist_ok=True)

    flow = nice.NICE(prior=prior, 
                coupling=coupling, 
                in_out_dim=full_dim, 
                mid_dim=mid_dim, 
                hidden=hidden, 
                mask_config=mask_config, device = device).to(device)
    optimizer = torch.optim.Adam(
        flow.parameters(), lr=lr, betas=(momentum, decay), eps=1e-4)

    total_iter = 0
    train = True
    running_loss = 0

    while train:
        for _, data in enumerate(trainloader, 1):
            flow.train()    # set to training mode
            if total_iter == max_iter:
                train = False
                break

            total_iter += 1
            optimizer.zero_grad()    # clear gradient tensors

            inputs, _ = data
            inputs = utils.prepare_data(
                inputs, dataset, zca=zca, mean=mean).to(device)

            # log-likelihood of input minibatch
            loss = -flow(inputs).mean()
            running_loss += float(loss)

            # backprop and update parameters
            loss.backward()
            optimizer.step()

            if total_iter % 1000 == 0:
                mean_loss = running_loss / 1000
                bit_per_dim = (mean_loss + np.log(256.) * full_dim) \
                            / (full_dim * np.log(2.))
                print('iter %s:' % total_iter, 
                    'loss = %.3f' % mean_loss, 
                    'bits/dim = %.3f' % bit_per_dim)
                running_loss = 0.0

                flow.eval()        # set to inference mode
                with torch.no_grad():
                    z, _ = flow.f(inputs)
                    z = z.to(device)
                    reconst = flow.g(z).cpu()
                    reconst = utils.prepare_data(
                        reconst, dataset, zca=zca, mean=mean, reverse=True)
                    samples = flow.sample(sample_size).cpu()
                    samples = utils.prepare_data(
                        samples, dataset, zca=zca, mean=mean, reverse=True)
                    torchvision.utils.save_image(torchvision.utils.make_grid(reconst),
                        result_dir + '/reconstruction/' + filename +'iter%d.png' % total_iter)
                    torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                        result_dir + '/samples/' + filename +'iter%d.png' % total_iter)

    print('Finished training! Start evaluating...')

    test_loss = 0.0
    test_inter = 0
    for _, data in enumerate(testloader, 1):
        flow.eval()  # set to training mode

        with torch.no_grad():
            inputs, _ = data
            inputs = utils.prepare_data(
                inputs, dataset, zca=zca, mean=mean).to(device)

            # log-likelihood of input minibatch
            loss = -flow(inputs).mean()
            test_loss += float(loss)
        test_inter += 1

    mean_test_loss = test_loss / test_inter
    test_bit_per_dim = (mean_test_loss + np.log(256.) * full_dim) \
                  / (full_dim * np.log(2.))

    print('Test Bit Per Dim: %.3f' % test_bit_per_dim)

    with open(os.path.join(result_dir, "results.txt"), "a") as f:
        print("Train BDP: {:4.3f}".format(bit_per_dim), file=f)
        print("Test BDP:  {:4.3f}".format(test_bit_per_dim), file=f)

    if args.savemodel:
        torch.save({
            'total_iter': total_iter,
            'model_state_dict': flow.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dataset': dataset,
            'batch_size': batch_size,
            'latent': latent,
            'coupling': coupling,
            'mid_dim': mid_dim,
            'hidden': hidden,
            'mask_config': mask_config},
            result_dir + '/model/' + filename +'iter%d.tar' % total_iter)

        print('Checkpoint Saved')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('MNIST NICE PyTorch implementation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=200)
    parser.add_argument('--latent',
                        help='latent distribution.',
                        type=str,
                        default='normal')
    parser.add_argument('--epoch',
                        type=int,
                        default=1500)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=1e-3)
    parser.add_argument('--momentum',
                        help='beta1 in Adam optimizer.',
                        type=float,
                        default=0.9)
    parser.add_argument('--decay',
                        help='beta2 in Adam optimizer.',
                        type=float,
                        default=0.999)
    parser.add_argument('--device', default='cpu', help='cuda or cpu.')
    parser.add_argument('--savemodel', action="store_true")
    args = parser.parse_args()
    main(args)

# bit-per-dim computation for NICE: https://stats.stackexchange.com/questions/423120/what-is-bits-per-dimension-bits-dim-exactly-in-pixel-cnn-papers