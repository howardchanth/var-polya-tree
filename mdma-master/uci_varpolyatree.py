import torch as t
import numpy as np

from experiments.UCI.gas import GAS
from experiments.UCI.hepmass import HEPMASS
from experiments.UCI.miniboone import MINIBOONE
from experiments.UCI.power import POWER
import argparse

from vpt.polya_tree import PolyaTree
from backbone_net import MLP
import torch.optim as optim

def load_dataset(h):
    h.dataset = 'gas'
    if h.dataset == 'gas':
        dataset = GAS(h.data_dir + '/gas/ethylene_CO.pickle')
    elif h.dataset == 'hepmass':
        dataset = HEPMASS(h.data_dir + '/hepmass')
    elif h.dataset == 'miniboone':
        dataset = MINIBOONE(h.data_dir + '/miniboone/data.npy')
    elif h.dataset == 'power':
        dataset = POWER(h.data_dir + '/power/data.npy')
    else:
        raise RuntimeError()


    dataset_train = t.utils.data.TensorDataset(
        t.tensor(np.expand_dims(dataset.trn.x, 1)).float())
    dataset_valid = t.utils.data.TensorDataset(
        t.tensor(np.expand_dims(dataset.val.x, 1)).float())

    data_loader_train = t.utils.data.DataLoader(dataset_train,
                                              batch_size=h.batch_size,
                                              shuffle=True)

    data_loader_valid = t.utils.data.DataLoader(dataset_valid,
                                              batch_size=h.batch_size,
                                              shuffle=False)

    dataset_test = t.utils.data.TensorDataset(t.tensor(dataset.tst.x).float())
    data_loader_test = t.utils.data.DataLoader(dataset_test,
                                             batch_size=h.batch_size,
                                             shuffle=False)


    return [data_loader_train, data_loader_valid, data_loader_test], dataset.n_dims, len(dataset_train), len(dataset_valid)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passing parameters using argparse.")

    # dataset
    parser.add_argument('--dataset', type=str, default='gas')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--data_dir', type=str, default='data/')

    # train param
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # Parse the arguments
    args = parser.parse_args()

    data, n_dim, n_train, n_val = load_dataset(args)
    print('load dataset: {}, dim: {}, n train:{}, n val: {}'.format(args.dataset, n_dim, n_train, n_val))

    train_loader, valid_loader, test_loader = data

    level = 4
    back_net = MLP(n_dim, [100, 100], n_dim)
    model = PolyaTree(level, n_dim)

    opt = optim.Adam([
        {'params': back_net.parameters()},
        {'params': model.parameters()},
    ], lr=args.lr, weight_decay=args.weight_decay)

    for _ in range(args.epochs):
        for x in train_loader:
            opt.zero_grad()
            x = x[0]
            features = back_net(x)
            likelihood = model(features)
            likelihood.backward()
            opt.step()
            print(likelihood.item())











