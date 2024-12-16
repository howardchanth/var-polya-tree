import torch as t
import numpy as np

from experiments.UCI.gas import GAS
from experiments.UCI.hepmass import HEPMASS
from experiments.UCI.miniboone import MINIBOONE
from experiments.UCI.power import POWER
import argparse

from vpt.polya_tree import PolyaTree
from backbone_net import MLPS
import torch.optim as optim
import sys
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


def eval_val(backnet, model, eval_dataloader, device):
    with t.no_grad():
        val_nlls = []
        for batch_idx, batch in enumerate(eval_dataloader):
            batch_data = batch[0].to(device)
            features = backnet(batch_data)
            val_nlls.append(- model(features))
        val_nll = t.cat(val_nlls)
    return val_nll.mean()

def eval_test(backnet, model, eval_dataloader, device):
    with t.no_grad():
        log_likes = []
        for batch_idx, batch in enumerate(eval_dataloader):
            batch_data = batch[0].to(device)
            features = backnet(batch_data)
            log_likes.append(model(features))
        log_like = t.cat(log_likes)
    return log_like.mean()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Passing parameters using argparse.")

    # dataset
    parser.add_argument('--dataset', type=str, default='gas')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--data_dir', type=str, default='data/')

    # train param
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # eval and test
    parser.add_argument('--print_every', type=int, default=200)


    # seed and device
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')



    # Parse the arguments
    args = parser.parse_args()

    if args.device == 'cpu':
        device = t.device(args.device)
    else:
        device = t.device('cuda:{}'.format(args.device))

    data, n_dim, n_train, n_val = load_dataset(args)
    print('load dataset: {}, dim: {}, n train:{}, n val: {}'.format(args.dataset, n_dim, n_train, n_val))

    train_loader, valid_loader, test_loader = data

    level = 4
    back_net = MLPS(n_dim, [
        64, 64, 32, 16
        ], n_dim
    ).to(device)
    model = PolyaTree(level, n_dim).to(device)

    opt = optim.Adam([
        {'params': back_net.parameters()},
        {'params': model.parameters()},
    ], lr=args.lr, weight_decay=args.weight_decay)

    scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(opt,
                                                       verbose=True,
                                                       patience=30,
                                                       min_lr=1e-3,
                                                       factor=0.5)

    inter = 0
    for _ in range(args.epochs):
        for x in train_loader:
            opt.zero_grad()
            x = x[0].squeeze().to(device)
            # x = x.sigmoid()
            features = back_net(x)
            nll = -model(features).mean()
            nll.backward()
            opt.step()

            if inter % args.print_every == 0:
                print('Iteration {}, train nll: {}'.format(inter, nll.item()))

            inter += 1

        val_nll = eval_val(back_net, model, test_loader, device)
        print('Validation nll: {}'.format(val_nll.item()))

        test_loglike = eval_test(back_net, model, test_loader, device)
        print('Test log likelihood: {}'.format(test_loglike.item()))

        scheduler.step(val_nll)










