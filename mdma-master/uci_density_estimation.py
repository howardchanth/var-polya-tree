# Copyright © 2021 Dar Gilboa, Ari Pakman and Thibault Vatter
# This file is part of the mdma library and licensed under the terms of the MIT license.
# For a copy, see the LICENSE file in the root directory.

from mdma import fit
import torch as t
import numpy as np

from experiments.UCI.gas import GAS
from experiments.UCI.hepmass import HEPMASS
from experiments.UCI.miniboone import MINIBOONE
from experiments.UCI.power import POWER


def fit_uci():
  h = fit.get_default_h()
  data = load_dataset(h)
  outs = fit.fit_mdma(h, data)


def load_dataset(h):
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

  if h.missing_data_pct > 0:
    # create missing data masks
    mask = np.random.rand(*dataset.trn.x.shape) > h.missing_data_pct
    data_and_mask = np.array([dataset.trn.x, mask]).swapaxes(0, 1)
    dataset_train = t.utils.data.TensorDataset(t.tensor(data_and_mask).float())
    val_mask = np.random.rand(*dataset.val.x.shape) > h.missing_data_pct
    val_data_and_mask = np.array([dataset.val.x, val_mask]).swapaxes(0, 1)
    dataset_valid = t.utils.data.TensorDataset(
        t.tensor(val_data_and_mask).float())
  else:
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

  h.d = dataset.n_dims
  h.M = len(dataset_train)
  h.M_val = len(dataset_valid)

  return [data_loader_train, data_loader_valid, data_loader_test]


if __name__ == '__main__':
  fit_uci()
