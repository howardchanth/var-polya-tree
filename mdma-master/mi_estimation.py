# Copyright © 2021 Dar Gilboa, Ari Pakman and Thibault Vatter
# This file is part of the mdma library and licensed under the terms of the MIT license.
# For a copy, see the LICENSE file in the root directory.

import numpy as np
import matplotlib.pyplot as plt
import mdma.fit as fit
import mdma.utils as utils
import torch as t


def run_mi_estimation(d=16,
                      batch_size=500,
                      m=1000,
                      M=1000000,
                      n_reps=5,
                      save_model=False,
                      plot=True):
  Sigma = np.eye(d)
  for i in range(d):
    for j in range(d):
      if i != j:
        Sigma[i, j] = (i + j) / (5 * d)

  ind_rng = range(1, d)
  mis = []
  for i in ind_rng:
    mis += [(1 / 2) * np.log(
        np.linalg.det(Sigma[:i, :i]) * np.linalg.det(Sigma[i:, i:]) /
        np.linalg.det(Sigma))]

  all_mi_ests_all_reps = []
  for _ in range(n_reps):
    A = np.linalg.cholesky(Sigma)
    Z = np.random.randn(d, M)
    data = np.dot(A, Z).transpose()
    h = fit.get_default_h()
    h.batch_size = batch_size
    h.d = d
    h.eval_validation = False
    h.save_checkpoints = False
    h.eval_test = False
    h.m = m
    h.use_HT = 1
    h.r = 5
    h.l = 4
    h.n_epochs = 2
    h.model_to_load = ''
    h.save_path = 'experiments'
    h.M = M
    h.patience = 200
    loaders = utils.create_loaders([data, None, None], h.batch_size)
    model = fit.fit_mdma(h, loaders)
    file_name = f'mi_estimation_d:{d}_bs:{batch_size}_M:{M}_m:{m}_n_reps:{n_reps}'
    if save_model:
      model_file = file_name + '_checkpoint.pt'
      print('Saving model to ' + model_file)
      t.save({
          'model': model.state_dict(),
      }, model_file)

    print('Computing mutual information')
    all_mi_ests = []
    samples_dataloader = loaders[0]
    with t.no_grad():
      for batch_idx, batch in enumerate(samples_dataloader):
        batch = batch[0][:, 0, :]
        mi_ests = []
        for i in ind_rng:
          mi_ests += [
              t.mean(
                  model.log_density(batch) -
                  model.log_density(batch[:, range(i)], inds=range(i)) -
                  model.log_density(batch[:, range(i, d)], inds=range(i, d))).
              cpu().detach().numpy()
          ]
        all_mi_ests.append(mi_ests)
      all_mi_ests_all_reps.append([mi_ests])

      # saving
      print(f'Saving results to {file_name}')
      np.save(file_name, [mis, all_mi_ests_all_reps])
  all_mi_ests_all_reps = np.array(all_mi_ests_all_reps)

  if plot:
    plt.figure()
    plt.scatter(ind_rng, mis, label='Ground Truth')
    m, s = all_mi_ests_all_reps.mean(axis=0), all_mi_ests_all_reps.std(axis=0)
    plt.scatter(ind_rng, m, label='mdma')
    plt.errorbar(ind_rng,
                 m[0],
                 yerr=s[0],
                 color='orange',
                 ls='none',
                 capsize=5)
    plt.ylabel('$I((X_1, ..., X_k);(X_{k+1},...,X_{d}))$')
    plt.xticks(ind_rng)
    plt.xlabel('$k$')
    plt.legend()
    plt.savefig('MI_estimation.pdf')
    plt.show()
  return all_mi_ests_all_reps, mis


if __name__ == '__main__':
  all_mi_ests_all_reps, mis = run_mi_estimation()
  print('Ground truth:')
  print(mis)
  print('mdma estimates:')
  print(all_mi_ests_all_reps)
