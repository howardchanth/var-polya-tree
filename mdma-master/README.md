# MDMA

PyTorch implementation of the Marginalizable Density Model Approximator — a density estimator that provides closed-form marginal and conditional densities and enables rapid sampling. 

For details, see: 
> Dar Gilboa, Ari Pakman and Thibault Vatter, [Marginalizable Density Models](https://arxiv.org/abs/2106.04741) (2021)

## Requirements

- **`python>=3.6`** 
- **`numpy>=1.20.2`** 
- **`pytorch>=1.0.0`**
- **`pandas>=1.2.3`**

Optional for visualization and plotting: `matplotlib` and `tensorboardX`.

## Structure

- mdma/models.py:   Implementation of the MDMA class.
- mdma/fit.py:      Fitting an MDMA model.
- mdma/utils.py:    Various auxiliary functions.
- experiments:      Additional code for reproducing the experiments in the paper.

## Usage

Below, example commands are given for running experiments.

#### Download datasets

Download UCI datasets:

```
bash download_datasets.sh
```

#### Toy 3D density estimation

Density estimation on a toy dataset of two spirals, showing the ability of MDMA to compute marginal and conditional densities.

Fit two spirals density using MDMA, and plot marginals and conditionals:

```
python3 toy_density_estimation.py --dataset spirals
```

Possible values for `dataset` are `spirals`, `checkerboard`, `gaussians`.

For a two spiral dataset, the samples and marginal histograms of the data take the following form:
![Data](media/s1.jpg)
Samples from the trained MDMA model and the learned marginal densities evaluated on a grid are indistinguishable:
![Samples and marginals](media/s2.jpg)
MDMA also provides closed-form expression for all conditional densities:
![Conditionals](media/s4.jpg?s=100)

#### UCI density estimation

Fit UCI POWER dataset using MDMA:

```
python3 uci_density_estimation.py --dataset power \
                                  --m 1000 \           # Width of tensor network
                                  --r 3 \              # Width of univariate CDF networks
                                  --l 2 \              # Depth of univariate CDF networks
                                  --batch_size 500 \
                                  --n_epochs 1000 \
                                  --lr 0.01 
```

Possible values for `dataset` are `power`, `gas`, `hepmass`, `miniboone`.

Fit UCI POWER dataset using the non-marginalizable variant nMDMA:

```
python3 uci_density_estimation.py --dataset power \
                                  --m 1000 \           # Width of tensor network
                                  --r 3 \              # Width of univariate CDF networks
                                  --l 2 \              # Depth of univariate CDF networks
                                  --batch_size 500 \
                                  --n_epochs 1000 \
                                  --lr 0.01 \
                                  --mix_vars 1 \       # Use nMDMA
                                  --n_mix_terms 5\     # Number of diagonals in the mixing matrix T
```

#### Density estimation with missing values

Fit UCI POWER dataset with 0.5 probability of missing values per entry using MDMA:

```
python3 uci_density_estimation.py --dataset gas \
                                  --m 4000 \
                                  --r 5 \
                                  --l 4 \
                                  --batch_size 500 \
                                  --n_epochs 1000 \
                                  --lr 0.01 \
                                  --missing_data_pct 0.5 # proportion of missing values
```

Density estimation using BNAF on the same dataset after performing MICE imputation:

```
python3 bnaf_density_estimation.py --dataset gas \
                                   --hidden_dim 320 \
                                   --missing_data_pct 0.5 \
                                   --missing_data_strategy mice
```

Requires (for imputation):

- **`miceforest>=2.0.4`**
- **`scikit-learn>=0.24.2`** 

#### Mutual information estimation

Generate data from a multivariate Gaussian, fit the joint density using MDMA and estimate the mutual information between subsets of variables:

```
python3 mi_estimation.py
```

#### Causal discovery

Run the causal discovery experiment, recovering a causal graph from data by testing for conditional independence using MDMA:

```
python3 causal_discovery.py --dataset "sachs" \
                            --lr .1 \
                            --r 3 \
                            --l 2 \
                            --m 1000 \
                            --batch_size 500 \
                            --patience 100 \
                            --n_epochs 50 \
                            --verbose 1 \
                            --save_checkpoints 0

```

Requires:

- **`R>=4.0.5`** 
  and R packages
  - **`pcalg>=2.7-3`** (on CRAN)
  - **`graph>=1.70.0`** (on Bioconductor)
  - **`RBGL>=1.68.0`** (on Bioconductor) 
  - **`graph>=1.70.0`** (on Bioconductor) 

as well as the python packages

- **`cdt>=0.5.23`** 
- **`networkx>=2.5.1`** 
- **`rpy2>=3.4.4`**

#### Sampling

The time complexity of sampling from MDMA is logarithmic in the input dimension. 

Generate `S` samples from a trained model:
```
samples = model.sample(S)
```

## Citation

```
@misc{gilboa2021marginalizable,
      title={Marginalizable Density Models}, 
      author={Dar Gilboa and Ari Pakman and Thibault Vatter},
      year={2021},
      eprint={2106.04741},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```
