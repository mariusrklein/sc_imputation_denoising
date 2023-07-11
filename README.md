# Single-cell imputation and denoising

This repository contains the corresponding code for the Master's Thesis ["Overcoming Sparsity and Technical Noise: Systematic Evaluation of Imputation and Denoising Methods for Single-Cell Metabolomics and Lipidomics Data"](https://github.com/mariusrklein/sc_imputation_denoising/blob/main/thesis_analysis/Thesis.pdf) by Marius Klein and additional functions for the future analysis of single-cell metabolomics and lipidomics data.

## Installation

The Python package can be downloaded from this GitHub repo via pip. Just run this command in your Terminal:
```
$ pip install git+https://github.com/mariusrklein/sc_imputation_denoising
```
This installs the package and all dependencies.

## Contents

The package contains the following modules:
 
* imputation: contains functions for the preparation of the data for imputation or denoising, e.g. [filtering of sparse ions and cells](https://github.com/mariusrklein/sc_imputation_denoising/blob/main/sc_imputation_denoising/imputation/filtering.py), [simulation of missing values](https://github.com/mariusrklein/sc_imputation_denoising/blob/main/sc_imputation_denoising/imputation/simulation.py) using MCAR or MNAR mechanisms, and the calculation of the missing value rate.
* evaluation: contains functions for the high-level evaluation of imputation and denoising methods and the visualization of the respective results, e.g. collective calculation of the [evaluation metrics](https://github.com/mariusrklein/sc_imputation_denoising/blob/main/sc_imputation_denoising/evaluation/evaluation_metrics.py), the [complete workflow](https://github.com/mariusrklein/sc_imputation_denoising/blob/main/sc_imputation_denoising/evaluation/evaluation_workflow.py) of sparsity simulation, imputation, and evaluation, [visualization of the results](https://github.com/mariusrklein/sc_imputation_denoising/blob/main/sc_imputation_denoising/evaluation/evaluation_plots.py)
* metrics: implementation of the individual metrics employed in the work, e.g. mean squared error of values (MSE), cluster separation metrics.


## Usage

Examples of the usage of the functions are provided in [this tutorial](https://github.com/mariusrklein/sc_imputation_denoising/blob/main/tutorial.ipynb).

