""" High-level functions to compute evaluation metrics collectively.

Functions:
    metrics: Compute manually specified metrics based on an imputed and non-imputed anndata object
    metrics_plotting: Compute metrics relevant for plotting 2x2 performance matrices based on 
        imputed and non-imputed anndata object
    metrics_all: Compute all metrics based on imputed and non-imputed anndata object


Author: Marius Klein, July 2023
"""
import os
import anndata as ad
import numpy as np
import pandas as pd

from sc_imputation_denoising.metrics import (
    kmeans_ari,
    silhouette,
    calinski_harabasz,
    davies_bouldin,
    mse_values,
    mse_variance,
    ion_corr,
    ion_corr_deviation,
    cell_corr,
)

from sc_imputation_denoising.evaluation.utils import check_adata, check_obs_variable


def metrics(
    adata,
    adata_ctrl,
    condition_key,
    batch_key=None,
    kmeans_ari_=False,
    silhouette_=False,
    calinski_=False,
    davies_=False,
    mse_values_=False,
    mse_variance_=False,
    corr_ions_=False,
    corr_ions_deviation_=False,
    corr_cells_=False,
    invert_metrics=True,
):
    """Master metrics function

    Wrapper for all metrics used.
    Compute of all metrics based on imputed and non-imputed control anndata object

    :param adata:
        imputed anndata object
    :param adata_ctrl:
        nonöimputed control anndata object
    :param condition_key:
        name of condition column in adata.obs and adata_ctrl.obs
    :param batch_key:
        name of batch column in adata.obs and adata_ctrl.obs
    :param `kmeans_ari_`:
        whether to compute ARI of kMeans clustering
    :param `silhouette_`:
        whether to compute the average silhouette width scores with conditions as labels
    :param `calinski_`:
        whether to compute Calinski-Harabasz score
    :param `davies_`:
        whether to compute Davies-Bouldin score
    :param `mse_values_`:
        whether to compute MSE of all individual intensities between imputed and control.
    :param `mse_variance_`: 
        whether to compute MSE of variance of all ions between imputed and control.
    :param `corr_ions_`:    
        whether to compute mean pairwise ion correlations
    :param `corr_ions_deviation_`:  
        whether to compute mean ion correlation deviation (imputed vs. control)
    :param `corr_cells_`:
        whether to compute mean pariwise cell correlations
    :param invert_metrics:
        whether to invert metrics that are better at lower scores (e.g. MSE, Davies-Bouldin)
    """


    check_adata(adata)
    check_obs_variable(condition_key, adata.obs)

    check_adata(adata_ctrl)
    check_obs_variable(condition_key, adata_ctrl.obs)

    if batch_key is not None:
        check_obs_variable(batch_key, adata.obs)
        check_obs_variable(batch_key, adata_ctrl.obs)

    result_dict = {}

    if kmeans_ari_:
        print("ARI...")
        result_dict["kMeans_ari"] = kmeans_ari(adata, condition_key=condition_key)
    else:
        result_dict["kMeans_ari"] = np.nan

    if silhouette_:
        print("Silhouette...")
        result_dict["Silhouette_Score"] = silhouette(adata, condition_key=condition_key)
    else:
        result_dict["Silhouette_Score"] = np.nan

    if davies_:
        print("Davies...")
        result_dict["Davies_Bouldin_Score"] = davies_bouldin(
            adata, condition_key=condition_key, inverted=invert_metrics
        )
    else:
        result_dict["Davies_Bouldin_Score"] = np.nan

    if calinski_:
        print("Calinski...")
        result_dict["Calinski_Harabasz_Score"] = calinski_harabasz(
            adata, condition_key=condition_key
        )
    else:
        result_dict["Calinski_Harabasz_Score"] = np.nan

    if mse_values_:
        print("MSE values...")
        result_dict["MSE_values"] = mse_values(
            adata, adata_ctrl, inverted=invert_metrics
        )
    else:
        result_dict["MSE_values"] = np.nan

    if mse_variance_:
        print("MSE variance...")
        result_dict["MSE_variance"] = mse_variance(
            adata, adata_ctrl, inverted=invert_metrics
        )
    else:
        result_dict["MSE_variance"] = np.nan

    if corr_ions_:
        print("Ion correlation...")
        result_dict["Ion_Correlation"] = ion_corr(adata)
    else:
        result_dict["Ion_Correlation"] = np.nan

    if corr_cells_:
        print("Cell correlation...")
        result_dict["Cell_Correlation"] = cell_corr(adata, condition_key=condition_key)
    else:
        result_dict["Cell_Correlation"] = np.nan

    if corr_ions_deviation_:
        print("Ion correlation deviation...")
        result_dict["Ion_Correlation_Deviation"] = ion_corr_deviation(
            adata, adata_ctrl, inverted=invert_metrics
        )
    else:
        result_dict["Ion_Correlation_Deviation"] = np.nan

    return pd.DataFrame.from_dict(result_dict, orient="index")



def metrics_plotting(
    adata,
    adata_ctrl,
    condition_key,
    batch_key=None,
):
    """Metrics for plotting

    Wrapper for metrics specifically for plotting 2x2 matrix plots.
    Compute of all metrics based on imputed and non-imputed control anndata object

    :param adata:
        imputed anndata object
    :param adata_ctrl:
        nonöimputed control anndata object
    :param condition_key:
        name of condition column in adata.obs and adata_ctrl.obs
    :param batch_key:
        name of batch column in adata.obs and adata_ctrl.obs
    """

    return metrics(
        adata=adata,
        adata_ctrl=adata_ctrl,
        condition_key=condition_key,
        kmeans_ari_=True,
        silhouette_=True,
        calinski_=True,
        davies_=True,
        mse_values_=True,
        mse_variance_=True,
        corr_ions_=False,
        corr_ions_deviation_=True,
        corr_cells_=True,
        invert_metrics=True,
    )[0].to_dict()

def metrics_all(
    adata,
    adata_ctrl,
    condition_key,
    batch_key=None,
):
    """All Metrics

    Wrapper to return all available metrics.
    Compute of all metrics based on imputed and non-imputed control anndata object

    :param adata:
        imputed anndata object
    :param adata_ctrl:
        nonöimputed control anndata object
    :param condition_key:
        name of condition column in adata.obs and adata_ctrl.obs
    :param batch_key:
        name of batch column in adata.obs and adata_ctrl.obs
    """

    return metrics(
        adata=adata,
        adata_ctrl=adata_ctrl,
        condition_key=condition_key,
        kmeans_ari_=True,
        silhouette_=True,
        calinski_=True,
        davies_=True,
        mse_values_=True,
        mse_variance_=True,
        corr_ions_=True,
        corr_ions_deviation_=True,
        corr_cells_=True,
        invert_metrics=True,
    )[0].to_dict()


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(os.path.dirname(cwd), "data", "Mx_Seahorse.h5ad")
    adata = ad.read_h5ad(data_path)
    print(
        metrics(
            adata,
            adata,
            "dataset_0",
            kmeans_ari_=True,
            silhouette_=False,
            calinski_=True,
            davies_=True,
            mse_values_=True,
            mse_variance_=True,
            corr_ions_=True,
            corr_ions_deviation_=True,
            corr_cells_=True,
        )
    )
