import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import warnings


def get_ion_corr_matrix(adata) -> pd.DataFrame:
    """
    Compute the correlation matrix of the imputed data with respect to columns
    """
    return adata.to_df().corr()
    

def get_cell_corr_matrix(adata, condition_key = 'condition', subset_above=10000) -> pd.DataFrame:
    """
    Compute the correlation matrix of the imputed data with respect to rows
    """

    # correlation matrix with more than 10k cells takes too long, subsetting stratified
    # by condition
    if len(adata.obs_names) > subset_above:
        print(f"subsetting adata with N_obs={len(adata.obs_names)} to 10k")
        subset = (
            adata.obs.groupby(condition_key, group_keys=False)
            .apply(
                lambda x: x.sample(
                    int(np.rint(subset_above * len(x) / len(adata.obs)))
                )
            )
            .sample(frac=1)
            .reset_index(drop=True)
        )
    else:
        subset = adata.obs

    return adata[subset.index].to_df().T.corr()
    
    

def calculate_umap(adata: ad.AnnData, umap_kws=None, neighbors_kws=None, **pca_kws) -> ad.AnnData:
    
    _umap_kws = dict(min_dist=0.5, spread=1.0, random_state=1, n_components=2) 
    _neighbors_kws = dict(n_neighbors=50, metric='cosine')

    if umap_kws is not None:
        _umap_kws.update(umap_kws)
    if neighbors_kws is not None:
        _neighbors_kws.update(neighbors_kws)

    sc.pp.pca(adata, **pca_kws)
    sc.pp.neighbors(adata, **_neighbors_kws)
    sc.tl.umap(adata, **_umap_kws)
    return adata




