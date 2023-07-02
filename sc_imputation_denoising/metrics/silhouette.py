import os
import anndata as ad
import scanpy as sc

from sc_imputation_denoising.metrics.utils import calculate_umap


def silhouette(
    adata: ad.AnnData,
    condition_key: str,
    umap_kws: dict = None,
    neighbors_kws: dict = None,
    **pca_kws
) -> float:
    """Silhouette score of the UMAP embedding of the dataset

    Cell clustering in UMAP space is 

    :param adata: AnnData object with intensity matrix stored in adata.X
    :param condition_key: column name of the condition labels in adata.obs
    :param umap_kws: additional arguments to sc.pp.neighbors
    :param neighbors_kws: additional arguments to sc.tl.umap
    :param **pca_kws: additional arguments to sc.pp.pca

    """
    from sklearn.metrics import silhouette_score

    if "X_umap" not in adata.obsm.keys():
        adata = calculate_umap(adata, umap_kws, neighbors_kws, **pca_kws)

    u_df = sc.get.obs_df(
        adata, 
        keys=[condition_key], obsm_keys=[("X_umap", 0), ("X_umap", 1)]
    )   

    silhouette = silhouette_score(u_df[['X_umap-0', 'X_umap-1']], u_df[condition_key])
    return silhouette



if __name__ == "__main__":
    cwd = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(os.path.dirname(cwd), "data", "Mx_Seahorse.h5ad")
    adata = ad.read_h5ad(data_path)
    print(silhouette(adata, "dataset_0"))
