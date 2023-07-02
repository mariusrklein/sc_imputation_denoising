import anndata as ad

def davies_bouldin(adata: ad.AnnData, condition_key: str, inverted=True) -> float:
    """ Davies-Bouldin score of the dataset

    Calculates the Davies-Bouldin score of the dataset. The score judges clustering quality using 
    the ratio of cluster compactness and distance between clusters. Lower values indicate better
    separation of clusters. Cluster labels are taken from the condition_key column of adata.obs.

    :param adata: AnnData object with intensity matrix stored in adata.X
    :param condition_key: column name of the condition labels in adata.obs
    :param inverted: if True, return -1*dbs such that higher values indicate better imputation
    
    """
    from sklearn.metrics import davies_bouldin_score

    
    dbs = davies_bouldin_score(adata.X, adata.obs[condition_key])

    if inverted:
        return -1*dbs
    else:
        return dbs