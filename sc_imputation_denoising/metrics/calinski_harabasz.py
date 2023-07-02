import anndata as ad

def calinski_harabasz(adata: ad.AnnData, condition_key: str) -> float:
    """ Calinski-Harabasz score of the dataset

    Calculates the Calinski-Harabasz score of the dataset. The score judges clustering quality using 
    the ratio of between-cluster dispersion and within-cluster dispersion. Values range between 0
    and arbitrary high values, the latter indicating better cluster spearation. Cluster labels are
    taken from the condition_key column of adata.obs.

    :param adata: AnnData object with intensity matrix stored in adata.X
    :param condition_key: column name of the condition labels in adata.obs
    
    """
    from sklearn.metrics import calinski_harabasz_score

    return calinski_harabasz_score(adata.X, adata.obs[condition_key])