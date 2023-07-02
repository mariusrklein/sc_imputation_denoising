import os
import anndata as ad


def kmeans_ari(adata: ad.AnnData, condition_key: str, **kwargs) -> float:
    """Adjusted Rand Index (ARI) of KMeans clustering

    Cells are cluster unsupersived using kMeans clustering, with the number of clusters set to the
    number of conditions in the dataset. The ARI is then computed between the clusters and the
    given condition labels. Returns float between -0.5 and 1, where 1 is perfect clustering.

    :param adata: AnnData object with intensity matrix stored in adata.X
    :param condition_key: column name of the condition labels in adata.obs
    :param kwargs: additional arguments to sklearn.cluster.KMeans

    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    n_clusters = len(adata.obs[condition_key].value_counts())

    kmeans_kwargs = {
        "init": "random",
        "n_init": 10,
        "random_state": 42,
    }
    if kwargs is not None:
        kmeans_kwargs.update(kwargs)

    kmeans = KMeans(n_clusters=n_clusters, **kmeans_kwargs)
    kmeans.fit(adata.X)

    ari = adjusted_rand_score(adata.obs[condition_key], kmeans.labels_)

    return ari


if __name__ == "__main__":
    cwd = os.path.dirname(os.path.realpath(__file__))
    data_path = os.path.join(os.path.dirname(cwd), "data", "Mx_Seahorse.h5ad")
    adata = ad.read_h5ad(data_path)
    print(kmeans_ari(adata, "dataset_0"))
