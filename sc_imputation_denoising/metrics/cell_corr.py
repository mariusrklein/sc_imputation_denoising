import numpy as np
import anndata as ad


def cell_corr(
    adata: ad.AnnData, condition_key="condition", subset_above=10000
) -> float:
    """Pairwise correlations between cells (rows) in the imputed data

    :param adata: AnnData object after imputation with intensity matrix stored in adata.X
    :param condition_key: column name of the condition labels in adata.obs
    :param subset_above: threshold for sampling cells from the dataframe to save resources

    """
    from sc_imputation_denoising.metrics.utils import get_cell_corr_matrix

    cell_corr_matrix = get_cell_corr_matrix(
        adata, condition_key=condition_key, subset_above=subset_above
    )
    ccorr = np.nanmean(cell_corr_matrix, axis=None)
    return ccorr
