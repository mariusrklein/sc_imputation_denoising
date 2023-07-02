import numpy as np
import anndata as ad


def ion_corr(
    adata: ad.AnnData
) -> float:
    """Pairwise correlations between cells (rows) in the imputed data

    :param adata: AnnData object after imputation with intensity matrix stored in adata.X
    :param condition_key: column name of the condition labels in adata.obs
    :param subset_above: threshold for sampling cells from the dataframe to save resources

    """
    from sc_imputation_denoising.metrics.utils import get_ion_corr_matrix

    ion_corr_mat = get_ion_corr_matrix(adata)
    icorr = np.nanmean(ion_corr_mat, axis=None)
    return icorr


def ion_corr_deviation(adata: ad.AnnData, adata_ctrl: ad.AnnData, inverted = True) -> float:
    """Pairwise correlations between cells (rows) in the imputed data

    :param adata: AnnData object after imputation with intensity matrix stored in adata.X
    :param adata_ctrl: AnnData object without imputation with intensity matrix stored in adata.X

    """
    from sc_imputation_denoising.metrics.utils import get_ion_corr_matrix

    ion_corr_mat = get_ion_corr_matrix(adata)
    ion_corr_mat_ctrl = get_ion_corr_matrix(adata_ctrl)
    ion_corr_deviation = np.abs(np.nanmean(ion_corr_mat - ion_corr_mat_ctrl), axis=None)
    if inverted:
        return -1*ion_corr_deviation
    else:
        return ion_corr_deviation
