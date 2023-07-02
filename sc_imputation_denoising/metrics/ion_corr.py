import numpy as np
import anndata as ad


def ion_corr(
    adata: ad.AnnData
) -> float:
    """Pairwise correlations between ions (columns) in the imputed data

    :param adata: AnnData object after imputation with intensity matrix stored in adata.X
    """
    from sc_imputation_denoising.metrics.utils import get_ion_corr_matrix

    ion_corr_mat = get_ion_corr_matrix(adata)
    icorr = np.nanmean(ion_corr_mat, axis=None)
    return icorr


def ion_corr_deviation(adata: ad.AnnData, adata_ctrl: ad.AnnData, inverted = True) -> float:
    """ absolute deviation of pairwise correlations between ions (columns) in the imputed data vs. 
    control data

    :param adata: AnnData object after imputation with intensity matrix stored in adata.X
    :param adata_ctrl: AnnData object without imputation with intensity matrix stored in adata.X
    :param inverted: if True, return -1*ion_corr_deviation

    """
    from sc_imputation_denoising.metrics.utils import get_ion_corr_matrix

    ion_corr_mat = get_ion_corr_matrix(adata)
    ion_corr_mat_ctrl = get_ion_corr_matrix(adata_ctrl)
    ion_corr_deviation = np.abs(np.nanmean(ion_corr_mat - ion_corr_mat_ctrl), axis=None)
    if inverted:
        return -1*ion_corr_deviation
    else:
        return ion_corr_deviation
