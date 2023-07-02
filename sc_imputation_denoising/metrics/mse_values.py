import anndata as ad

def mse_values(adata: ad.AnnData, adata_ctrl: ad.AnnData, inverted = True) -> float:
    """ Mean squared error between imputed and control data

    Calculates the mean squared error between the imputed and control data for all intensities.
    For that, the iontensity matrices are flattened and compared using mean squared error.

    :param adata: AnnData object after imputation with intensity matrix stored in adata.X
    :param adata_ctrl: AnnData object without imputation with intensity matrix stored in adata.X
    :param inverted: if True, return -1*mse such that higher values indicate better imputation
    
    """
    from sklearn.metrics import mean_squared_error

    mse = mean_squared_error(adata_ctrl.X.flatten(), adata.X.flatten())

    if inverted:
        return -1*mse
    else:
        return mse