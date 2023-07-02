import anndata as ad

def mse_variance(adata: ad.AnnData, adata_ctrl: ad.AnnData, inverted = True) -> float:
    """ MSE of the ion variances between imputed and control data

    takes the variance of each ion and compares them between imputed and control data using mean
    squared error. 

    :param adata: AnnData object after imputation with intensity matrix stored in adata.X
    :param adata_ctrl: AnnData object without imputation with intensity matrix stored in adata.X
    :param inverted: if True, return -1*mse such that higher values indicate better imputation
    
    """
    from sklearn.metrics import mean_squared_error

    mse_v = mean_squared_error(adata_ctrl.to_df().var(), adata.to_df().var())

    if inverted:
        return -1*mse_v
    else:
        return mse_v