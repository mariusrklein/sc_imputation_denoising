""" Functions for assessing sparsity and simulating dropouts in anndata objects.

Functions:
    get_dropout_rate: calculate the dropout rate of an AnnData object or a certain layer of it.
    simulate_dropouts_adata: simulate dropouts in an AnnData object.
    calculate_dropout_rate_array: calculate the dropout rate of an array.
    simulate_dropouts_mcar_array: simulate dropouts in an array with missing completely at random
    (MCAR) mechanism.
    simulate_dropouts_mnar_array: simulate dropouts in an array with missing not at random (MNAR)
    mechanism.
    plot_dropout_simulation_decision: plot decision scores for dropout simulation.

Author: Marius Klein, July 2023

"""

from typing import Union
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
import math


def get_dropout_rate(
    adata: ad.AnnData,
    layer: str = None,
    verbose: bool = True,
    groupby: Union[str, list] = None,
) -> Union[float, pd.DataFrame]:
    """Calculate the dropout rate of an AnnData object or a certain layer of it.

    :param adata: anndata object
    :param layer: layer to calculate the dropout rate of. If None, the X layer is used.
    :param verbose: if True, prints additional information
    :param groupby: if not None, calculates the dropout rate for each group in the groupby column
    and returns a dataframe with the dropout rate for each group

    :return: dropout rate as float or dataframe of floats for individual groups
    """
    if layer is None:
        rate = calculate_dropout_rate_array(adata.X, verbose=verbose)
    else:
        rate = calculate_dropout_rate_array(adata.layers[layer], verbose=verbose)

    if groupby is not None:
        if not isinstance(groupby, list):
            groupby = [groupby]
        rate = (
            sc.get.obs_df(adata, layer=layer, keys=list(adata.var_names) + groupby)
            .reset_index()
            .melt(id_vars=["cell_id"] + groupby, value_name="dropout_ratio")[
                ["dropout_ratio"] + groupby
            ]
            .groupby(groupby)
            .agg(lambda x: x.eq(0).sum() / x.count())
            .dropna()
        )
    return rate


def calculate_dropout_rate_array(array: np.array, verbose=True) -> float:
    """Calculate the dropout rate of an array.

    :param array: array to calculate the dropout rate of
    :param verbose: if True, also prints the dropout rate

    :return: dropout rate as float
    """
    if array.size > 0:
        rate = 1 - (np.count_nonzero(array) / array.size)
    else:
        rate = 1
    if verbose:
        print(f"Global dropout rate: {rate: .2%}")
    return rate


def simulate_dropouts_adata(
    adata,
    rate,
    method="mcar",
    copy=True,
    use_layer=None,
    create_layer=None,
    simulate_raw=True,
    verbose=False,
    **method_kws,
) -> ad.AnnData:
    """Simulate dropouts in an AnnData object.

    :param adata: anndata object
    :param rate: dropout rate that should be simulated. Must be higher than the current dropout rate
    of the data.
    :param method: method to use for dropout simulation. Either 'mcar' or 'mnar' for missing
    completely at random or missing not at random.
    :param copy: if True, returns a copy of the adata object with the simulated dropouts. If False,
    the adata object is modified in place.
    :param use_layer: layer to use for dropout simulation. If None, the X layer is used.
    :param create_layer: layer to save the simulated dropouts in. If None, the X layer is used.
    :param simulate_raw: if True, dropouts are simulated in the raw intensities (adata.raw) as well.
    :param verbose: if True, prints additional information
    :param method_kws: additional keyword arguments for the dropout simulation method

    returns: anndata object with simulated dropouts
    """

    if use_layer is None:
        array = adata.X
    else:
        array = adata.layers[use_layer]

    if method == "mcar":
        dropout_array, mask_array = simulate_dropouts_mcar_array(
            array=array, rate=rate, return_mask=True, verbose=verbose, **method_kws
        )
    elif method == "mnar":
        dropout_array, mask_array = simulate_dropouts_mnar_array(
            array=array, rate=rate, return_mask=True, verbose=verbose, **method_kws
        )
    else:
        raise NotImplementedError(
            f"The dropout simulation method {method} is not implemented. "
            + "Please use 'mcar' or 'mnar'."
        )

    if copy:
        adata_out = adata.copy()
    else:
        adata_out = adata

    if create_layer is None:
        adata_out.X = dropout_array
    else:
        adata_out.layers[create_layer] = dropout_array

    adata_out.var["dropout_ratio"] = rate
    adata_out.obsm["dropout_mask"] = mask_array

    if hasattr(adata.raw, "X") and simulate_raw:
        print(
            "supplied adata has raw attribute. Simulating dropouts here aswell."
            + " You can preven this by setting argument simulate_raw=False"
        )
        raw_ad = adata_out.copy()
        raw_ad.X = adata_out.raw.X * mask_array
        adata_out.raw = raw_ad

    return adata_out


def simulate_dropouts_mcar_array(
    array, rate, verbose=True, return_mask=False
) -> np.array:
    """Simulate dropouts in an array with missing completely at random (MCAR) mechanism.

    :param array: array to simulate dropouts in
    :param rate: dropout rate that should be simulated. Must be higher than the current dropout rate
    of the data.
    :param v: if True, prints additional information (plot of decision scores)
    :param return_mask: if True, returns the dropout mask as well as the array with simulated dropouts

    returns: array with simulated dropouts

    """
    if rate < calculate_dropout_rate_array(array, verbose=False):
        raise ValueError(
            "Desired dropout rate is below current dropout rate. Use higher rate."
        )
    elif rate == calculate_dropout_rate_array(array, verbose=False):
        if return_mask:
            return (array, np.ones_like(array).astype(bool))
        return array
    # draw random number for every data point
    rand_array = np.random.uniform(size=array.shape)

    # zero values from array should not be dropped out manually
    zero_fillin = 0.0
    rand_array = np.where(array != 0, rand_array, zero_fillin)

    # finding percentile cutoff for removing values, to get to exact dropout rate
    cutoff = np.percentile(rand_array, 100 * rate)

    if verbose:
        plot_dropout_simulation_decision(array, rand_array, zero_fillin, cutoff)

    # simulating dropouts
    array_dropped = np.where(rand_array > cutoff, array, 0)
    if return_mask:
        return (array_dropped, rand_array > cutoff)
    return array_dropped


def simulate_dropouts_mnar_array(
    array, rate, value_importance=5, verbose=True, return_mask=False
) -> np.array:
    """Simulate dropouts in an array with missing not at random (MNAR) mechanism.

    :param array: array to simulate dropouts in
    :param rate: dropout rate that should be simulated. Must be higher than the current dropout rate
    of the data.
    :param value_importance: importance of deterministic part in scoring, in particular, the value
    of a data point for the dropout decision. Higher values mean that the value of a data point is
    more important for the dropout decision.
    :param verbose: if True, prints additional information (plot of decision scores)
    :param return_mask: if True, returns the dropout mask as well as the array with simulated dropouts

    returns: array with simulated dropouts
    """
    sigmoid_v = np.vectorize(lambda x: 1 / (1 + math.exp(-x)))
    if rate < calculate_dropout_rate_array(array, verbose=False):
        raise ValueError(
            "Desired dropout rate is below current dropout rate. Use higher rate."
        )
    # draw random number for every data point
    rand_array = np.random.uniform(size=array.shape)
    # zero values from array should not be dropped out manually
    rand_array = np.add(
        rand_array, value_importance * sigmoid_v(scipy.stats.zscore(array, axis=None))
    )
    rand_array = rand_array / np.max(rand_array)
    zero_fillin = 0.0
    rand_array = np.where(array != 0, rand_array, zero_fillin)

    # finding percentile cutoff for removing values, to get to exact dropout rate
    cutoff = np.percentile(rand_array, 100 * rate)

    if verbose:
        plot_dropout_simulation_decision(array, rand_array, zero_fillin, cutoff)

    # simulating dropouts
    array_dropped = np.where(rand_array > cutoff, array, 0)
    if return_mask:
        return (array_dropped, rand_array > cutoff)
    return array_dropped


def plot_dropout_simulation_decision(array, rand_array, zero_fillin, cutoff):
    """Plot decision scores for dropout simulation.

    :param array: array to simulate dropouts in
    :param rand_array: array of random numbers used for dropout decision
    :param zero_fillin: value used to fill in zero values in array
    :param cutoff: cutoff value for dropout decision


    """
    array = array.flatten()
    rand_array = rand_array.flatten()

    if np.max(array) > 20:
        print("plotting in log scale")
        array = np.log1p(array)

    idx = range(0, len(array))
    if array.size > 1e4:
        print(f"you have {array.size} data points. sampling down scatterplot to 10k")
        idx = np.random.randint(0, array.size, 10000)
        # array = array[idx]
        # rand_array = rand_array[idx]

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    hue_list = np.array(
        [
            "zero" if i == zero_fillin else "dropped" if i < cutoff else "kept"
            for i in rand_array
        ]
    )

    sns.scatterplot(
        x=array[idx],
        y=rand_array[idx],
        hue=hue_list[idx],
        # palette='cividis',
        hue_order=["dropped", "kept", "zero"],
        ax=ax[0],
    ).set(
        title="Decision making on dropouts",
        xlabel="log ion intensity",
        ylabel="combined removal score",
    )

    sns.histplot(
        x=array,
        hue=hue_list,
        multiple="stack",
        stat="proportion",
        bins=20,
        # palette='cividis',
        hue_order=["dropped", "kept", "zero"],
        ax=ax[1],
    ).set(
        title="Effect on intensity distribution",
        xlabel="log ion intensity",
        ylabel="proportion of data points",
    )

    ax[0].axhline(y=cutoff)
    ax[0].get_legend().remove()

    fig.tight_layout()
