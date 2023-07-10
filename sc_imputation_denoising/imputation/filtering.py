""" Functions for filtering of ions and cells.

Functions:
    plot_filtering_qc: plot QC metrics for systematic filtering of ions and cells

Author: Marius Klein, July 2023

"""
import os
from typing import Union
import pandas as pd
import anndata as ad
import scanpy as sc
import seaborn as sns

from sc_imputation_denoising.imputation.simulation import get_dropout_rate


def plot_filtering_qc(
    adata: ad.AnnData,
    n_ions: Union[int, float] = None,
    cell_hue_variable: str = None,
    n_cells: Union[int, float] = None,
    save_path: str = None,
    copy=True,
    ion_plot_kws: dict = None,
    **cell_plot_kws,
):
    """Plot QC metrics for filtering ions and cells

    :param adata: AnnData object
    :param n_ions: int or float, number of unique ions that cells need to be kept in the dataset.
        If float, it is interpreted as the fraction of unique ions of the dataset that cells need
        to have in order to be kept.
    :param cell_hue_variable: str, name of the column in adata.obs that contains the cell labels to use for coloring
    :param n_cells: int or float, number of unique cells that ions need to present in to remain in
        the dataset. If float, it is interpreted as the fraction of unique cells, that ions need to
        be present in, in order to be kept.
    :param save_path: str, path to save the plots to. The file names are specifified inside the
        function. If None, the plots are not saved
    :param copy: bool, if True, a copy of adata is returned. If False, adata is modified in-place
        and returned.
    :param ion_plot_kws: dict, keyword arguments to pass to seaborn.displot for plotting the ion distribution
    :param **cell_plot_kws: additional keyword arguments to pass to seaborn.displot for plotting the cell distribution
    """

    adata_orig = adata.copy()
    default_plot_kws = dict(
        bins=50,
        linewidth=0,
        multiple="stack",
    )
    default_ion_plot_kws = default_plot_kws.copy()

    if ion_plot_kws is not None:
        default_ion_plot_kws.update(ion_plot_kws)

    ion_plot_kws = default_ion_plot_kws

    if cell_hue_variable is not None:
        default_plot_kws["hue"] = cell_hue_variable

    if cell_plot_kws is not None:
        default_plot_kws.update(cell_plot_kws)

    cell_plot_kws = default_plot_kws

    # processing n_ions and n_cells according to their type (int/float)
    if n_ions is not None:
        if type(n_ions) is int:
            min_ions = n_ions
            perc_ions = n_ions / len(adata.var_names)
        else:
            min_ions = int(n_ions * len(adata.var_names))
            perc_ions = n_ions
    if n_cells is not None:
        if type(n_cells) is int:
            min_cells = n_cells
            perc_cells = n_cells / len(adata.obs_names)
        else:
            min_cells = int(n_cells * len(adata.obs_names))
            perc_cells = n_cells

    # generating QC metrics dataframes and connecting them to other ion/cell information
    cell_qc_df, ion_qc_df = sc.pp.calculate_qc_metrics(adata=adata, percent_top=None)
    cell_qc = pd.merge(adata.obs, cell_qc_df, left_index=True, right_index=True)
    ion_qc = pd.merge(adata.var, ion_qc_df, left_index=True, right_index=True)

    # creating plots
    ipl = sns.displot(data=cell_qc, x="n_genes_by_counts", **cell_plot_kws)

    ipl.set_xlabels("Unique ions")
    ipl.set_ylabels("N cells")
    for ax in ipl.axes.flat:
        ax.set_xlim(left=0)
        ax.set_title("Distribution of cells, stacked by condition")
        if n_ions is not None:
            ax.axvline(x=min_ions, linestyle=":", c="#CC3333")
            ax.text(
                x=min_ions,
                y=ax.get_ylim()[1],
                s=f" {min_ions} / {perc_ions:.0%}",
                va="top",
                ha="left",
                color="#CC3333",
            )
    ipl.tight_layout()

    cpl = sns.displot(data=ion_qc, x="n_cells_by_counts", **ion_plot_kws)
    cpl.set_xlabels("Unique cells")
    cpl.set_ylabels("N ions")
    for ax in cpl.axes.flat:
        ax.set_xlim(left=0)
        ax.set_title("Distribution of ion intensities")
        if n_cells is not None:
            ax.axvline(x=min_cells, linestyle=":", c="#CC3333")
            ax.text(
                x=min_cells,
                y=ax.get_ylim()[1],
                s=f" {min_cells} / {perc_cells:.0%}",
                va="top",
                ha="left",
                color="#CC3333",
            )
    cpl.tight_layout()

    if save_path is not None:
        ipl.fig.savefig(os.path.join(save_path, "ion_filtering.pdf"))
        cpl.fig.savefig(os.path.join(save_path, "cell_filtering.pdf"))

    # filtering adata according to specifified thresholds.
    if copy:
        adata_out = adata.copy()
    else:
        adata_out = adata

    if n_ions is not None:
        sc.pp.filter_genes(adata_out, min_cells=min_cells)
        print(
            f"Removed {len(adata_orig.var_names) - len(adata_out.var_names)} out of {len(adata_orig.var_names)} ions"
        )
    if n_cells is not None:
        sc.pp.filter_cells(adata_out, min_genes=min_ions)
        print(
            f"Filtered {len(adata_orig.obs_names) - len(adata_out.obs_names)} out of {len(adata_orig.obs_names)} cells"
        )
        if cell_hue_variable is not None:
            n_cells_orig = adata_orig.obs.groupby("condition")["condition"].count()
            n_cells_out = adata_out.obs.groupby("condition")["condition"].count()

            # if there are conditions that have no cells left, add them to the series with 0 cells
            if len(n_cells_out) == 0:
                n_cells_out = pd.Series(0, index=n_cells_orig.index)
            rred_df = pd.concat(
                [
                    pd.Series(n_cells_orig - n_cells_out, name="difference_cells"),
                    pd.Series(
                        (n_cells_orig - n_cells_out) / n_cells_orig,
                        name="relative_reduction",
                    ).map(lambda x: f"{x:.2%}"),
                ],
                axis=1,
            )
            print(
                f"Different effect of filtering on different subsets ({cell_hue_variable}):\n{rred_df}"
            )

    if n_cells is not None or n_ions is not None:
        print(
            f"Reduced dropout rate from {get_dropout_rate(adata_orig, verbose=False):.2%} to" +
            f" {get_dropout_rate(adata_out, verbose=False):.2%}."
        )
        # get rid of cells that have no ions left after last filtering step
        sc.pp.filter_cells(adata_out, min_genes=1)

    return (adata_out, cpl, ipl)
