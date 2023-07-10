"""Utility functions for evaluation of imputation and denoising methods

Functions:
    cosine_similarity: compute cosine similarity between two matrices
    correlation_matrix: compute correlation matrix between two matrices
    cosine_vector: compute cosine similarity between two vectors
    calculate_umap: calculate UMAP embedding of an AnnData object
    print_labels_on_line_canvas: print labels on a line plot, next to lines
    print_stat_info_on_grid: print statistical correlation information on a seaborn FacetGrid scatterplot
    generate_color_palette_2d: generate a color palette for grouped elements in seaborn plots
    generate_color_palette: generate a color palette for seaborn plots

Author: Marius Klein, July 2023

"""

import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr, pearsonr
import matplotlib
import seaborn as sns


def cosine_similarity(A: np.array, B: np.array) -> np.array:
    """Cosine similarity between two matrices

    :param A: np.array, first matrix, rows are variables, columns are observations
    :param B: np.array, second matrix, rows are variables, columns are observations

    :return: np.array, cosine similarity matrix
    """
    num = np.dot(A, B.T)
    p1 = np.sqrt(np.sum(A**2, axis=1))[:, np.newaxis]
    p2 = np.sqrt(np.sum(B**2, axis=1))[np.newaxis, :]
    return num / (p1 * p2)


def correlation_matrix(a: np.array, b: np.array, method="pearson") -> np.array:
    """Compute correlation matrix between two matrices

    :param a: np.array, first matrix, rows are variables, columns are observations
    :param b: np.array, second matrix, rows are variables, columns are observations
    :param method: str, method to compute correlation. Can be 'pearson' or 'cosine'

    :return: np.array, correlation matrix
    """

    if method == "pearson":
        return np.corrcoef(a, b)
    elif method == "cosine":
        return cosine_similarity(a, b, corr=False)


# compute cosine similarity
def cosine_vector(A, B):
    """Cosine similarity between two vectors

    :param A: np.array, one-dimensional, first vector
    :param B: np.array, one-dimensional, second vector
    """
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def calculate_umap(adata, umap_kws=None, neighbors_kws=None, **pca_kws):
    """ Utility function to calculate UMAP embedding of an AnnData object

    :param adata: AnnData object
    :param umap_kws: dict, keyword arguments for sc.tl.umap
    :param neighbors_kws: dict, keyword arguments for sc.pp.neighbors
    :param pca_kws: dict, keyword arguments for sc.pp.pca

    returns: AnnData object with attributes for UMAP embedding.
    
    """
    from sc_imputation_denoising.metrics.utils import calculate_umap
    return calculate_umap(adata, umap_kws, neighbors_kws, **pca_kws)


def print_labels_on_line_canvas(
    plot,
    label_point=-1,
    exact_x=True,
    ha="left",
    offset=0.1,
    keep_legend=False,
    fontsize=7,
):
    """Utility function to print labels on a line plot, next to lines

    Function modifies the plot in-place and returns None

    :param plot: matplotlib.axes.Axes or seaborn.axisgrid.FacetGrid
    :param label_point: x-position of line to place label at. -1 means at the end of the line
    :param exact_x: if True, label_point is interpreted as the exact x-position of the label. If False,
    the x-position to label_point is adjusted automatically.
    :param ha: horizontal alignment of label. Can be 'left', 'right' or 'center'
    :param offset: offset of label from label_point
    :param keep_legend: if True, the legend of the plot is kept. If False, the legend is removed.
    :param fontsize: fontsize of the label

    :return: None

    """
    from adjustText import adjust_text

    # we accept both axis- and figure-level plots
    if isinstance(plot, matplotlib.axes.Axes):
        axes_subplot = plot
    elif isinstance(plot, sns.axisgrid.FacetGrid):
        if len(plot.axes.flat) > 1:
            raise NotImplementedError(
                "Figure-level plot can only have one axis. "
                + f"The plot you supplied has {len(plot.axes.flat)}"
            )

        axes_subplot = plot.axes.flat[0]
    else:
        raise NotImplementedError(
            "print_labels_on_line_canvas is only implemented "
            + "for matplotlib.axes.Axes and seaborn.axisgrid.FacetGrid. You supplied "
            + f"{type(axes_subplot)}"
        )

    # extract line labels from figure legend.
    labels = [t.get_text() for t in list(axes_subplot.legend().get_texts())]
    texts = []
    x_positions = axes_subplot.lines[0].get_xdata()
    y_positions = []

    # create labels on plot
    for i, l in enumerate(axes_subplot.lines):
        y = l.get_ydata()
        # only create label if line has data points (lines associated to legend are ignored)
        if len(y) > 0:
            texts.append(
                axes_subplot.annotate(
                    f"{labels[i]}",
                    ha=ha,
                    xy=(x_positions[label_point], y[label_point]),
                    xycoords=("data", "data"),
                    color=l.get_color(),
                    size=fontsize,
                )
            )

        y_positions.extend(y)

    # repel-magic moves labels of lines that are close to each other
    adjust_text(
        texts,
        x=list(x_positions) * len(labels),
        y=y_positions,
        ha=ha,
        only_move={"text": "y" if exact_x else "xy"},
    )

    if exact_x:
        # prevents overlap of label with text
        if ha == "left":
            final_pos = x_positions[label_point] + offset
        else:
            final_pos = x_positions[label_point] - offset
        [t.set(x=final_pos, ha=ha) for t in texts]

    if isinstance(plot, sns.axisgrid.FacetGrid):
        plot.tight_layout()
        axes_subplot.get_legend().remove()
        if not keep_legend:
            plot.legend.remove()
    else:
        if not keep_legend:
            axes_subplot.get_legend().remove()


def print_stat_info_on_grid(
    grid,
    ionx=None,
    iony=None,
    fontsize=7,
    posx=0.05,
    posy=0.97,
    methods=["spearman", "cosine"],
    dropna=False,
):
    """ Utility function to print statistical correlation information on a seaborn FacetGrid 
    scatterplot

    Modifies the plot inplace and returns None.

    :param grid: seaborn.axisgrid.FacetGrid
    :param ionx: str, name of the x axis ion
    :param iony: str, name of the y axis ion
    :param fontsize: int, fontsize of the text
    :param posx: float, x position of the text
    :param posy: float, y position of the text
    :param methods: list of str, methods to calculate the correlation
    :param dropna: bool, whether to drop rows with NaN values

    :return: None
    """

    if ionx is None:
        ionx = grid.axes[-1][0].xaxis.get_label_text()

    if iony is None:
        iony = grid.axes[-1][0].yaxis.get_label_text()

    temp = []
    for (row, col, hue), data in reversed(list(grid.facet_data())):
        if hue != 0:
            temp.append(data)
            continue
        else:
            data = pd.concat([data] + temp)
            temp = []

        if dropna:
            data = data.dropna(subset=[ionx, iony])

        if len(data) == 0:
            continue

        if ionx not in data.columns:
            raise ValueError(
                f"Supplied or inferred x axis ion {ionx} was not found in facet grid data."
            )
        if iony not in data.columns:
            raise ValueError(
                f"Supplied or inferred y axis ion {iony} was not found in facet grid data."
            )

        ax = grid.axes[row][col]
        texts = []

        if "pearson" in methods:
            corr = pearsonr(data[ionx], data[iony])
            texts.append(f"P' r={corr.statistic: .2f},P={corr.pvalue:.2g}")

        if "spearman" in methods:
            corr = spearmanr(data[ionx], data[iony])
            texts.append(
                f"S' \N{GREEK SMALL LETTER RHO}={corr.statistic: .2f},P={corr.pvalue:.2g}"
            )

        if "cosine" in methods:
            cosine = cosine_vector(data[ionx], data[iony])
            texts.append(f"cos={cosine: .2f}")

        ax.text(
            posx,
            posy,
            ",\n".join(texts),
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(alpha=0.5, facecolor="white"),
            fontsize=fontsize,
        )

def generate_color_palette_2d(n_colors_list, palette='colorblind'):
    """ Generate a color palette for seaborn plots.

    This function generates a custom color palette for grouped elements, e.g. different variants of
    imputation methods that can be classified into groups. The different variants will get different
    shades of the same color, while the groups will get different colors.

    :param n_colors_list: list of int, number of colors for each group. The order of the list is
        the same as the order of the colors in the returned palette.

    :return: list of RGB tuples, color palette for seaborn plots

    Example:
        >>> palette = generate_color_palette_2d([3, 2])
        >>> palette
        [
            (0.0, 0.2, 0.1), # group 1, shade 1
            (0.0, 0.3, 0.1),
            (0.0, 0.4, 0.1),
            (0.2, 0.5, 0.4), # group 2, shade 1
            (0.3, 0.5, 0.4)
        ]
        >>> # use palette in seaborn plot. Ensure that the order given to generate_color_palette_2d
        >>> # is the same as the order of the groups in hue_order
        >>> sns.relplot(
                ..., 
                palette=palette, 
                hue_order=['g1s1', 'g1s2', 'g1s3', 'g2s1', 'g2s2']
            )   
    
    """
    # Generate a base color palette
    base_palette = generate_color_palette(len(n_colors_list), palette=palette)
    
    # Generate shades for each base color
    color_palette = []
    for i, color in enumerate(base_palette):
        shades = sns.light_palette(color, n_colors=n_colors_list[i] + 1)
        color_palette.extend(shades[1:])
    
    return color_palette


def generate_color_palette(n_colors, palette='colorblind'):
    """ Generate a color palette for seaborn plots.

    This function generates a custom color palette for elements in a seaborn plot.
    
    :param n_colors: int, number of colors in the palette
    :param palette: str, name of the palette. Can be any palette supported by seaborn.color_palette

    :return: list of RGB tuples, color palette for seaborn plots

    Example:
        >>> palette = generate_color_palette(3)
        >>> palette
        [(0.0, 0.2, 0.1), (0.0, 0.3, 0.1), (0.0, 0.4, 0.1)]
        >>> # use palette in seaborn plot
        >>> sns.relplot(..., palette=palette)

    """
    # Generate a base color palette
    if palette == 'colorblind':
        base_palette = sns.color_palette(
            ['#006BA4', '#FF800E', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF'], 
            n_colors=n_colors)
    else:
        base_palette = sns.color_palette(palette, n_colors=n_colors)

    return base_palette
