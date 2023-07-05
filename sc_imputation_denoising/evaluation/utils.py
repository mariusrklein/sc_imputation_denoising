import pandas as pd
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr, pearsonr
import matplotlib
import seaborn


def cosine_similarity(A, B=None, corr=False):
    if B is None:
        B = A
    if corr:
        B = B - B.mean(axis=1)[:, np.newaxis]
        A = A - A.mean(axis=1)[:, np.newaxis]
    num = np.dot(A, B.T)
    p1 = np.sqrt(np.sum(A**2, axis=1))[:, np.newaxis]
    p2 = np.sqrt(np.sum(B**2, axis=1))[np.newaxis, :]
    return num / (p1 * p2)


def correlation_matrix(a, b=None, method="pearson"):
    if method == "pearson":
        return np.corrcoef(a, b)
    elif method == "cosine":
        return cosine_similarity(a, b, corr=False)


# compute cosine similarity
def cosine_vector(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))


def calculate_umap(adata, umap_kws=None, neighbors_kws=None, **pca_kws):
    """ Utility function to calculate UMAP embedding of an AnnData object

    :param adata: AnnData object
    :param umap_kws: dict, keyword arguments for sc.tl.umap
    :param neighbors_kws: dict, keyword arguments for sc.pp.neighbors
    :param pca_kws: dict, keyword arguments for sc.pp.pca

    returns: AnnData object with attributes for UMAP embedding.
    
    """
    _umap_kws = dict(min_dist=0.5, spread=1.0, random_state=1, n_components=2)
    _neighbors_kws = dict(n_neighbors=50, metric="cosine")

    if umap_kws is not None:
        _umap_kws.update(umap_kws)
    if neighbors_kws is not None:
        _neighbors_kws.update(neighbors_kws)

    sc.pp.pca(adata, **pca_kws)
    sc.pp.neighbors(adata, **_neighbors_kws)
    sc.tl.umap(adata, **_umap_kws)
    return adata


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
    elif isinstance(plot, seaborn.axisgrid.FacetGrid):
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

    if isinstance(plot, seaborn.axisgrid.FacetGrid):
        plot.tight_layout()
        axes_subplot.get_legend().remove()
        if not keep_legend:
            plot.legend.remove()
    else:
        if not keep_legend:
            axes_subplot.get_legend().remove()


def print_labels_on_scatter_canvas(plot, keep_legend=False):
    """Utility function to print labels on a scatter plot next to the points.

    Modifies the plot inplace and returns None.

    :param plot: matplotlib.axes.Axes or seaborn.axisgrid.FacetGrid
    :param keep_legend: bool, whether to keep the legend of the plot

    :return: None

    """
    from adjustText import adjust_text

    # we accept both axis- and figure-level plots
    if isinstance(plot, matplotlib.axes.Axes):
        plots = [plot]
    elif isinstance(plot, seaborn.axisgrid.FacetGrid):
        plots = list(plot.axes.flat)

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

    if isinstance(plot, seaborn.axisgrid.FacetGrid):
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
