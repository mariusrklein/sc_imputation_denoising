import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import re
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm
import pickle
import copy

from sc_imputation_denoising.imputation.constants import const
import sc_imputation_denoising.imputation.imputation as imp
import sc_imputation_denoising.evaluation.utils as imp_eval
from sc_imputation_denoising.evaluation.evaluation_metrics import metrics_plotting

print("IE version 1.3")


class evaluation_workflow:
    dataset_dict = {}
    dataset_dict_full = {}
    adata_list = []
    simulated_rates = []
    baseline_key = None
    _analysis_ions = None

    def __init__(self, adata, sample_col=None, batch_col=None, condition_col=None):
        """Evaluation workflow for imputation methods

        :param adata: Fully prepared (normalized and log-transformed) AnnData object
        :param sample_col: Column name in adata.obs that contains sample/well information
        :param batch_col: Column name in adata.obs that contains batch information
        :param condition_col: Column name in adata.obs that contains condition information

        """
        if sample_col:
            adata.obs[const.SAMPLE_COL] = adata.obs[sample_col]

        if batch_col:
            adata.obs[const.BATCH_COL] = adata.obs[batch_col]

        if condition_col:
            adata.obs[const.CONDITION_COL] = adata.obs[condition_col]

        self.dataset = adata
        self.baseline_dr = imp.get_dropout_rate(self.dataset)

    @property
    def analysis_ions(self) -> list:
        if self._analysis_ions is None:
            self._analysis_ions = self._get_differential_ions()

        return self._analysis_ions

    @analysis_ions.setter
    def analysis_ions(self, value):
        self._analysis_ions = value

    @analysis_ions.deleter
    def analysis_ions(self):
        self._analysis_ions = None

    def _get_differential_ions(self, groupby=const.CONDITION_COL, n_ions=2):
        sc.tl.rank_genes_groups(
            self.dataset, groupby=groupby, method="t-test_overestim_var", n_genes=n_ions
        )

        diff_ions = list(
            sc.get.rank_genes_groups_df(self.dataset, group=None).sort_values(
                "logfoldchanges", ascending=False
            )["names"]
        )

        return diff_ions

    def get_variable_ions(self):
        sc.pp.highly_variable_genes(self.dataset)

        var_ions = self.dataset.var.loc[
            self.dataset.var["highly_variable"] is True,
            ["highly_variable", "means", "dispersions_norm"],
        ].index
        return var_ions

    def simulate_dropouts(
        self,
        dropout_rates: list = None,
        n_logspace: int = 0,
        method="mcar",
        **method_kws,
    ):
        """Simulate dropouts for the given dataset

        :param dropout_rates: List of dropout rates to simulate. If None, the baseline dropout rate
            is used and n_logspace is used to generate a list of dropout rates with increasing
            dropout rates.
        :param n_logspace: Number of dropout rates to simulate with increasing dropout rates.
            If 0, only the baseline dropout rate is simulated.
        :param method: Method to use for simulating dropouts. See
            sc_imputation_denoising.imputation.imputation.simulate_dropouts_adata
        :param method_kws: Keyword arguments for the dropout simulation method

        """
        if dropout_rates is None:
            dropout_rates = [self.baseline_dr]

            if n_logspace > 0:
                dropout_rates = dropout_rates + list(
                    np.round(
                        np.logspace(
                            start=np.log10(self.baseline_dr), stop=0, num=n_logspace + 2
                        ),
                        decimals=2,
                    )[
                        1:-1
                    ]  # first and last are the baseline dropout ratio and 1
                )

            print(
                f"Generated {len(dropout_rates)} datasets with increasing dropout rates:"
                + f" {', '.join([str(i) for i in dropout_rates])}"
            )

        else:
            if np.min(dropout_rates) < self.baseline_dr:
                raise ValueError(
                    f"baseline dropout ratio is {self.baseline_dr}. Given dropout ratios "
                    + "to simulate should be all higher."
                )
            elif np.min(dropout_rates) > self.baseline_dr:
                dropout_rates.append(self.baseline_dr)

        # we want the baseline dropout ratio first
        dropout_rates = sorted(dropout_rates)
        self.simulated_rates = dropout_rates
        print(f"simulating dropout rates {dropout_rates} with method {method}")
        self.dataset_dict = {}
        for i, dr in enumerate(dropout_rates):
            simulated_adata = imp.simulate_dropouts_adata(
                adata=self.dataset, rate=dr, method=method, copy=True, **method_kws
            )

            # copying raw data to a layer as it is easier to access by iterating
            simulated_adata.layers["ctrl"] = simulated_adata.X

            if dr == self.baseline_dr:
                dr = f"{np.round(self.baseline_dr, 2)}_baseline"
                self.baseline_key = dr
            self.dataset_dict[dr] = simulated_adata

    def init_analysis(self, precalc_umap=True, n_jobs=None, test_subset=None):
        """Initialize analysis workflow for imputation methods

        :param precalc_umap: If True, PCs and UMAPs are precalculated for all datasets
        :param n_jobs: Number of jobs to use for precalculation
        :param test_subset: For testing purposes, only the first n datasets are precalculated on
            one core

        """
        self.adata_list = []

        layers = self.dataset_dict[self.baseline_key].layers

        for dr, adata in self.dataset_dict.items():
            for layer in layers:
                adata_analysis_copy = adata.copy()
                adata_analysis_copy.X = adata_analysis_copy.layers[layer]

                del (
                    adata_analysis_copy.obsp,
                    adata_analysis_copy.obsm,
                    adata_analysis_copy.layers,
                )

                self.adata_list.append(
                    {
                        "dropout_rate": str(dr),
                        "imputation": layer,
                        "adata": adata_analysis_copy,
                    }
                )

        def update_adata_list_precalc(element: dict):
            element["adata"] = imp_eval.calculate_umap(element["adata"])
            return element

        if precalc_umap:
            print("Precalculating PCs and UMAPs")

            if n_jobs is None:
                n_jobs = min(multiprocessing.cpu_count(), len(self.adata_list))

            if test_subset is not None:
                out = [
                    update_adata_list_precalc(l)
                    for l in tqdm(self.adata_list[:test_subset])
                ]
                return out

            self.adata_list = Parallel(n_jobs=n_jobs, verbose=50)(
                delayed(function=update_adata_list_precalc)(l)
                for l in tqdm(self.adata_list)
            )

    def analyse_imputation(
        self, function=None, n_jobs=None, test_subset=None, verbose=0
    ) -> pd.DataFrame:
        """Analysis workflow for imputation methods

        :param function: Function to use for analysis. If None, the default function
            _analyse_imputation is used.
        :param n_jobs: Number of jobs to use for analysis
        :param test_subset: For testing purposes, only the first n datasets are analysed on
            one core
        :param verbose: Verbosity level for Parallel

        return: DataFrame with analysis results
        """
        if len(self.adata_list) == 0:
            self.init_analysis()

        if self._analysis_ions is None:
            self._analysis_ions = self._get_differential_ions()

        if function is None:
            function = self._analyse_imputation

        if test_subset is not None:
            cond_list = self.adata_list[:test_subset]
            return pd.concat(
                [function(params=i) for i in tqdm(cond_list[:test_subset])]
            )

        if n_jobs is None:
            n_jobs = min(multiprocessing.cpu_count(), len(self.adata_list))

        if n_jobs == 1:
            analysis_dfs = [function(params=i) for i in tqdm(self.adata_list)]
        else:
            analysis_dfs = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(function=function)(l) for l in tqdm(self.adata_list)
            )

        analysis_df = pd.concat(analysis_dfs)

        if "dr" in analysis_df.columns:
            analysis_df["dropout_rate"] = (
                analysis_df["dr"]
                .apply(lambda x: re.match("[\d\.]*", x).group(0))
                .astype(float)
            )

        return analysis_df

    def _analyse_imputation(self, params) -> pd.DataFrame:
        """Prototype analysis function for imputation methods

        :param params: Dictionary with parameters for analysis, with keys 'adata', 'imputation',
            'dropout_rate' as present in self.adata_list elements

        returns: DataFrame with analysis results of one dataset

        """
        adata = params["adata"]

        metrics = metrics_plotting(
            adata=adata, adata_ctrl=self.dataset, condition_key=const.CONDITION_COL
        )
        metrics["imputation"] = params["imputation"]
        metrics["dropout_rate"] = params["dropout_rate"]

        out_df = pd.DataFrame(
            metrics, index=[f"{params['imputation']}_{params['dropout_rate']}"]
        )
        return out_df

    def save_to_pickle(self, save_to):
        """Save evaluation_workflow object to pickle file

        :param save_to: path to save the object to

        """
        if not save_to.endswith(".pkl"):
            warnings.warn("File extension is not .pkl. Adding it automatically.")
            save_to = save_to + ".pkl"

        with open(save_to, "wb") as outp:
            pickle.dump(self, outp, pickle.HIGHEST_PROTOCOL)

        return save_to

    def copy(self):
        return copy.copy(self)


def get_from_pickle(path):
    """Load evaluation_workflow object from pickle file"""
    with open(path, "rb") as inp:
        load_obj = pickle.load(inp)

    return load_obj
