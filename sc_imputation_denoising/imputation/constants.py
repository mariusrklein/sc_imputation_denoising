"""Class of constants used in the imputation/denoising pipeline.

Author: Marius Klein, July 2023
"""


class const:
    IN_DATA_DIR = "/home/mklein/sc_imputation_denoising/data/"
    SAMPLE_COL = "sample"
    BATCH_COL = "batch"
    CONDITION_COL = "condition"
    CTRL_LAYER = "ctrl"

    COND_ORDER = dict(
        Mx_Seahorse=["NStim", "Stim", "2DG", "Oligo"],
        Lx_Glioblastoma=[
            "Naive_WT",
            "TMD_sM",
            "TMD_dM",
            "TMD_tM",
            "TMD_CD95_WT",
            "TMD_CD95_KO",
        ],
        Lx_Pancreatic_Cancer=["HPAF", "HPAC", "PSN1", "MiaPaca2"],
        Lx_HepaRG=["U", "F", "FI", "FIT"],
    )

    IMPUTATION_TYPES = {
        "ctrl": "imputation",
        "fancy": "imputation",
        "knn": "imputation",
        "bbmagic": "denoising",
        "MAGIC": "denoising",
        "ALRA": "denoising",
        "dca": "denoising",
    }

    IMPUTATION_GROUPS = {
        "ctrl": "ctrl",
        "ctrl_mean": "fixed value imputation",
        "fancy_multi": "MICE imputation",
        "fancy_itersvd": "SVD imputation",
        "knn_3": "kNN imputation",
        "knn_5": "kNN imputation",
        "MAGIC_t1": "kNN denoising",
        "MAGIC_t1": "kNN denoising",
        "ALRA": "SVD denoising",
        "dca_nb": "ML denoising",
    }

    COLORBLIND_COLORS = [
        "#006BA4",
        "#FF800E",
        "#595959",
        "#5F9ED1",
        "#C85200",
        "#898989",
        "#A2C8EC",
        "#FFBC79",
        "#CFCFCF",
    ]

    NARROW_BBOX = dict(
        boxstyle="round,pad=0.2", fc=(1, 1, 1, 0.3), ec=(0, 0, 0, 0), lw=0
    )

    EXCLUDE_IMPUTATION = [
        "dca_nb-conddisp",
        "dca_zinb-conddisp",
        "fancy_softbi",
        "bbmagic_1",
        "bbmagic_2",
        "bbmagic_3",
        "bbmagic_5",
        "fancy_soft",
        "knn_1",
        "MAGIC_t5",
        "MAGIC_t2",
        "ctrl_median",
        "ctrl_random",
        "fancy_iterative",
    ]
