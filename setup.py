#!/usr/bin/env python3

import setuptools

setuptools.setup(
    name="sc_imputation_denoising",
    version="0.0.1",
    author="Marius Klein",
    author_email="marius.klein@embl.de",
    packages=setuptools.find_packages(),
    description="Benchmarking framework for imputation and denoising in sc-metabolomics and lipidomics.",
    url="https://github.com/mariusrklein/sc_imputation_denoising",
    license='GPLv3',
    python_requires='>=3.0',
    install_requires=[
        "scanpy>=1.9.1",
        "scikit-learn",
        "seaborn",
        "scipy", 
    ]
)