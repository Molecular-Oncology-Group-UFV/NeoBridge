# NeoBridge

NeoBridge is a hybrid ensemble machine learning framework for predicting neoantigen immunogenicity, combining gradient boosting (XGBoost) with a regularized deep neural network (PyTorch MLP). It integrates four biologically grounded feature dimensions — HLA binding and processing, differential agretopicity, tumor neoantigen abundance, and TCR recognition potential.

The framework was trained on **NeoPepBase**, a curated dataset of 11,188 experimentally validated neoantigens from pan-cancer cohorts, and validated on an independent cohort of 180 pancreatic ductal adenocarcinoma (PDAC) patients, integrating immunogenicity predictions with phylogenetic reconstruction of clonal tumor architecture.

## Table of Contents
* [Overview](#overview)
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Usage](#usage)
* [Data Availability](#data-availability)
* [Reproducibility](#reproducibility)
* [Citation](#citation)
* [License](#license)

## Overview
* **Model:** Heterogeneous soft-voting ensemble (XGBoost + PyTorch MLP)
* **Training data:** NeoPepBase (11,188 neoantigens; 23.4% immunogenic)
* **Validation cohort:** 180 PDAC patients (ICGC ARGO: PACA-CA, APGI-AU)
* **Pipeline:** Feature matrix integration → neoantigen prediction → immunogenicity scoring → clonal architecture integration (PhyClone)

## Repository Structure
```text
NeoBridge/
├── src/
│   ├── models/              # XGBoost, MLP, and ensemble implementations
│   ├── pipeline_features/   # Feature engineering (HLA binding, agretopicity, TCR recognition)
│   ├── evaluation/          # Cross-cohort validation, benchmarking against ImmuneMirror/NeoRanking
│   └── visualization/       # Figure generation (Matplotlib/Seaborn), SHAP analysis
├── notebooks/               # Experimental layer: model selection, hypothesis testing
├── models/
│   └── production/          # Trained model weights, scaler.pkl, optimal_configuration.pkl
├── results/
│   └── figures/             # Output figures (e.g., feature_importance)
├── environment.yml          # Conda environment specification
└── README.md
```

## Installation

```bash
git clone https://github.com/Molecular-Oncology-Group-UFV/NeoBridge.git
cd NeoBridge
conda env create -f environment.yml
conda activate neobridge
```
**Core dependencies:** Python 3.10, PyTorch (v2.x), XGBoost (v2.x), scikit-learn, Optuna, SHAP.

## Usage
*(Complete with your actual entry-point commands / example scripts.)*

```bash
# Example: run inference on a preprocessed feature matrix
python src/models/predict.py --input data/example_features.csv --output results/predictions.csv
```

## Data Availability
Curated datasets (feature matrices, immunogenicity labels, cohort annotations, HLA allele mappings), anonymized to comply with ethical standards, are deposited at Figshare:
**DOI:** 10.6084/m9.figshare.32065212

## Reproducibility
Trained model weights, fitted scalers, and configuration files are provided under `models/production/` so that new data are transformed using the exact statistics derived from the original training set. All figures are generated programmatically and are directly traceable to source data.

## Citation
If you use NeoBridge in your research, please cite:
*[Complete with full citation once accepted/published]*

## License
*[Complete with the license under which the repository is released]*
