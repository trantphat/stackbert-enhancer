# StackBERT-Enhancer

StackBERT-Enhancer is a deep learning framework for accurate identification and classification of crucial regulatory DNA sequences known as enhancers.

## Key Features

- **Enhancer Identification & Classification**: Distinguishes enhancer sequences from non-enhancers and classifies enhancer strength.
- **Stacked Transformer Models**: Multiple BERT-based models trained on different k-mer tokenizations, capturing sequence dependencies at various scales.
- **Stacking Ensemble**: Integrates individual models for improved accuracy, robustness, and generalization.
- **Interpretability**: SHAP for feature importance and attention score analysis for motif discovery.
- **Distributed Training**: Supports multi-GPU training for scalability.
- **Biological Insight**: Designed to bridge predictive performance with interpretability for biomedical research.

## Project Structure

- **model.py**: Core model definitions for sequence classification.
- **dnabert/**: DNABERT pre-trained model resources ([DNABERT README](dnabert/README.md)).
- **analysis/**: Jupyter notebooks for visualization and analysis.
- **data/**: Contains datasets.
- **outputs/**: Directory for model outputs, predictions, and intermediate files.
- **utils/**: Utility scripts and helper functions.
- **notebook/**: Notebooks for model training, evaluation, and ensemble methods.
- **logs/**: Training and evaluation logs.
- **scripts (.sh, .py)**: Shell and Python scripts for distributed and local training/tuning.

## Requirements

Install dependencies with:

```sh
pip install -r requirements.txt
```

## Usage
- **Training & Tuning**: Use the provided shell and Python scripts for model training and hyperparameter tuning.
- **Analysis**: Explore the `analysis/` and `notebook/` directories for motif analysis, interpretability, and ensemble evaluation.
- **Data**: Place your sequence data in the `data/` directory, following the provided format.
