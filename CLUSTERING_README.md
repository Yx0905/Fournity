# Latent-Sparse Clustering Analysis

This script implements a comprehensive 4-phase clustering workflow for the CHAMPIONS GROUP dataset.

## Overview

The clustering analysis follows a structured approach:

1. **Phase 1: Context & Meta-Data Analysis** - Feature profiling, sparsity checks, and metric selection
2. **Phase 2: Filtering & Encoding Engine** - Redundancy pruning, scaling, and dimensionality reduction
3. **Phase 3: Iterative Clustering Loop** - Hyperparameter optimization and stability testing
4. **Phase 4: Interpretability** - Feature importance, cluster profiling, and visualizations

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install clustering-specific dependencies:

```bash
pip install scikit-learn matplotlib seaborn scipy prince umap-learn shap scikit-learn-extra gower
```

For **K-Medoids + Gower** (when >50% categorical): `scikit-learn-extra`, `gower`.

## Usage

1. Make sure you have the processed dataset (`champions_group_processed.xlsx`) in the same directory.

2. Run the clustering analysis:

```bash
python3 clustering_analysis.py
```

## Output Files

The script generates:

1. **clustering_analysis_report.txt** - Comprehensive text report with all results
2. **umap_clusters.png** - UMAP 2D projection of clusters (if umap-learn is available)
3. **cluster_sizes.png** - Bar chart showing cluster size distribution
4. **feature_importance.png** - Feature importance visualization
5. **cluster_profiles_heatmap.png** - Heatmap comparing cluster characteristics

## Algorithm Selection Logic

The script automatically selects the appropriate algorithm based on data characteristics:

- **>50% categorical**: K-Medoids with **Gower distance** (requires `scikit-learn-extra`, `gower`)
- **Pure numerical AND n > 10,000**: MiniBatch K-Means on PCA components
- **Otherwise**: Standard K-Means on FAMD/PCA-reduced data

## Validation

- **Validation gate**: Proceeds only if Silhouette Score **> 0.5**. If **≤ 0.5**, logs a clear warning and suggests re-running with a Sparse Autoencoder or different features. Optionally, use `strict_validation_gate=True` in `LatentSparseClustering(...)` to **raise an error** and stop execution when Silhouette < 0.5.
- **Stability test**:
  - **K-Means**: 5-fold shuffle of the data; fit same *k* five times and check centroid stability (average center shift).
  - **K-Medoids**: 5 runs with different `random_state` on the same data; report mean ARI vs reference labeling.

## Notes

- Optional dependencies (FAMD, UMAP, SHAP, `scikit-learn-extra`, `gower`) are used when available; the script still runs without them (with fallbacks).
- Missing values: >50% missing → drop feature; otherwise impute.
- Redundant features (correlation > 0.9) are automatically removed.
- **Strict validation**: `LatentSparseClustering(..., strict_validation_gate=True)` stops with an error if Silhouette < 0.5.
