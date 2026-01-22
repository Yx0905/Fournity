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
pip install scikit-learn matplotlib seaborn scipy prince umap-learn shap
```

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

- **Mixed data (categorical + numerical) AND n < p**: Uses FAMD → Sparse K-Means
- **Pure numerical AND n > 10,000**: Uses MiniBatch K-Means on PCA components
- **Otherwise**: Uses Standard K-Means

## Validation

The script includes validation gates:

- **Silhouette Score > 0.5**: Proceeds to profiling
- **Silhouette Score ≤ 0.5**: Suggests re-running with Sparse Autoencoder
- **Stability Test**: 5-fold shuffle test to check cluster center stability

## Notes

- Some optional dependencies (FAMD, UMAP, SHAP) will be used if available but the script will work without them
- The script handles missing values automatically (>50% missing = drop, otherwise impute)
- Redundant features (correlation > 0.9) are automatically removed
