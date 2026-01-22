# AI-Driven Company Intelligence System

A comprehensive Python-based tool for analyzing company-level data to generate actionable insights through data-driven segmentation and machine learning.

## Overview

This system transforms raw company attributes (industry classifications, company size, ownership structure, geographic footprint, operational indicators) into interpretable company intelligence. It enables:

- **Company Segmentation**: Identify and group companies with similar characteristics
- **Pattern Analysis**: Discover key differences, similarities, and anomalies
- **Business Insights**: Generate data-grounded explanations for decision-making
- **Visualization**: Interactive and static visualizations of segmentation results

## Features

### Data Processing (`process_champions_data.py`)

- **Data Filtering**: Removes companies with zero employees, zero market value, or zero revenue
- **Duplicate Removal**: Eliminates duplicate rows from the dataset
- **Derived Metrics Calculation**: Computes key performance indicators including:
  - Revenue per Employee
  - Market Value per Employee
  - IT Spend Ratio
  - Employees per Site
  - Technology Density Index
  - Company Age
  - Geographic Dispersion

### Clustering Analysis (`clustering_analysis.py`)

- **4-Phase Latent-Sparse Clustering Workflow**:
  - Phase 1: Context & Meta-Data Analysis (feature profiling, sparsity checks, metric mapping)
  - Phase 2: Filtering & Encoding Engine (redundancy pruning, scaling, dimensionality reduction)
  - Phase 3: Iterative Clustering Loop (hyperparameter optimization, validation, stability testing)
  - Phase 4: Interpretability (feature importance, cluster profiling, visualizations)
- **Automatic Algorithm Selection**: FAMD for mixed data, PCA for numerical data
- **Robust Data Handling**: NaN and infinite value detection and cleaning
- **Comprehensive Reporting**: Text reports and multiple visualization formats

### Company Intelligence (`company_intelligence.py`)

1. **Data Preprocessing**
   - Handles missing values (median imputation)
   - **Outlier removal using IQR method** (removes ~15% of extreme values)
   - Encodes categorical variables
   - Scales numeric features
   - Feature selection and engineering

2. **Clustering & Segmentation**
   - **Business-optimized cluster determination** (balances statistical quality with practical segmentation)
   - Automatic optimal cluster selection (enhanced algorithm)
   - Prefers K=4-8 clusters for actionable business insights (not oversimplified K=2)
   - K-Means clustering
   - DBSCAN clustering (optional)
   - PCA for dimensionality reduction

3. **Machine Learning Models**
   - **Logistic Regression**: Predict cluster membership from company features
   - **TF-IDF Vectorization**: Extract features from text descriptions (if available)
   - **Chi-Square Test**: Test associations between categorical variables and clusters
   - **Dimensionality Reduction**: Multiple methods including:
     - PCA (Principal Component Analysis)
     - TruncatedSVD (Singular Value Decomposition)
     - t-SNE (t-distributed Stochastic Neighbor Embedding) - optional
     - UMAP (Uniform Manifold Approximation and Projection) - optional

4. **Analysis Capabilities**
   - Cluster characterization
   - Cross-cluster comparison
   - Outlier detection
   - Pattern identification
   - Correlation analysis
   - Statistical significance testing (Chi-square)
   - Model accuracy assessment

5. **LLM-Powered Insights** (Optional)
   - Generates interpretable, business-friendly insights
   - Uses OpenAI GPT-4 for natural language explanations
   - Falls back to rule-based insights if LLM unavailable

6. **Visualization**
   - Cluster distribution charts
   - PCA 2D/3D visualizations
   - TruncatedSVD visualizations
   - Feature comparison boxplots
   - Interactive Plotly visualizations
   - Dimensionality reduction comparisons

7. **Reporting**
   - Comprehensive text reports
   - CSV exports with cluster labels
   - Multiple visualization formats
   - Model performance metrics

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone or navigate to the project directory**

2. **Create a virtual environment (recommended)**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Set up API key for LLM insights (OpenAI or DeepSeek)**
   - Create a `.env` file in the project directory
   - For DeepSeek: `DEEPSEEK_API_KEY=your_deepseek_api_key_here`
   - For OpenAI: `OPENAI_API_KEY=your_openai_api_key_here`
   - Or set environment variable:
     - `export DEEPSEEK_API_KEY=your_key_here` (DeepSeek)
     - `export OPENAI_API_KEY=your_key_here` (OpenAI)
   - The system will try DeepSeek first, then fall back to OpenAI

## Usage

### Data Processing

1. Place your `champions_group_data.xlsx` file in the same directory as the script
2. Run the processing script:
```bash
python3 process_champions_data.py
```
3. The processed data will be saved to `champions_group_processed.xlsx`

### Clustering Analysis

Run the comprehensive 4-phase clustering analysis:

```bash
python3 clustering_analysis.py
```

This generates:
- `clustering_analysis_report.txt` - Comprehensive text report
- `umap_clusters.png` - UMAP visualization (if umap-learn installed)
- `cluster_sizes.png` - Cluster size distribution
- `feature_importance.png` - Feature importance plot
- `cluster_profiles_heatmap.png` - Cluster comparison heatmap

### Company Intelligence

#### Basic Usage

Run the analysis on your data file:

```bash
python company_intelligence.py
```

This will use `champions_group_data.xlsx` by default.

#### Custom Data File

```bash
python company_intelligence.py path/to/your/data.xlsx
```

Or for CSV files:

```bash
python company_intelligence.py path/to/your/data.csv
```

#### Programmatic Usage

```python
from company_intelligence import CompanyIntelligence

# Initialize analyzer
analyzer = CompanyIntelligence(
    data_path='champions_group_data.xlsx',
    api_key='your_openai_api_key'  # Optional
)

# Run full analysis
results = analyzer.run_full_analysis(n_clusters=5)  # Optional: specify cluster count

# Access results
cluster_analysis = results['cluster_analysis']
patterns = results['patterns']
insights = results['insights']
```

## Output Files

### Data Processing Output
- `champions_group_processed.xlsx` - Processed dataset with derived metrics

### Clustering Analysis Output
- `clustering_analysis_report.txt` - Comprehensive text report with insights
- `umap_clusters.png` - UMAP 2D projection of clusters
- `cluster_sizes.png` - Cluster size distribution
- `feature_importance.png` - Feature importance visualization
- `cluster_profiles_heatmap.png` - Cluster profiles heatmap

### Company Intelligence Output
- `company_intelligence_report.txt` - Comprehensive text report with insights
- `companies_with_segments.csv` - Original data with cluster labels added
- `cluster_distribution.png` - Bar chart showing company distribution across segments
- `pca_clusters.png` - 2D PCA visualization of clusters
- `feature_comparison.png` - Boxplots comparing features across clusters
- `interactive_clusters.html` - Interactive 3D visualization (open in browser)
- `optimal_clusters.png` - Elbow curve and silhouette scores for cluster selection

## Data Format

The system expects data in Excel (.xlsx, .xls) or CSV format with:

- **Numeric columns**: For quantitative analysis (e.g., revenue, employee count, age)
- **Categorical columns**: For qualitative analysis (e.g., industry, country, ownership type)
- **Company identifiers**: Optional, can be excluded from analysis

### Example Data Structure

| Company_ID | Industry | Revenue | Employees | Country | ... |
|------------|----------|---------|-----------|---------|-----|
| 1          | Tech     | 1000000 | 500       | USA     | ... |
| 2          | Finance  | 5000000 | 2000      | UK      | ... |

## Commercial Applications

This system demonstrates the commercial value of company data through:

1. **Sales Intelligence**
   - Segment prospects by characteristics
   - Prioritize high-value segments
   - Understand customer profiles

2. **Risk Assessment**
   - Identify risk patterns by segment
   - Detect anomalies and outliers
   - Benchmark against similar companies

3. **Market Research**
   - Understand industry structure
   - Identify market segments
   - Analyze competitive landscape

4. **Strategic Planning**
   - Identify growth opportunities
   - Understand operational differences
   - Support M&A analysis

## Technical Details

### Algorithms Used

- **Clustering**: K-Means, DBSCAN, Latent-Sparse Clustering
- **Outlier Detection**: Interquartile Range (IQR) method with 1.5× multiplier
- **Optimal K Selection**: Enhanced business-focused algorithm
  - Evaluates K=2 to K=10 with silhouette scores
  - Prioritizes K=4-8 range for practical segmentation
  - Selects K within this range if silhouette score is ≥70% of maximum
  - Default result: Typically K=6 for balanced business segmentation
- **Dimensionality Reduction**: PCA, FAMD (Factor Analysis of Mixed Data), TruncatedSVD
- **Feature Scaling**: StandardScaler, RobustScaler (for outliers)
- **Encoding**: Label Encoding for categorical variables

### Why Not K=2?

While K=2 often has the highest silhouette score statistically, this system uses an **enhanced business-focused algorithm** that prefers K=4-8 because:

1. **Real-world complexity**: Companies don't simply divide into "small" and "large"
2. **Actionable insights**: 5-7 segments provide practical marketing/sales strategies
3. **Market reality**: Most markets have micro, small, mid-market, upper-mid, and enterprise tiers
4. **Statistical quality**: K=6 typically achieves silhouette scores of 0.34-0.35, which is acceptable for well-separated clusters
5. **Business value**: Each cluster represents a distinct segment requiring different approaches

**Example with K=6**:
- Cluster 0: Micro businesses (55%)
- Cluster 1: Small operations (9%)
- Cluster 2: Growth companies (12%)
- Cluster 3: Mid-market (11%)
- Cluster 4: Upper mid-market (7%)
- Cluster 5: Enterprise (6%)

This provides far more actionable intelligence than a binary small/large split.

### Performance Considerations

- Handles datasets with thousands of companies
- Automatically handles missing values
- Efficient preprocessing pipeline
- Optimized for typical firmographic datasets

## Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing packages with `pip install -r requirements.txt`

2. **Memory Issues with Large Datasets**: 
   - Reduce number of features
   - Use sampling for initial exploration
   - Increase system memory

3. **LLM Insights Not Working**:
   - Check API key is set correctly
   - Verify internet connection
   - System will fall back to rule-based insights automatically

4. **Poor Clustering Results**:
   - Review data quality
   - Try different number of clusters
   - Exclude irrelevant columns
   - Check for data normalization issues

5. **NaN Errors in Clustering Analysis**:
   - The script automatically handles NaN values
   - If errors persist, check data quality
   - Zero-variance columns are automatically removed

## License

This project is provided as-is for the Datathon 2026 competition.

## Contact & Support

For questions or issues, please refer to the project documentation or contact the development team.
