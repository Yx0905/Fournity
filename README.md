# AI-Driven Company Intelligence System
## SDS DATATHON 2026 - Category A Submission

A comprehensive Python-based machine learning system for analyzing company data to generate actionable business insights through data-driven segmentation, statistical analysis, and predictive modeling.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Data Format](#data-format)
- [Technical Details](#technical-details)
- [Output Files](#output-files)
- [Team Members](#team-members)

---

## 📊 Overview

This system transforms raw company attributes (financial metrics, employee data, technology infrastructure, industry classifications) into interpretable company intelligence. It provides:

- **Company Segmentation**: Identify and group companies with similar characteristics using advanced clustering algorithms
- **Pattern Analysis**: Discover key differences, similarities, and anomalies across company segments
- **Business Insights**: Generate data-grounded explanations for decision-making
- **Predictive Modeling**: Forecast company performance and predict segment membership
- **Statistical Validation**: Comprehensive statistical tests with proper assumption checking

### Problem Statement
Organizations need to understand company landscapes to make informed decisions about partnerships, investments, market positioning, and competitive strategies. This system automates the discovery of meaningful company segments and generates actionable insights from complex datasets.

---

## 🎯 Key Features

### 1. **Intelligent Data Processing**
- **Automatic Column Detection**: Pattern-based matching handles various naming conventions
- **Feature Engineering**: Derives 10+ business metrics from raw data
  - Revenue per Employee
  - Market Value per Employee
  - IT Spend Ratio
  - Technology Density Index
  - Company Age
  - Geographic Dispersion
  - Growth Potential Index
  - Maturity Stage Classification
- **Outlier Handling**: IQR-based winsorization (caps outliers instead of removing data)
- **Missing Value Imputation**: Median imputation for numeric features
- **Data Validation**: Filters inactive companies and validates data quality

### 2. **Advanced Clustering & Segmentation**

#### 4-Phase Latent-Sparse Clustering Workflow
**Phase 1: Context & Meta-Data Analysis**
- Feature profiling and sparsity checks
- Metric mapping and data quality assessment
- Automatic distance metric selection (Gower/Euclidean/Mixed)

**Phase 2: Filtering & Encoding Engine**
- Redundancy pruning (removes highly correlated features)
- Automated scaling (RobustScaler for outliers, StandardScaler otherwise)
- Dimensionality reduction (FAMD for mixed data, PCA for numerical)

**Phase 3: Iterative Clustering Loop**
- Hyperparameter optimization with grid search
- Validation and stability testing
- Multiple algorithm support (K-Means, K-Medoids, DBSCAN, HDBSCAN, GMM)
- Multi-metric validation (Silhouette, Davies-Bouldin, Calinski-Harabasz)

**Phase 4: Interpretability & Insights**
- Feature importance extraction using Random Forest and SHAP
- Cluster profiling with comprehensive statistics
- Enhanced visualizations with business context

#### Business-Optimized Cluster Selection
- Balances statistical quality with business practicality
- Prefers K=4-8 clusters for actionable segmentation
- Multi-metric scoring with configurable thresholds

### 3. **Machine Learning Models**

#### Logistic Regression (Cluster Prediction)
- Predicts segment membership for new companies
- Interpretable coefficients for business understanding
- 5-fold cross-validation for robust performance estimates
- Proper train/test split to prevent data leakage

#### Linear Regression (Performance Forecasting)
- Forecasts revenue or market value
- Regularization options (Ridge/Lasso/ElasticNet) to handle multicollinearity
- VIF (Variance Inflation Factor) calculation
- Feature importance ranking

### 4. **Statistical Analysis**

#### Chi-Square Tests
- Tests relationships between categorical variables and clusters
- Validates statistical assumptions (expected frequency ≥ 5)
- Multiple testing correction (Bonferroni/FDR-BH/FDR-BY)
- Comprehensive reporting with effect sizes

#### Dimensionality Reduction
- **PCA**: Principal Component Analysis for variance maximization
- **TruncatedSVD**: Efficient sparse matrix decomposition
- **t-SNE**: Non-linear manifold learning (perplexity scaled with dataset size)
- **UMAP**: Uniform Manifold Approximation (preserves global structure)
- **FAMD**: Factor Analysis of Mixed Data (handles categorical + numerical)

### 5. **Text Analytics**
- **TF-IDF Vectorization**: Extracts features from industry descriptions
- **Interpretable Features**: Uses actual term names instead of generic indices
- **Multi-Source Analysis**: Processes SIC, NAICS, and NACE descriptions

### 6. **Business Indicators**
Automatically calculates 10+ derived metrics:
1. Market Value/Revenue Ratio (Price-to-Sales) - Growth vs. value companies
2. IT Investment Intensity - Tech-forward vs. traditional segmentation
3. Single-Site Concentration Ratio - Geographic strategy indicator
4. Workforce Technology Ratio - Knowledge economy indicator
5. Technology Sophistication Index - IT maturity score
6. Mobile vs. Desktop Ratio - Remote work culture indicator
7. Growth Potential Index - Composite growth indicator
8. Company Maturity Stage - Startup/Growth/Mature/Established
9. Revenue Scale Category - Quantile-based segmentation
10. Employee Scale Category - Size-based classification

### 7. **Enhanced Visualizations**
- **PCA Clusters**: Scatter plots with cluster centroids and labels
- **Feature Comparison**: Violin plots + boxplots with statistical annotations
- **Cluster Heatmaps**: Normalized mean values across features
- **Interactive 3D Plots**: Plotly-based explorable visualizations
- **Explained Variance**: Component contribution analysis
- **Distribution Charts**: Cluster sizes and feature distributions

### 8. **LLM-Powered Insights (Optional)**
- **OpenAI/DeepSeek Integration**: Generates natural language insights
- **Automated Analysis**: Cluster characterization and business recommendations
- **Fallback System**: Rule-based insights if LLM unavailable
- **Temperature Control**: Lower temperature (0.3) for analytical output

---

## 🏗️ System Architecture

### Project Structure
```
Fournity/
├── company_intelligence.py       # Main analysis engine (3100+ lines)
├── clustering_analysis.py        # 4-phase clustering workflow (2700+ lines)
├── process_champions_data.py     # Data preprocessing script
├── generate_report.py            # Quick report generation
├── visualization_improvements.py # Enhanced graphics module
├── company_intelligence.ipynb    # Interactive Jupyter notebook
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── champions_group_data.xlsx     # Input dataset
└── [Generated outputs]           # Reports, CSVs, visualizations
```

### Module Breakdown

**1. `process_champions_data.py`** (271 lines)
- Preprocesses raw company data
- Calculates derived business metrics
- Removes duplicates and invalid entries
- Exports to `champions_group_processed.xlsx`

**2. `clustering_analysis.py`** (2750+ lines)
- Implements 4-phase Latent-Sparse Clustering
- Supports multiple algorithms (K-Means, K-Medoids, DBSCAN, HDBSCAN, GMM)
- Feature importance with Random Forest and SHAP
- Comprehensive validation metrics
- Persona generation

**3. `company_intelligence.py`** (3100+ lines)
**Main Class: `CompanyIntelligence`**
- Data loading and exploration
- Feature engineering and preprocessing
- Clustering and segmentation
- Statistical analysis (Chi-square, VIF)
- Machine learning (Logistic/Linear regression)
- Dimensionality reduction (PCA, t-SNE, UMAP)
- Visualization generation
- Report generation

**4. `visualization_improvements.py`** (412 lines)
- Enhanced PCA visualizations
- Feature comparison violin plots
- Cluster heatmaps
- Statistical annotations (ANOVA)

**5. `generate_report.py`** (64 lines)
- Quick script to generate complete analysis report
- Simple command-line interface

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Navigate to project directory**
   ```bash
   cd Fournity
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Optional: Set up API key for LLM insights**
   
   Create a `.env` file in the project directory:
   ```
   # For DeepSeek
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   
   # OR for OpenAI
   OPENAI_API_KEY=your_openai_api_key_here
   ```

---

## 🚀 Usage

### Quick Start - Three Ways to Run

#### Option 1: Quick Report Generation (Recommended)
```bash
python generate_report.py
```
This automatically:
- Loads `champions_group_data.xlsx`
- Runs complete analysis pipeline
- Generates comprehensive report
- Creates visualizations
- Exports results with cluster labels

#### Option 2: Command Line
```bash
# Use default data file
python company_intelligence.py

# Or specify custom file
python company_intelligence.py path/to/your/data.xlsx
```

#### Option 3: Jupyter Notebook (Interactive)
```bash
jupyter notebook company_intelligence.ipynb
```

### Data Processing Pipeline

**Step 1: Process Raw Data (Optional)**
```bash
python process_champions_data.py
```
Processes raw data and generates derived metrics.

**Step 2: Clustering Analysis (Optional)**
```bash
python clustering_analysis.py
```
Runs comprehensive 4-phase clustering workflow.

**Step 3: Company Intelligence Analysis**
```bash
python generate_report.py
```
Generates complete analysis report.

### Programmatic Usage

```python
from company_intelligence import CompanyIntelligence

# Initialize analyzer
analyzer = CompanyIntelligence(
    data_path='champions_group_data.xlsx',
    api_key='your_api_key'  # Optional
)

# Option A: Run complete pipeline (recommended)
results = analyzer.run_full_analysis(n_clusters=None)  # None = auto-determine

# Option B: Step-by-step analysis
# Step 1: Explore data
analyzer.explore_data()

# Step 2: Preprocess (automatically calculates business indicators)
analyzer.preprocess_data(
    calculate_indicators=True,
    include_key_indicators_in_clustering=True
)

# Step 3: Determine optimal clusters
optimal_k = analyzer.determine_optimal_clusters(max_k=10, practical_threshold=0.70)

# Step 4: Perform clustering
analyzer.perform_clustering(n_clusters=optimal_k)

# Step 5: Analyze clusters
cluster_analysis = analyzer.analyze_clusters()

# Step 6: Compare clusters
comparison = analyzer.compare_clusters()

# Step 7: Identify patterns
patterns = analyzer.identify_patterns()

# Step 8: Statistical tests
chi_square_results = analyzer.perform_chi_square_test(correction_method='fdr_bh')

# Step 9: Train predictive models
lr_results = analyzer.train_logistic_regression(use_original_features=True, cv_folds=5)
linear_results = analyzer.train_linear_regression(
    target_feature='revenue',
    regularization='ridge',
    check_multicollinearity=True
)

# Step 10: Generate insights
insights = analyzer.generate_llm_insights(cluster_analysis, patterns)

# Step 11: Visualize
analyzer.visualize_results()

# Step 12: Generate report
report = analyzer.generate_report(cluster_analysis, patterns, insights)

# Step 13: Export results
analyzer.df.to_csv('companies_with_segments.csv', index=False)
```

---

## 📈 Data Format

### Expected Input Format
Excel (.xlsx, .xls) or CSV format with the following columns:

**Required Numeric Columns:**
- Revenue (USD)
- Market Value (USD)
- Employee Total
- Employee Single Sites
- IT Budget / IT Spend
- Technology Infrastructure (PCs, Servers, Storage, Routers, Laptops, Desktops)
- Year Founded

**Optional Text Columns (for TF-IDF):**
- SIC Description
- NAICS Description
- NACE Rev 2 Description

**Optional Categorical Columns (for Chi-square):**
- Entity Type
- Ownership Type
- Manufacturing Status
- Import/Export Status
- Franchise Status

**Note:** The system uses intelligent pattern matching to find columns, so exact names aren't required.

---

## 🔬 Technical Details

### Algorithms & Methods

**Clustering:**
- K-Means (default) - Fast, scalable centroid-based
- K-Medoids - Robust to outliers, uses actual data points
- DBSCAN - Density-based, finds arbitrary shapes
- HDBSCAN - Hierarchical density-based, automatic cluster count
- GMM (Gaussian Mixture Model) - Probabilistic soft clustering

**Validation Metrics:**
- Silhouette Score (cluster quality: -1 to 1, higher is better)
- Davies-Bouldin Index (separation: lower is better)
- Calinski-Harabasz Index (variance ratio: higher is better)

**Dimensionality Reduction:**
- PCA (Principal Component Analysis) - Linear, variance maximization
- TruncatedSVD (Singular Value Decomposition) - Efficient for sparse matrices
- t-SNE (t-Distributed Stochastic Neighbor Embedding) - Non-linear, local structure
- UMAP (Uniform Manifold Approximation) - Non-linear, preserves global structure
- FAMD (Factor Analysis of Mixed Data) - Handles categorical + numerical

**Regression:**
- Logistic Regression (multinomial, L2 regularization)
- Linear Regression with Ridge/Lasso/ElasticNet regularization

**Statistical Tests:**
- Chi-Square Test (categorical associations)
- ANOVA (feature differences across clusters)
- VIF (multicollinearity detection)

**Feature Engineering:**
- TF-IDF (text feature extraction)
- Business indicator derivation
- IQR-based outlier winsorization

### Statistical Best Practices

1. **No Data Leakage**: Proper train/test split before scaling
2. **Outlier Handling**: Winsorization (capping) instead of removal
3. **Multicollinearity Detection**: VIF calculation before regression
4. **Regularization**: Ridge/Lasso to handle correlated features
5. **Cross-Validation**: 5-fold CV for stable metrics
6. **Assumption Validation**: Chi-square expected frequency checks
7. **Multiple Testing Correction**: Prevents false discoveries
8. **Interpretable Models**: Original features for logistic regression

---

## 📊 Output Files

### Generated Files

**Data Processing:**
- `champions_group_processed.xlsx` - Processed dataset with derived metrics

**Clustering Analysis:**
- `clustering_analysis_report.txt` - Comprehensive 4-phase workflow report
- Various PNG files (if generated during clustering)

**Company Intelligence:**
- `company_intelligence_report.txt` - Comprehensive analysis report
- `optimal_clusters.png` - Cluster validation metrics
- `pca_clusters.png` - PCA visualization with centroids
- `pca_clusters_enhanced.png` - Enhanced PCA with feature importance
- `feature_comparison.png` - Feature comparison boxplots
- `feature_comparison_enhanced.png` - Enhanced violin plots with ANOVA
- `cluster_heatmap_enhanced.png` - Normalized cluster characteristics
- `interactive_clusters.html` - Interactive 3D visualization
- `companies_with_segments.csv` - Data with cluster labels

---

## 🎯 Use Cases

### 1. Sales Intelligence
- Segment prospects by characteristics
- Prioritize high-value segments
- Understand customer profiles
- Predict segment membership for new leads

### 2. Risk Assessment
- Identify risk patterns by segment
- Detect anomalies and outliers
- Benchmark against similar companies
- Assess technology dependency

### 3. Market Research
- Understand industry structure
- Identify market segments
- Analyze competitive landscape
- Growth potential analysis

### 4. Strategic Planning
- Identify growth opportunities
- Understand operational differences
- Support M&A analysis
- Technology investment planning

---

## ⚙️ Configuration

### Clustering Parameters
```python
# Customize cluster selection
optimal_k = analyzer.determine_optimal_clusters(
    max_k=10,
    practical_threshold=0.75  # Adjust for business practicality
)

# Force specific number of clusters
analyzer.perform_clustering(n_clusters=5)
```

### Regression Parameters
```python
# Linear regression with regularization
linear_results = analyzer.train_linear_regression(
    target_feature='revenue',
    regularization='ridge',  # 'ridge', 'lasso', 'elasticnet', or 'none'
    alpha=1.0,  # Regularization strength
    check_multicollinearity=True
)
```

### Business Indicators
```python
# Calculate additional indicators
analyzer.calculate_business_indicators()

# Include in clustering
analyzer.preprocess_data(
    calculate_indicators=True,
    include_key_indicators_in_clustering=True
)
```

---

## 🐛 Troubleshooting

### Common Issues

**1. ModuleNotFoundError**
```bash
pip install -r requirements.txt
```

**2. Memory Issues**
- Reduce number of features
- Use sampling for initial exploration
- Increase system memory

**3. LLM Insights Not Working**
- Check API key is set correctly
- Verify internet connection
- System falls back to rule-based insights automatically

**4. Poor Clustering Results**
- Review data quality
- Try different number of clusters
- Check for data normalization issues
- Verify business indicators are meaningful

**5. Statistical Test Warnings**
- Chi-square assumption violations: Consider combining categories
- High VIF scores: Use Ridge/Lasso regression
- Multiple testing: Results are automatically corrected

---

## 📚 Dependencies

### Core Libraries
- pandas (≥2.0.0) - Data manipulation
- numpy (≥1.24.0) - Numerical computing
- scikit-learn (≥1.3.0) - Machine learning

### Clustering & Advanced Analysis
- scikit-learn-extra (≥0.3.0) - K-Medoids
- prince (≥0.12.0) - FAMD
- umap-learn (≥0.5.4) - UMAP
- hdbscan (≥0.8.33) - HDBSCAN

### Visualization
- matplotlib (≥3.7.0) - Static plots
- seaborn (≥0.12.0) - Statistical graphics
- plotly (≥5.17.0) - Interactive plots

### Statistical Analysis
- scipy (≥1.10.0) - Statistical functions
- statsmodels (≥0.14.0) - Advanced statistics

### Explainability
- shap (≥0.42.0) - SHAP values

### Distance Metrics
- gower (≥0.1.2) - Gower distance

### LLM Integration (Optional)
- openai (≥1.0.0) - OpenAI/DeepSeek API
- python-dotenv (≥1.0.0) - Environment variables

---

## 👥 Team Members

**Team Fournity - SDS DATATHON 2026**

This project was developed for the SDS DATATHON 2026 - Category A competition, focusing on AI-driven company intelligence and segmentation.

---

## 📝 License

This project is provided for the SDS DATATHON 2026 competition.

---

## 🙏 Acknowledgments

Special thanks to:
- SDS (Singapore Data Science Society) for organizing the DATATHON 2026
- Champions Group for providing the dataset
- Open-source community for the amazing libraries

---

**Last Updated**: January 2026  
**Version**: 3.0 (Comprehensive Documentation)  
**Competition**: SDS DATATHON 2026 - Category A
