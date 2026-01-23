# AI-Driven Company Intelligence System

A comprehensive Python-based machine learning system for analyzing company data to generate actionable business insights through data-driven segmentation, statistical analysis, and predictive modeling.

## 📋 Overview

This system transforms raw company attributes (financial metrics, employee data, technology infrastructure, industry classifications) into interpretable company intelligence. It enables:

- **Company Segmentation**: Identify and group companies with similar characteristics using advanced clustering algorithms
- **Pattern Analysis**: Discover key differences, similarities, and anomalies across company segments
- **Business Insights**: Generate data-grounded explanations for decision-making
- **Predictive Modeling**: Forecast company performance and predict segment membership
- **Statistical Validation**: Comprehensive statistical tests with proper assumption checking

## 🏗️ System Architecture

The system consists of three main components:

### 1. **Data Processing** (`process_champions_data.py`)
Preprocesses raw company data and calculates derived business metrics.

**Key Features:**
- Removes duplicate rows and invalid entries (zero employees, revenue, or market value)
- Calculates derived metrics:
  - Revenue per Employee
  - Market Value per Employee
  - IT Spend Ratio
  - Employees per Site
  - Technology Density Index
  - Company Age
  - Geographic Dispersion
- Handles multiple column name variations (case-insensitive pattern matching)

### 2. **Clustering Analysis** (`clustering_analysis.py`)
Implements a comprehensive 4-phase Latent-Sparse Clustering workflow.

**4-Phase Workflow:**
- **Phase 1: Context & Meta-Data Analysis**
  - Feature profiling and sparsity checks
  - Metric mapping and data quality assessment
- **Phase 2: Filtering & Encoding Engine**
  - Redundancy pruning
  - Scaling and normalization
  - Dimensionality reduction (FAMD for mixed data, PCA for numerical)
- **Phase 3: Iterative Clustering Loop**
  - Hyperparameter optimization
  - Validation and stability testing
  - Multiple algorithm support (K-Means, K-Medoids, DBSCAN, HDBSCAN)
- **Phase 4: Interpretability**
  - Feature importance extraction
  - Cluster profiling
  - Comprehensive visualizations

### 3. **Company Intelligence** (`company_intelligence.py`)
Main analysis engine with comprehensive ML and statistical capabilities.

**Core Class: `CompanyIntelligence`**

## 🔧 Key Features

### Data Preprocessing
- **Intelligent Column Matching**: Pattern-based column detection handles various naming conventions
- **Feature Engineering**: 
  - Company age calculation from year founded
  - Business indicator derivation (Market Value/Revenue, IT Investment Intensity)
  - Range parsing (handles "X to Y", "X-Y", "<X", ">X" formats)
- **Outlier Handling**: IQR method with winsorization (caps outliers instead of removing rows)
- **Missing Value Imputation**: Median imputation for numeric features
- **Data Validation**: Year validation (checks for future dates, invalid years)

### Clustering & Segmentation
- **Multi-Metric Validation**: 
  - Silhouette Score (cluster quality)
  - Davies-Bouldin Index (cluster separation)
  - Calinski-Harabasz Index (variance ratio)
- **Business-Optimized K Selection**: 
  - Balances statistical quality with business practicality
  - Prefers K=4-8 for actionable segmentation
  - Configurable threshold (default: 70% of max score)
- **Clustering Algorithms**: K-Means (default), DBSCAN (optional)

### Machine Learning Models

#### Logistic Regression (Cluster Prediction)
- **Interpretable Features**: Uses original features (not PCA) for interpretable coefficients
- **Cross-Validation**: 5-fold CV for stable performance estimates
- **Data Leakage Prevention**: Proper train/test split before scaling
- **Feature Importance**: Maps coefficients to actual feature names

#### Linear Regression (Performance Forecasting)
- **Regularization**: Ridge (default), Lasso, or ElasticNet to handle multicollinearity
- **VIF Check**: Variance Inflation Factor calculation to detect multicollinearity
- **Proper Data Splitting**: Split before scaling to prevent data leakage
- **Target Features**: Revenue or Market Value prediction

### Statistical Analysis

#### Chi-Square Tests
- **Assumption Validation**: Checks expected frequency ≥ 5 requirement
- **Multiple Testing Correction**: Bonferroni, FDR-BH, or FDR-BY methods
- **Comprehensive Reporting**: Detailed statistics and significance testing

#### Dimensionality Reduction
- **PCA**: Principal Component Analysis
- **TruncatedSVD**: Singular Value Decomposition
- **t-SNE**: Scaled perplexity based on dataset size (`sqrt(n_samples)`)
- **UMAP**: Uniform Manifold Approximation (optional)

### Text Analysis
- **TF-IDF Vectorization**: Extracts features from industry descriptions (SIC, NAICS, NACE)
- **Interpretable Features**: Uses actual term names instead of generic "TFIDF_0"
- **Separate Processing**: Can be applied per text source (future enhancement)

### Business Indicators
Automatically calculates 10+ derived business metrics:

1. **Market Value to Revenue Ratio** - Identifies growth vs. value companies
2. **IT Investment Intensity** - Tech-forward vs. traditional segmentation
3. **Single-Site Concentration Ratio** - Geographic strategy indicator
4. **Workforce Technology Ratio** - Knowledge economy indicator
5. **Technology Sophistication Index** - IT maturity score
6. **Mobile vs. Desktop Ratio** - Remote work culture indicator
7. **Growth Potential Index** - Composite growth indicator
8. **Company Maturity Stage** - Startup/Growth/Mature/Established (configurable thresholds)
9. **Revenue Scale Category** - Quantile-based segmentation (Micro/Small/Mid-Market/Enterprise)
10. **Employee Scale Category** - Size-based classification

### LLM-Powered Insights (Optional)
- **OpenAI/DeepSeek Integration**: Generates natural language insights
- **Temperature Control**: Lower temperature (0.3) for deterministic analytical output
- **Fallback**: Rule-based insights if LLM unavailable
- **Auto-Detection**: Automatically tries DeepSeek first, then OpenAI

### Visualization
- Cluster distribution charts
- PCA 2D/3D visualizations
- Feature comparison boxplots
- Interactive Plotly visualizations
- Dimensionality reduction comparisons
- Model performance plots

## 📦 Installation

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

4. **Optional: Install advanced statistical libraries**
   ```bash
   pip install statsmodels  # For VIF and multiple testing correction
   ```

5. **Optional: Set up API key for LLM insights**
   - Create a `.env` file in the project directory
   - For DeepSeek: `DEEPSEEK_API_KEY=your_deepseek_api_key_here`
   - For OpenAI: `OPENAI_API_KEY=your_openai_api_key_here`
   - Or set environment variables:
     ```bash
     export DEEPSEEK_API_KEY=your_key_here
     export OPENAI_API_KEY=your_key_here
     ```

## 🚀 Usage

### Quick Start

#### 1. Data Processing
```bash
python3 process_champions_data.py
```
Processes raw data and generates `champions_group_processed.xlsx` with derived metrics.

#### 2. Clustering Analysis
```bash
python3 clustering_analysis.py
```
Runs comprehensive 4-phase clustering workflow.

#### 3. Company Intelligence Analysis

**Command Line:**
```bash
python company_intelligence.py
```
Uses `champions_group_data.xlsx` by default.

**Custom Data File:**
```bash
python company_intelligence.py path/to/your/data.xlsx
```

**Jupyter Notebook:**
```bash
jupyter notebook company_intelligence.ipynb
```

### Programmatic Usage

```python
from company_intelligence import CompanyIntelligence

# Initialize analyzer
analyzer = CompanyIntelligence(
    data_path='champions_group_data.xlsx',
    api_key='your_api_key'  # Optional
)

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
lr_results = analyzer.train_logistic_regression(
    use_original_features=True,  # Interpretable coefficients
    cv_folds=5
)

linear_results = analyzer.train_linear_regression(
    target_feature='revenue',
    regularization='ridge',  # or 'lasso', 'elasticnet'
    check_multicollinearity=True
)

# Step 10: Generate insights
insights = analyzer.generate_llm_insights(cluster_analysis, patterns)

# Step 11: Visualize
analyzer.visualize_results()

# Step 12: Generate comprehensive report
report = analyzer.generate_report(cluster_analysis, patterns, insights)

# Or run complete pipeline
results = analyzer.run_full_analysis(n_clusters=None)
```

## 📊 Output Files

### Data Processing
- `champions_group_processed.xlsx` - Processed dataset with derived metrics

### Clustering Analysis
- `clustering_analysis_report.txt` - Comprehensive text report
- Various visualization PNG files (if generated)

### Company Intelligence
- `company_intelligence_report.txt` - Comprehensive analysis report
- `optimal_clusters.png` - Cluster validation metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)
- `pca_clusters.png` - PCA visualization
- `feature_comparison.png` - Feature comparison across clusters
- `interactive_clusters.html` - Interactive 3D visualization
- `companies_with_segments.csv` - Data with cluster labels (if exported)

## 🔬 Technical Details

### Algorithms & Methods

**Clustering:**
- K-Means (default)
- DBSCAN (optional)
- Multi-metric validation (Silhouette, Davies-Bouldin, Calinski-Harabasz)

**Dimensionality Reduction:**
- PCA (Principal Component Analysis)
- TruncatedSVD (Singular Value Decomposition)
- t-SNE (perplexity scaled with dataset size)
- UMAP (optional)

**Regression:**
- Logistic Regression (multinomial, interpretable features)
- Linear Regression with regularization (Ridge/Lasso/ElasticNet)

**Statistical Tests:**
- Chi-Square Test (with assumption validation)
- Multiple testing correction (Bonferroni/FDR)

**Feature Engineering:**
- TF-IDF vectorization (text features)
- Business indicator calculation
- Outlier winsorization (IQR method)

### Statistical Improvements

The system implements best practices for statistical analysis:

1. **No Data Leakage**: Proper train/test split before scaling
2. **Outlier Handling**: Winsorization (capping) instead of removal
3. **Multicollinearity Detection**: VIF calculation before regression
4. **Regularization**: Ridge/Lasso to handle correlated features
5. **Cross-Validation**: 5-fold CV for stable metrics
6. **Assumption Validation**: Chi-square expected frequency checks
7. **Multiple Testing Correction**: Prevents false discoveries
8. **Interpretable Models**: Original features for logistic regression

### Business Indicators

**Automatically Calculated:**
- Market Value/Revenue Ratio (growth vs. value)
- IT Investment Intensity (tech-forward vs. traditional)
- Single-Site Concentration (geographic strategy)
- Workforce Technology Ratio (knowledge economy)
- Technology Sophistication Index (IT maturity)
- Growth Potential Index (composite growth score)
- Maturity Stage (configurable thresholds)
- Revenue/Employee Scale (quantile-based)

## 📈 Data Format

The system expects data in Excel (.xlsx, .xls) or CSV format with:

**Required Numeric Columns:**
- Revenue (USD)
- Market Value (USD)
- Employee Total
- Employee Single Sites
- IT Budget / IT Spend
- Technology Infrastructure (PCs, Servers, Storage, Routers, Laptops, Desktops)
- Year Founded

**Optional Text Columns:**
- SIC Description
- NAICS Description
- NACE Rev 2 Description

**Optional Categorical Columns:**
- Entity Type
- Ownership Type
- Manufacturing Status
- Import/Export Status
- Franchise Status

The system uses pattern matching to find columns, so exact names aren't required.

## 🎯 Use Cases

### Sales Intelligence
- Segment prospects by characteristics
- Prioritize high-value segments
- Understand customer profiles
- Predict segment membership for new leads

### Risk Assessment
- Identify risk patterns by segment
- Detect anomalies and outliers
- Benchmark against similar companies
- Assess technology dependency

### Market Research
- Understand industry structure
- Identify market segments
- Analyze competitive landscape
- Growth potential analysis

### Strategic Planning
- Identify growth opportunities
- Understand operational differences
- Support M&A analysis
- Technology investment planning

## ⚙️ Configuration

### Clustering Parameters
```python
# Customize cluster selection
optimal_k = analyzer.determine_optimal_clusters(
    max_k=10,
    practical_threshold=0.75  # Adjust threshold for business practicality
)
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

# Indicators are automatically included in clustering if enabled
analyzer.preprocess_data(include_key_indicators_in_clustering=True)
```

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - Install missing packages: `pip install -r requirements.txt`
   - For advanced features: `pip install statsmodels`

2. **Memory Issues**
   - Reduce number of features
   - Use sampling for initial exploration
   - Increase system memory

3. **LLM Insights Not Working**
   - Check API key is set correctly
   - Verify internet connection
   - System falls back to rule-based insights automatically

4. **Poor Clustering Results**
   - Review data quality
   - Try different number of clusters
   - Check for data normalization issues
   - Verify business indicators are meaningful

5. **Statistical Test Warnings**
   - Chi-square assumption violations: Consider combining categories
   - High VIF scores: Use Ridge/Lasso regression
   - Multiple testing: Results are automatically corrected

## 📚 Code Structure

```
company_intelligence.py
├── CompanyIntelligence (Main Class)
    ├── Data Loading & Exploration
    ├── Preprocessing (with business indicators)
    ├── Clustering (multi-metric validation)
    ├── Statistical Analysis (Chi-square, VIF)
    ├── Machine Learning (Logistic/Linear Regression)
    ├── Dimensionality Reduction (PCA, t-SNE, UMAP)
    ├── Visualization
    └── Report Generation

clustering_analysis.py
└── LatentSparseClustering (4-Phase Workflow)

process_champions_data.py
└── Data Processing & Metric Calculation
```

## 🔄 Workflow

1. **Data Loading** → Load Excel/CSV data
2. **Data Exploration** → Understand data structure
3. **Preprocessing** → Feature engineering, outlier handling, scaling
4. **Business Indicators** → Calculate derived metrics
5. **Clustering** → Multi-metric validation, optimal K selection
6. **Analysis** → Cluster characterization, comparison
7. **Statistical Tests** → Chi-square with corrections
8. **Predictive Modeling** → Logistic/Linear regression
9. **Visualization** → Charts and interactive plots
10. **Reporting** → Comprehensive text report

## 📝 License

This project is provided as-is for the Datathon 2026 competition.

## 🤝 Contributing

For questions or issues, please refer to the project documentation or contact the development team.

---

**Last Updated**: January 2026  
**Version**: 2.0 (with statistical improvements and business indicators)
