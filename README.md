# Company Intelligence Analysis System

An AI-driven company intelligence system that analyzes company data to generate actionable insights through data-driven segmentation, clustering, and predictive modeling.

## Overview

This system provides comprehensive analysis of company datasets, including:
- **Data Cleaning**: Automatic filtering of inactive companies
- **Data Transformation**: Log10 transformation of all numeric data for better normalization
- **Clustering**: K-Means clustering with balanced distribution options
- **Pattern Recognition**: Outlier detection, correlation analysis, and anomaly identification
- **Predictive Modeling**: Logistic and linear regression models
- **Dimensionality Reduction**: PCA, t-SNE, and UMAP visualization
- **LLM-Powered Insights**: Optional AI-generated insights using OpenAI or DeepSeek APIs
- **Comprehensive Reporting**: Detailed analysis reports with visualizations

## Features

### Core Functionality

1. **Data Loading & Cleaning**
   - Supports Excel (.xlsx, .xls) and CSV file formats
   - Automatic filtering of inactive companies
   - Missing data handling with intelligent imputation
   - Outlier detection and mitigation

2. **Data Preprocessing**
   - Log10 transformation of all numeric columns (handles zeros and negatives)
   - StandardScaler and RobustScaler for feature normalization
   - Automatic column detection using pattern matching
   - Focus on key metrics: Revenue, Employee Total, IT Budget, IT Spending, Market Value

3. **Clustering Analysis**
   - Optimal cluster determination using elbow method and silhouette score
   - Balanced K-Means clustering for even distribution
   - Stratified balanced clustering for optimal results
   - DBSCAN clustering option for density-based segmentation

4. **Statistical Analysis**
   - Chi-square tests for categorical variables
   - Correlation analysis between features
   - Outlier detection using IQR method
   - Cluster comparison and profiling

5. **Machine Learning Models**
   - Logistic Regression for cluster classification
   - Linear Regression for revenue prediction
   - Train/test split with performance metrics
   - Feature importance analysis

6. **Dimensionality Reduction**
   - Principal Component Analysis (PCA)
   - t-SNE visualization (optional)
   - UMAP visualization (optional)
   - 2D/3D scatter plots

7. **Business Metrics Calculation**
   - Revenue to Employee Ratio
   - IT Intensity (IT Spending / Revenue)
   - IT Budget Utilization (IT Budget / IT Spending)
   - PS Ratio (Price-to-Sales: Market Value / Revenue)
   - Market Cap per Employee
   - Market Value to IT Spending Ratio
   - Underlying Value (Revenue/Employee × IT Intensity × Employees)

8. **Visualization**
   - Cluster distribution charts
   - Feature comparison plots
   - Dimensionality reduction visualizations
   - Regression analysis plots
   - Revenue to Employee Ratio distribution analysis
   - Business metrics distribution and cluster comparison

8. **AI-Powered Insights**
   - LLM-generated insights using OpenAI or DeepSeek APIs
   - Rule-based insights as fallback
   - Comprehensive analysis reports

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Required Packages

```bash
pip install pandas numpy scikit-learn matplotlib seaborn plotly scipy openpyxl python-dotenv
```

### Optional Packages (for enhanced features)

```bash
# For LLM insights
pip install openai

# For t-SNE visualization
pip install scikit-learn[tsne]

# For UMAP visualization
pip install umap-learn
```

## How to Generate the Report

### Method 1: Quick Report Generation (Easiest - Recommended)

**Using Python Script:**

Create a file called `generate_report.py`:

```python
from company_intelligence import CompanyIntelligence
import os
from dotenv import load_dotenv

# Load environment variables (for API key - optional)
load_dotenv()

# Initialize the analyzer
data_path = 'champions_group_data.xlsx'
api_key = os.getenv('OPENAI_API_KEY')  # Optional: for LLM insights

analyzer = CompanyIntelligence(data_path, api_key=api_key)

# Run complete analysis pipeline (generates report automatically)
print("Starting analysis...")
results = analyzer.run_full_analysis()

print("\n" + "="*60)
print("Report Generated Successfully!")
print("="*60)
print("Report saved to: company_intelligence_report.txt")
print("CSV with clusters saved to: companies_with_segments.csv")
print("Visualizations saved as PNG files")
```

**Run it:**
```bash
python generate_report.py
```

### Method 2: Using Jupyter Notebook

1. Open `company_intelligence.ipynb` in Jupyter
2. Run all cells sequentially, OR
3. Run the "Quick Full Analysis" cell at the bottom:

```python
# Run complete analysis pipeline
results = analyzer.run_full_analysis(n_clusters=None)  # None = auto-determine

# Access results
print("\nAnalysis complete!")
print(f"Clusters identified: {len(set(analyzer.clusters))}")
print("\nCheck generated files for detailed results.")
```

### Method 3: Step-by-Step (For Customization)

For more control over each step:

```python
from company_intelligence import CompanyIntelligence
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize
analyzer = CompanyIntelligence('champions_group_data.xlsx', api_key=os.getenv('OPENAI_API_KEY'))

# Step 1: Preprocess data (includes log10 transformation and inactive company filtering)
analyzer.preprocess_data()

# Step 2: Determine optimal clusters
optimal_k = analyzer.determine_optimal_clusters(max_k=10)

# Step 3: Perform clustering
analyzer.perform_clustering(n_clusters=optimal_k)

# Step 4: Analyze clusters
cluster_analysis = analyzer.analyze_clusters()

# Step 5: Identify patterns
patterns = analyzer.identify_patterns()

# Step 6: Generate insights (LLM or rule-based)
insights = analyzer.generate_llm_insights(cluster_analysis, patterns)

# Step 7: Generate visualizations
analyzer.visualize_results()

# Step 8: Generate and save the report
report = analyzer.generate_report(cluster_analysis, patterns, insights)
# Report is automatically saved to: company_intelligence_report.txt

# Step 9: Export data with clusters
analyzer.df.to_csv('companies_with_segments.csv', index=False)
```

## Quick Start

### Basic Usage

```python
from company_intelligence import CompanyIntelligence
import os
from dotenv import load_dotenv

# Load environment variables (for API key)
load_dotenv()

# Initialize the analyzer
data_path = 'champions_group_data.xlsx'
api_key = os.getenv('OPENAI_API_KEY')  # Optional

analyzer = CompanyIntelligence(data_path, api_key=api_key)

# Run complete analysis pipeline
results = analyzer.run_full_analysis()
```

### Step-by-Step Usage

```python
from company_intelligence import CompanyIntelligence

# 1. Initialize
analyzer = CompanyIntelligence('champions_group_data.xlsx')

# 2. Explore data
analyzer.explore_data()

# 3. Preprocess data (includes log10 transformation and inactive company filtering)
analyzer.preprocess_data()

# 4. Determine optimal clusters
optimal_k = analyzer.determine_optimal_clusters(max_k=10)

# 5. Perform clustering
analyzer.perform_clustering(n_clusters=optimal_k)

# 6. Analyze clusters
cluster_analysis = analyzer.analyze_clusters()

# 7. Compare clusters
comparison = analyzer.compare_clusters()

# 8. Identify patterns
patterns = analyzer.identify_patterns()

# 9. Generate insights
insights = analyzer.generate_llm_insights(cluster_analysis, patterns)

# 10. Visualize results
analyzer.visualize_results()

# 11. Generate report
report = analyzer.generate_report(cluster_analysis, patterns, insights)

# 12. Export results
analyzer.df.to_csv('companies_with_segments.csv', index=False)
```

## Data Requirements

### Required Columns

The system automatically detects columns using pattern matching. Expected columns include:

**Numeric Columns:**
- Revenue (USD) - `Revenue`, `Revenue (USD)`, `revenue_usd`, etc.
- Employee Total - `Employees Total`, `employee_total`, `total_employees`, etc.
- Employee Single Sites - `Employees Single Site`, `employee_single_sites`, etc.
- Market Value - `Market Value (USD)`, `market_value`, `market_cap`, etc.
- IT Budget - `IT Budget`, `it_budget`, `technology_budget`, etc.
- IT Spending - `IT spend`, `it_spending`, `technology_spending`, etc.

**Categorical Columns:**
- Company Status - `Company Status (Active/Inactive)`, `status`, etc.
- Entity Type - `Entity Type`, `entity_type`, etc.
- Ownership Type - `Ownership Type`, `ownership_type`, etc.

**Text Columns (for TF-IDF):**
- SIC Description - `SIC Description`, `sic_description`, etc.
- NAICS Description - `NAICS Description`, `naics_description`, etc.
- NACE Description - `NACE Rev 2 Description`, `nace_description`, etc.

### Data Format

- **File Format**: Excel (.xlsx, .xls) or CSV
- **Encoding**: UTF-8 (for CSV files)
- **Missing Values**: Handled automatically (median imputation for numeric, mode for categorical)

## Key Methods

### Data Management

- `load_data()`: Load data from Excel or CSV file
- `filter_inactive_companies()`: Remove inactive companies from dataset
- `explore_data()`: Perform initial data exploration
- `preprocess_data(exclude_cols=None)`: Preprocess data with log10 transformation

### Clustering

- `determine_optimal_clusters(max_k=10)`: Find optimal number of clusters
- `perform_clustering(n_clusters=None, method='kmeans', balanced=True)`: Perform clustering
- `analyze_clusters()`: Analyze characteristics of each cluster
- `compare_clusters(feature=None)`: Compare clusters across features

### Analysis

- `identify_patterns()`: Identify outliers, correlations, and anomalies
- `apply_tfidf(text_columns, max_features=100)`: Apply TF-IDF to text columns
- `perform_chi_square_test(categorical_cols)`: Chi-square tests for categorical variables
- `apply_dimensionality_reduction(method='pca', n_components=2)`: Dimensionality reduction

### Machine Learning

- `train_logistic_regression(test_size=0.2)`: Train logistic regression model
- `train_linear_regression(target_feature, test_size=0.2)`: Train linear regression model

### Visualization

- `visualize_results()`: Generate all visualizations
- `visualize_dimensionality_reduction(methods=['pca', 'tsne'])`: Visualize reduced dimensions

### Reporting

- `generate_llm_insights(cluster_analysis, patterns)`: Generate AI-powered insights
- `generate_rule_based_insights(cluster_analysis, patterns)`: Generate rule-based insights
- `generate_report(cluster_analysis, patterns, insights)`: Generate comprehensive report
- `run_full_analysis(n_clusters=None, exclude_cols=None)`: Run complete analysis pipeline

## Data Processing Details

### Log10 Transformation

All numeric data is automatically transformed using log base 10:
- Formula: `log10(1 + abs(x))`
- Handles zeros and negative values
- Improves normalization and reduces impact of outliers
- Original values preserved with `_original` suffix

### Business Metrics

The system automatically calculates the following business metrics using original values (before log transformation):

1. **Revenue_to_Employee_Ratio**: Revenue per employee
   - Formula: `Revenue / (Employees + 1)`

2. **IT_Intensity**: IT spending efficiency
   - Formula: `IT Spending / Revenue`
   - Higher values indicate higher IT investment relative to revenue

3. **IT_Budget_Utilization**: IT budget usage efficiency
   - Formula: `IT Budget / IT Spending`
   - Values > 1 indicate overspending, < 1 indicates underspending

4. **PS_Ratio**: Price-to-Sales ratio
   - Formula: `Market Value / Revenue`
   - Valuation metric comparing market value to revenue

5. **Market_Cap_per_Employee**: Market capitalization per employee
   - Formula: `Market Value / (Employees + 1)`
   - Indicates market value efficiency per employee

6. **Market_Value_to_IT_Spending**: Market value relative to IT investment
   - Formula: `Market Value / (IT Spending + 1)`
   - Shows market valuation per IT dollar spent

7. **Underlying_Value**: Composite efficiency metric
   - Formula: `(Revenue/Employee) × IT Intensity × Employees`
   - Combines revenue efficiency, IT intensity, and scale

All metrics are automatically calculated during preprocessing and included in the exported CSV file.

### Inactive Company Filtering

Companies are automatically filtered based on status:
- **Kept**: "Active", "active", "ACTIVE", "Act", "A", "1", "True", "Yes", "Y" (case-insensitive)
- **Removed**: "Inactive", "inactive", "INACTIVE", "Inact", "I", "0", "False", "No", "N" (case-insensitive)
- **Kept**: Missing/NaN status values (assumed active)

### Scaling Methods

- **StandardScaler**: Used for columns without significant outliers
- **RobustScaler**: Used for columns with >5% outliers (outlier-resistant)

## Output Files

The system generates several output files:

1. **companies_with_segments.csv**: Dataset with cluster labels (includes Revenue_to_Employee_Ratio metric)
2. **company_intelligence_report.txt**: Comprehensive analysis report
3. **Visualization files** (PNG):
   - `cluster_distribution.png` - Distribution of companies across clusters
   - `feature_comparison.png` - Feature comparison across clusters
   - `revenue_to_employee_ratio.png` - Revenue to Employee ratio distribution analysis
   - `business_metrics_distribution.png` - Distribution of all business metrics
   - `business_metrics_by_cluster.png` - Business metrics comparison across clusters
   - `linear_regression_results.png` - Linear regression analysis
   - `pca_clusters.png` - PCA visualization
   - Dimensionality reduction plots

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# For DeepSeek (alternative)
DEEPSEEK_API_KEY=your_deepseek_api_key_here
```

### API Providers

The system supports multiple LLM providers:
- **OpenAI**: Default provider, uses `OPENAI_API_KEY`
- **DeepSeek**: Alternative provider, uses `DEEPSEEK_API_KEY`
- **Auto**: Automatically detects and uses available API key

## Jupyter Notebook

A Jupyter notebook (`company_intelligence.ipynb`) is provided for interactive analysis:

1. Open the notebook
2. Run cells sequentially
3. Modify parameters as needed
4. View results inline

## Example Use Cases

### 1. Market Segmentation

```python
analyzer = CompanyIntelligence('company_data.xlsx')
analyzer.preprocess_data()
analyzer.perform_clustering(n_clusters=5)
cluster_analysis = analyzer.analyze_clusters()
```

### 2. Revenue Prediction

```python
analyzer.train_linear_regression(target_feature='Revenue (USD)')
```

### 3. Industry Classification

```python
analyzer.apply_tfidf(text_columns=['SIC Description', 'NAICS Description'])
analyzer.train_logistic_regression()
```

### 4. Outlier Detection

```python
patterns = analyzer.identify_patterns()
outliers = patterns['outliers']
```

## Troubleshooting

### Common Issues

1. **File Not Found Error**
   - Ensure the data file path is correct
   - Use absolute path if relative path doesn't work

2. **Missing Columns**
   - Check column names match expected patterns
   - System will warn about missing columns

3. **Memory Issues**
   - Reduce `max_features` in TF-IDF
   - Use smaller `max_k` for cluster determination
   - Process data in batches

4. **API Key Issues**
   - Verify API key is set in environment variables
   - Check API key format (OpenAI keys start with 'sk-')
   - System will use rule-based insights if LLM unavailable

## Performance Considerations

- **Large Datasets**: Use `exclude_cols` to reduce feature space
- **Clustering**: Balanced clustering may take longer but provides better distribution
- **TF-IDF**: Reduce `max_features` for faster processing
- **Dimensionality Reduction**: t-SNE can be slow for large datasets

## License

This project is part of a Datathon competition submission.

## Contributing

This is a competition project. For questions or issues, please refer to the competition guidelines.

## Acknowledgments

- Built with scikit-learn, pandas, and numpy
- Visualization powered by matplotlib, seaborn, and plotly
- Optional LLM insights via OpenAI or DeepSeek APIs
