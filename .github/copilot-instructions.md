# AI Coding Agent Instructions for Company Intelligence System

## Project Overview
This is an AI-driven company intelligence system that analyzes company data through clustering, machine learning, and generates actionable business insights. The core architecture centers on the `CompanyIntelligence` class in `company_intelligence.py`.

## Architecture & Data Flow
- **Main Class**: `CompanyIntelligence` handles the entire analysis pipeline
- **Data Flow**: Load → Explore → Preprocess → Cluster → Analyze → Visualize → Report
- **Key Components**:
  - Clustering: K-Means (default) or DBSCAN
  - ML Models: Logistic Regression for cluster prediction, Linear Regression for feature relationships
  - Dimensionality Reduction: PCA (default), optional t-SNE/UMAP
  - LLM Integration: OpenAI/DeepSeek for insights, falls back to rule-based analysis

## Essential Workflows

### Setup & Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Validate setup
python test_setup.py
```

### Basic Analysis Run
```bash
# Use default data file
python company_intelligence.py

# Use custom data file
python company_intelligence.py path/to/data.xlsx
```

### Programmatic Usage Pattern
```python
from company_intelligence import CompanyIntelligence

# Initialize
analyzer = CompanyIntelligence('data.xlsx', api_key='your_key')

# Full pipeline
results = analyzer.run_full_analysis(n_clusters=5)

# Step-by-step (for customization)
analyzer.explore_data()
analyzer.preprocess_data(exclude_cols=['Company_ID'])
optimal_k = analyzer.determine_optimal_clusters()
analyzer.perform_clustering(n_clusters=optimal_k)
cluster_analysis = analyzer.analyze_clusters()
patterns = analyzer.identify_patterns()
insights = analyzer.generate_llm_insights(cluster_analysis, patterns)
analyzer.visualize_results()
report = analyzer.generate_report(cluster_analysis, patterns, insights)
```

## Data Format Requirements
- **Supported Formats**: Excel (.xlsx/.xls) or CSV
- **Column Types**: Mix of numeric (revenue, employees) and categorical (industry, country)
- **Expected Columns**: Revenue, employee counts, industry classifications, geographic data
- **Column Exclusion**: Use `exclude_cols` parameter for IDs, notes, or non-analytical fields

## API Integration Patterns
- **Auto-Detection**: Set `api_provider='auto'` to try DeepSeek first, then OpenAI
- **Environment Variables**: `DEEPSEEK_API_KEY` or `OPENAI_API_KEY`
- **Fallback**: System gracefully uses rule-based insights if LLM unavailable
- **DeepSeek Setup**: Uses OpenAI-compatible API with custom base URL

## Output Files & Artifacts
- `company_intelligence_report.txt`: Comprehensive text analysis
- `companies_with_segments.csv`: Original data with cluster labels
- `cluster_distribution.png`: Cluster size visualization
- `pca_clusters.png`: 2D PCA scatter plot
- `feature_comparison.png`: Box plots across clusters
- `interactive_clusters.html`: 3D Plotly visualization
- `optimal_clusters.png`: Elbow curve for cluster selection

## Key Implementation Patterns

### Error Handling
- Comprehensive try/catch blocks with user-friendly error messages
- File existence checks before processing
- Graceful degradation for optional features (LLM, t-SNE, UMAP)

### Method Chaining
- Most methods return `self` for fluent interface
- Example: `analyzer.preprocess_data().perform_clustering().analyze_clusters()`

### Column Pattern Matching
- Uses `_find_column_by_pattern()` for flexible column detection
- Case-insensitive pattern matching for common column names

### Visualization Strategy
- Static plots (matplotlib/seaborn) for reports
- Interactive Plotly charts for exploration
- Multiple dimensionality reduction methods for comparison

## Testing & Validation
```bash
# Quick setup validation
python test_setup.py

# Comprehensive feature testing
python validate_code.py
```

## Jupyter Notebook Usage
- Interactive exploration in `company_intelligence.ipynb`
- Cell-by-cell execution for debugging and customization
- Load environment variables with `python-dotenv`

## Common Customization Points
- Cluster count: Auto-determine or specify manually
- Feature exclusion: Remove irrelevant columns from analysis
- Dimensionality reduction: Choose PCA, t-SNE, or UMAP
- Visualization methods: Mix static and interactive outputs

## Performance Considerations
- Handles datasets with thousands of companies
- Automatic missing value handling
- Efficient preprocessing with StandardScaler
- Memory-conscious visualization generation