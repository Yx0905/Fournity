# Quick Start Guide

Get started with the Company Intelligence System in 3 simple steps!

## Step 1: Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

## Step 2: Validate Setup

```bash
python test_setup.py
```

This will check if all packages are installed and if your data file is accessible.

## Step 3: Run Analysis

### Option A: Command Line (Easiest)

```bash
python company_intelligence.py
```

This will:
- Load your data (`champions_group_data.xlsx`)
- Automatically determine optimal clusters
- Perform segmentation
- Generate insights and visualizations
- Create a comprehensive report

### Option B: Interactive Notebook

```bash
jupyter notebook company_intelligence.ipynb
```

Open the notebook and run cells interactively to explore the analysis step-by-step.

### Option C: Python Script

```python
from company_intelligence import CompanyIntelligence

analyzer = CompanyIntelligence('champions_group_data.xlsx')
results = analyzer.run_full_analysis()
```

## Output Files

After running, you'll get:

- **company_intelligence_report.txt** - Full analysis report
- **companies_with_segments.csv** - Your data with cluster labels
- ***.png** - Various visualizations
- **interactive_clusters.html** - Interactive 3D visualization

## Optional: Enable LLM Insights

1. Get an OpenAI API key from https://platform.openai.com/api-keys
2. Create a `.env` file:
   ```
   OPENAI_API_KEY=your_key_here
   ```
3. The system will automatically use LLM for enhanced insights!

## Troubleshooting

**Problem**: `ModuleNotFoundError`
- **Solution**: Run `pip install -r requirements.txt`

**Problem**: Can't read Excel file
- **Solution**: Ensure `openpyxl` is installed: `pip install openpyxl`

**Problem**: Data file not found
- **Solution**: Place your Excel/CSV file in the same directory, or specify the path:
  ```bash
  python company_intelligence.py path/to/your/data.xlsx
  ```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [example_usage.py](example_usage.py) for advanced usage examples
- Explore the Jupyter notebook for interactive analysis

Happy analyzing! 🚀
