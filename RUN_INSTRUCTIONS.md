# How to Run the Company Intelligence System

## Quick Start (3 Steps)

### Step 1: Install Required Packages

Open your terminal/command prompt in the Datathon folder and run:

```bash
# If you don't have packages installed, install them:
pip install -r requirements.txt
```

Or if you prefer using a virtual environment (recommended):

```bash
# Create virtual environment
python3 -m venv venv

# Activate it (Mac/Linux):
source venv/bin/activate

# Activate it (Windows):
venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### Step 2: Verify Setup (Optional)

Test that everything is ready:

```bash
python test_setup.py
```

Or validate all new features:

```bash
python validate_code.py
```

### Step 3: Run the Analysis

**Option A: Simple Run (Recommended)**

```bash
python company_intelligence.py
```

This will automatically use `champions_group_data.xlsx` in the current directory.

**Option B: Specify a Different Data File**

```bash
python company_intelligence.py path/to/your/file.xlsx
```

Or for CSV:

```bash
python company_intelligence.py path/to/your/file.csv
```

## What Happens When You Run It

The system will:

1. ✅ Load your data from `champions_group_data.xlsx`
2. ✅ Explore and preprocess the data
3. ✅ Determine optimal number of clusters
4. ✅ Perform clustering/segmentation
5. ✅ Apply TF-IDF (if text columns exist)
6. ✅ Run Chi-square tests on categorical variables
7. ✅ Apply dimensionality reduction (PCA, SVD)
8. ✅ Train logistic regression model
9. ✅ Generate visualizations
10. ✅ Create comprehensive report

## Output Files Generated

After running, you'll find these files in your directory:

- **company_intelligence_report.txt** - Complete analysis report
- **companies_with_segments.csv** - Original data with cluster labels added
- **cluster_distribution.png** - Bar chart of cluster sizes
- **pca_clusters.png** - 2D PCA visualization
- **feature_comparison.png** - Boxplots comparing features across clusters
- **interactive_clusters.html** - Interactive 3D visualization (open in browser)
- **optimal_clusters.png** - Cluster selection analysis
- **dimensionality_reduction.png** - Comparison of reduction methods

## Alternative Ways to Run

### Using Python Interactively

```python
from company_intelligence import CompanyIntelligence

# Initialize
analyzer = CompanyIntelligence('champions_group_data.xlsx')

# Run full analysis
results = analyzer.run_full_analysis()

# Access results
print(f"Found {len(results['cluster_analysis'])} clusters")
print(f"Logistic Regression Accuracy: {results['logistic_regression']['test_accuracy']:.2%}")
```

### Using Jupyter Notebook

```bash
jupyter notebook company_intelligence.ipynb
```

Then run the cells step-by-step for interactive analysis.

## Troubleshooting

### Error: "ModuleNotFoundError: No module named 'pandas'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Error: "FileNotFoundError: champions_group_data.xlsx"

**Solution:** 
- Make sure the Excel file is in the same directory as `company_intelligence.py`
- Or specify the full path: `python company_intelligence.py /full/path/to/file.xlsx`

### Error: "Externally-managed-environment"

**Solution:** Use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### Error: Can't read Excel file

**Solution:**
```bash
pip install openpyxl
```

## Need Help?

- Check `README.md` for detailed documentation
- See `example_usage.py` for code examples
- Run `python validate_code.py` to test all features

---

**Ready to run? Just execute:**
```bash
python company_intelligence.py
```

Happy analyzing! 🚀
