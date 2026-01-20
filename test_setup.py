"""
Quick test script to validate setup and data loading
"""

import sys
import os

def test_imports():
    """Test if all required packages are installed"""
    print("Testing package imports...")
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
        'openpyxl': 'openpyxl'
    }
    
    missing = []
    for module_name, package_name in required_packages.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            missing.append(package_name)
    
    # Optional packages
    try:
        import openai
        print(f"  ✓ openai (optional - for LLM insights)")
    except ImportError:
        print(f"  ⚠ openai (optional - not installed, will use rule-based insights)")
    
    if missing:
        print(f"\n⚠ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All required packages installed!")
        return True


def test_data_file():
    """Test if data file exists and can be read"""
    print("\nTesting data file...")
    
    data_files = ['champions_group_data.xlsx', 'champions_group_data.xls']
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"  ✓ Found: {data_file}")
            try:
                import pandas as pd
                df = pd.read_excel(data_file, nrows=5)  # Just read first 5 rows
                print(f"  ✓ Can read file: {df.shape[0]} rows, {df.shape[1]} columns")
                print(f"  ✓ Columns: {list(df.columns)[:10]}..." if len(df.columns) > 10 else f"  ✓ Columns: {list(df.columns)}")
                return True, data_file
            except Exception as e:
                print(f"  ✗ Error reading file: {e}")
                return False, None
    
    print("  ✗ Data file not found!")
    print("  Expected: champions_group_data.xlsx or champions_group_data.xls")
    return False, None


def main():
    print("="*60)
    print("COMPANY INTELLIGENCE SYSTEM - SETUP VALIDATION")
    print("="*60)
    
    # Test imports
    imports_ok = test_imports()
    
    # Test data file
    data_ok, data_file = test_data_file()
    
    print("\n" + "="*60)
    if imports_ok and data_ok:
        print("✓ SETUP VALIDATION PASSED")
        print("="*60)
        print("\nYou can now run the analysis:")
        print(f"  python company_intelligence.py {data_file if data_file else ''}")
        print("\nOr use the Jupyter notebook:")
        print("  jupyter notebook company_intelligence.ipynb")
        return 0
    else:
        print("✗ SETUP VALIDATION FAILED")
        print("="*60)
        if not imports_ok:
            print("\nPlease install missing packages:")
            print("  pip install -r requirements.txt")
        if not data_ok:
            print("\nPlease ensure the data file is in the current directory")
        return 1


if __name__ == "__main__":
    sys.exit(main())
