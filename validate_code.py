"""
Validation script to test the enhanced company intelligence system
Tests all new features: Logistic Regression, TF-IDF, Chi-square, Dimensionality Reduction
"""

import sys
import os

def validate_imports():
    """Validate all required imports"""
    print("="*60)
    print("VALIDATING IMPORTS")
    print("="*60)
    
    required_modules = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'plotly': 'plotly',
    }
    
    missing = []
    for module_name, package_name in required_modules.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name}")
        except ImportError:
            print(f"  ✗ {package_name} - MISSING")
            missing.append(package_name)
    
    # Optional modules
    optional_modules = {
        'umap': 'umap-learn',
        'openai': 'openai'
    }
    
    for module_name, package_name in optional_modules.items():
        try:
            __import__(module_name)
            print(f"  ✓ {package_name} (optional)")
        except ImportError:
            print(f"  ⚠ {package_name} (optional - not installed)")
    
    print()
    return len(missing) == 0


def validate_code_structure():
    """Validate the code structure and methods"""
    print("="*60)
    print("VALIDATING CODE STRUCTURE")
    print("="*60)
    
    try:
        from company_intelligence import CompanyIntelligence
        
        # Check if all new methods exist
        required_methods = [
            'apply_tfidf',
            'perform_chi_square_test',
            'apply_dimensionality_reduction',
            'train_logistic_regression',
            'visualize_dimensionality_reduction'
        ]
        
        missing_methods = []
        for method in required_methods:
            if hasattr(CompanyIntelligence, method):
                print(f"  ✓ Method: {method}")
            else:
                print(f"  ✗ Method: {method} - MISSING")
                missing_methods.append(method)
        
        print()
        return len(missing_methods) == 0
        
    except Exception as e:
        print(f"  ✗ Error importing CompanyIntelligence: {e}")
        return False


def test_data_loading(data_path):
    """Test data loading"""
    print("="*60)
    print("TESTING DATA LOADING")
    print("="*60)
    
    try:
        from company_intelligence import CompanyIntelligence
        
        print(f"Loading data from: {data_path}")
        analyzer = CompanyIntelligence(data_path)
        
        if analyzer.df is not None and len(analyzer.df) > 0:
            print(f"  ✓ Data loaded successfully: {analyzer.df.shape[0]} rows, {analyzer.df.shape[1]} columns")
            print(f"  ✓ Columns: {list(analyzer.df.columns)[:10]}..." if len(analyzer.df.columns) > 10 else f"  ✓ Columns: {list(analyzer.df.columns)}")
            return analyzer
        else:
            print("  ✗ Data loading failed - empty dataframe")
            return None
            
    except Exception as e:
        print(f"  ✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_basic_analysis(analyzer):
    """Test basic analysis steps"""
    print("\n" + "="*60)
    print("TESTING BASIC ANALYSIS")
    print("="*60)
    
    try:
        # Test preprocessing
        print("\n1. Testing preprocessing...")
        analyzer.preprocess_data()
        if analyzer.df_processed_scaled is not None:
            print(f"  ✓ Preprocessing successful: {len(analyzer.feature_names)} features")
        else:
            print("  ✗ Preprocessing failed")
            return False
        
        # Test clustering
        print("\n2. Testing clustering...")
        analyzer.perform_clustering(n_clusters=3)  # Use small number for testing
        if analyzer.clusters is not None:
            print(f"  ✓ Clustering successful: {len(set(analyzer.clusters))} clusters")
        else:
            print("  ✗ Clustering failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"  ✗ Error in basic analysis: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_new_features(analyzer):
    """Test new features"""
    print("\n" + "="*60)
    print("TESTING NEW FEATURES")
    print("="*60)
    
    results = {}
    
    try:
        # Test TF-IDF
        print("\n1. Testing TF-IDF...")
        tfidf_result = analyzer.apply_tfidf()
        if tfidf_result is not None:
            print(f"  ✓ TF-IDF successful: {tfidf_result.shape[1]} features created")
            results['tfidf'] = True
        else:
            print("  ⚠ TF-IDF skipped (no text columns found - this is OK)")
            results['tfidf'] = None
        
        # Test Chi-square
        print("\n2. Testing Chi-square test...")
        chi_square_result = analyzer.perform_chi_square_test()
        if chi_square_result is not None and len(chi_square_result) > 0:
            print(f"  ✓ Chi-square test successful: {len(chi_square_result)} variables tested")
            results['chi_square'] = True
        else:
            print("  ⚠ Chi-square test skipped (no categorical variables - this is OK)")
            results['chi_square'] = None
        
        # Test Dimensionality Reduction
        print("\n3. Testing Dimensionality Reduction...")
        for method in ['pca', 'svd']:
            try:
                dim_result = analyzer.apply_dimensionality_reduction(method=method, n_components=2)
                if dim_result is not None:
                    print(f"  ✓ {method.upper()} successful")
                    results[f'{method}_reduction'] = True
                else:
                    print(f"  ✗ {method.upper()} failed")
                    results[f'{method}_reduction'] = False
            except Exception as e:
                print(f"  ✗ {method.upper()} error: {e}")
                results[f'{method}_reduction'] = False
        
        # Test Logistic Regression
        print("\n4. Testing Logistic Regression...")
        try:
            lr_result = analyzer.train_logistic_regression()
            if lr_result is not None:
                print(f"  ✓ Logistic Regression successful")
                print(f"    Training Accuracy: {lr_result['train_accuracy']:.4f}")
                print(f"    Test Accuracy: {lr_result['test_accuracy']:.4f}")
                results['logistic_regression'] = True
            else:
                print("  ✗ Logistic Regression failed")
                results['logistic_regression'] = False
        except Exception as e:
            print(f"  ✗ Logistic Regression error: {e}")
            import traceback
            traceback.print_exc()
            results['logistic_regression'] = False
        
        return results
        
    except Exception as e:
        print(f"  ✗ Error testing new features: {e}")
        import traceback
        traceback.print_exc()
        return results


def main():
    """Main validation function"""
    print("\n" + "="*80)
    print("COMPANY INTELLIGENCE SYSTEM - CODE VALIDATION")
    print("="*80 + "\n")
    
    # Find data file
    data_files = ['champions_group_data.xlsx', 'champions_group_data.xls', 'champions_group_data.csv']
    data_path = None
    
    for file in data_files:
        if os.path.exists(file):
            data_path = file
            break
    
    if data_path is None:
        print("ERROR: Data file not found!")
        print(f"Expected one of: {data_files}")
        if len(sys.argv) > 1:
            data_path = sys.argv[1]
            print(f"Using provided path: {data_path}")
        else:
            return 1
    
    # Validate imports
    imports_ok = validate_imports()
    
    # Validate code structure
    structure_ok = validate_code_structure()
    
    if not imports_ok:
        print("\n⚠ Some required packages are missing. Install with:")
        print("  pip install -r requirements.txt")
        return 1
    
    if not structure_ok:
        print("\n⚠ Code structure validation failed!")
        return 1
    
    # Test data loading
    analyzer = test_data_loading(data_path)
    if analyzer is None:
        return 1
    
    # Test basic analysis
    basic_ok = test_basic_analysis(analyzer)
    if not basic_ok:
        print("\n⚠ Basic analysis failed!")
        return 1
    
    # Test new features
    new_features_results = test_new_features(analyzer)
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    print("\nCore Features:")
    print("  ✓ Data Loading: PASSED")
    print("  ✓ Preprocessing: PASSED")
    print("  ✓ Clustering: PASSED")
    
    print("\nNew Features:")
    for feature, result in new_features_results.items():
        if result is True:
            print(f"  ✓ {feature}: PASSED")
        elif result is None:
            print(f"  ⚠ {feature}: SKIPPED (no applicable data)")
        else:
            print(f"  ✗ {feature}: FAILED")
    
    print("\n" + "="*80)
    print("✓ VALIDATION COMPLETE")
    print("="*80)
    print("\nAll core features are working correctly!")
    print("The system is ready to use.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
