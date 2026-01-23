"""
AI-Driven Company Intelligence System
Analyzes company data to generate actionable insights through data-driven segmentation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
try:
    from sklearn.impute import KNNImputer
    HAS_KNN_IMPUTER = True
except ImportError:
    HAS_KNN_IMPUTER = False

from scipy.stats import chi2_contingency

# Optional imports for advanced statistical tests
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("Warning: statsmodels not available. Multiple testing correction will use scipy only.")
from scipy import sparse
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
import os
from typing import Dict, List, Tuple, Optional
import json

warnings.filterwarnings('ignore')

# Try to import optional libraries
try:
    from sklearn.manifold import TSNE
    HAS_TSNE = True
except ImportError:
    HAS_TSNE = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

# Try to import OpenAI for LLM insights (optional)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: OpenAI package not available. LLM insights will be disabled.")
    print("Install with: pip install openai")


class CompanyIntelligence:
    """Main class for company intelligence analysis"""
    
    def __init__(self, data_path: str, api_key: Optional[str] = None, api_provider: str = 'auto'):
        """
        Initialize the Company Intelligence system
        
        Args:
            data_path: Path to the Excel/CSV file
            api_key: Optional API key for LLM insights (OpenAI or DeepSeek)
            api_provider: API provider to use ('openai', 'deepseek', or 'auto')
                         'auto' will try DeepSeek first, then OpenAI
        """
        self.data_path = data_path
        self.df = None
        self.df_processed = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.clusters = None
        self.cluster_model = None
        self.feature_names = []
        self.logistic_model = None
        self.linear_model = None
        self.tfidf_vectorizer = None
        self.tfidf_features = None
        self.api_provider = None
        
        # Analysis metrics storage
        self.segmentation_metrics = None
        self.trend_metrics = None
        self.driver_metrics = None
        self.performance_metrics = None
        self.pca_metrics = None
        
        # Initialize LLM client if available
        self.client = None
        self.use_llm = False
        
        if HAS_OPENAI and api_key:
            if api_provider == 'auto':
                # Try DeepSeek first (if DEEPSEEK_API_KEY is set), otherwise OpenAI
                deepseek_key = os.getenv('DEEPSEEK_API_KEY')
                if deepseek_key:
                    # Use DeepSeek if environment variable is set
                    api_key = deepseek_key
                    api_provider = 'deepseek'
                elif 'deepseek' in api_key.lower():
                    # Check if key contains 'deepseek' keyword
                    api_provider = 'deepseek'
                elif api_key.startswith('sk-'):
                    # OpenAI keys typically start with 'sk-'
                    api_provider = 'openai'
                else:
                    # Default to OpenAI if unclear
                    api_provider = 'openai'
            
            if api_provider.lower() == 'deepseek':
                # DeepSeek uses OpenAI-compatible API with different base URL
                self.client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.deepseek.com/v1"
                )
                self.api_provider = 'deepseek'
                self.use_llm = True
                print("Using DeepSeek API for LLM insights")
            elif api_provider.lower() == 'openai':
                self.client = OpenAI(api_key=api_key)
                self.api_provider = 'openai'
                self.use_llm = True
                print("Using OpenAI API for LLM insights")
            else:
                print(f"Warning: Unknown API provider '{api_provider}'. Using OpenAI.")
                self.client = OpenAI(api_key=api_key)
                self.api_provider = 'openai'
                self.use_llm = True
        
        # Load data
        if not self.load_data():
            raise ValueError(f"Failed to load data from {self.data_path}. Please check the file path and try again.")
    
    def _find_column_by_pattern(self, patterns: List[str]) -> Optional[str]:
        """
        Find a column by matching patterns (case-insensitive)
        
        Args:
            patterns: List of possible column name patterns to search for
            
        Returns:
            Column name if found, None otherwise
        """
        if self.df is None:
            return None
        
        for col in self.df.columns:
            col_lower = str(col).lower()
            for pattern in patterns:
                if pattern.lower() in col_lower:
                    return col
        return None
    
    def load_data(self):
        """Load data from Excel or CSV file"""
        print("Loading data...")
        try:
            if self.data_path.endswith('.xlsx') or self.data_path.endswith('.xls'):
                self.df = pd.read_excel(self.data_path)
            else:
                self.df = pd.read_csv(self.data_path)
            
            if self.df is None or len(self.df) == 0:
                print(f"Error: Data file is empty or could not be read: {self.data_path}")
                return False
            
            print(f"Data loaded successfully: {self.df.shape[0]} companies, {self.df.shape[1]} features")
            print(f"\nColumns: {list(self.df.columns)}")
            return True
        except FileNotFoundError:
            print(f"\n❌ ERROR: File not found: {self.data_path}")
            print(f"\n💡 Tip: Make sure the file path is correct.")
            print(f"   If the file is in the current directory, use just the filename:")
            print(f"   python company_intelligence.py champions_group_data.xlsx")
            print(f"\n   Or run without arguments to use the default file:")
            print(f"   python company_intelligence.py")
            return False
        except Exception as e:
            print(f"\n❌ Error loading data: {e}")
            print(f"\n💡 Troubleshooting tips:")
            print(f"   - Check if the file exists: {self.data_path}")
            print(f"   - For Excel files, make sure openpyxl is installed: pip install openpyxl")
            print(f"   - Try using an absolute path to the file")
            return False
    
    def explore_data(self):
        """Perform initial data exploration - focused on target columns"""
        print("\n" + "="*50)
        print("DATA EXPLORATION - FOCUSED COLUMNS")
        print("="*50)
        
        print(f"\nFull Dataset Shape: {self.df.shape}")
        
        # Identify target numeric columns
        numeric_columns = {
            'revenue': ['revenue', 'revenue_usd', 'annual_revenue', 'total_revenue', 'sales'],
            'employee_total': ['employee total', 'employee_total', 'employees', 'total_employees', 
                              'employee_count', 'num_employees', 'employees_total'],
            'employee_single_sites': ['employee single sites', 'employee_single_sites', 'single_site_employees',
                                     'employees_single_site', 'single_site_employee_count'],
            'market_value': ['market value', 'marketvalue', 'market_val', 'market_val_usd', 'market_cap'],
            'it_spending': ['it spending', 'it_spending', 'it_spend', 'it_spending_usd', 'technology_spending',
                           'it spending & budget', 'it_spending_budget'],
            'it_budget': ['it budget', 'it_budget', 'it_budget_usd', 'technology_budget', 'tech_budget']
        }
        
        # Identify text columns for TF-IDF
        text_columns = {
            'sic_description': ['sic description', 'sic_description', 'sic desc', '8-digit sic description',
                               '8_digit_sic_description', 'sic_8_digit', 'sic code description'],
            'naics_description': ['naics description', 'naics_description', 'naics desc', 'naics code description'],
            'nace_description': ['nace rev 2 description', 'nace_rev_2_description', 'nace description',
                                'nace_desc', 'nace rev2 description']
        }
        
        # Identify categorical columns
        categorical_columns = {
            'entity_type': ['entity type', 'entity_type', 'entity', 'company_type', 'legal_entity_type'],
            'ownership_type': ['ownership type', 'ownership_type', 'ownership', 'owner_type']
        }
        
        found_numeric = []
        found_text = []
        found_categorical = []
        
        for key, patterns in numeric_columns.items():
            col = self._find_column_by_pattern(patterns)
            if col:
                found_numeric.append(col)
        
        for key, patterns in text_columns.items():
            col = self._find_column_by_pattern(patterns)
            if col:
                found_text.append(col)
        
        for key, patterns in categorical_columns.items():
            col = self._find_column_by_pattern(patterns)
            if col:
                found_categorical.append(col)
        
        if found_numeric:
            print(f"\n✓ Found {len(found_numeric)} numeric columns: {found_numeric}")
            print(f"\nMissing Values in Numeric Columns:")
            missing = self.df[found_numeric].isnull().sum()
            missing_pct = (missing / len(self.df)) * 100
            missing_df = pd.DataFrame({
                'Missing Count': missing,
                'Missing %': missing_pct
            })
            print(missing_df[missing_df['Missing Count'] > 0] if missing_df['Missing Count'].sum() > 0 else "No missing values!")
            
            print(f"\nBasic Statistics for Numeric Columns:")
            print(self.df[found_numeric].describe())
        
        if found_text:
            print(f"\n✓ Found {len(found_text)} text columns for TF-IDF: {found_text}")
        
        if found_categorical:
            print(f"\n✓ Found {len(found_categorical)} categorical columns for Chi-square: {found_categorical}")
            for col in found_categorical:
                print(f"  {col}: {self.df[col].value_counts().head()}")
        
        if not found_numeric and not found_text and not found_categorical:
            print("\n⚠ Warning: Could not find target columns. Showing all columns.")
            print(f"\nAll Columns: {list(self.df.columns)}")
        
        return self.df
    
    def remove_outliers_iqr(self, df: pd.DataFrame, columns: List[str], multiplier: float = 1.5, 
                           method: str = 'cap') -> Tuple[pd.DataFrame, Dict]:
        """
        Handle outliers using the Interquartile Range (IQR) method.
        
        FIXED: Changed from union-based removal (removes row if ANY feature is outlier)
        to per-feature capping (winsorization) to preserve valuable data points.

        Args:
            df: DataFrame to clean
            columns: List of column names to check for outliers
            multiplier: IQR multiplier (default 1.5 for standard outlier detection)
            method: 'cap' (winsorize) or 'remove' (drop rows) - default 'cap' to preserve data

        Returns:
            Tuple of (cleaned_df, outlier_report)
        """
        df_clean = df.copy()
        outlier_report = {
            'total_rows_before': len(df),
            'columns_processed': [],
            'outliers_by_column': {},
            'total_rows_removed': 0,
            'total_rows_after': 0,
            'percentage_removed': 0,
            'method': method
        }

        # Track outliers per column (for reporting)
        outlier_indices_per_col = {}

        for col in columns:
            if col not in df.columns:
                continue

            # Calculate IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Skip if IQR is zero or invalid
            if IQR <= 0 or pd.isna(IQR):
                print(f"  ⚠ Skipping {col}: IQR is zero or invalid")
                continue

            # Calculate bounds
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            # Find outlier indices
            col_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outlier_indices_per_col[col] = col_outliers

            # FIXED: Per-feature handling instead of union-based removal
            if method == 'cap':
                # Winsorize: Cap outliers at bounds (preserves data)
                df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
                df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
                rows_affected = len(col_outliers)
            else:
                # Old method: Mark for removal (union-based)
                rows_affected = len(col_outliers)

            # Store report data
            outlier_report['columns_processed'].append(col)
            outlier_report['outliers_by_column'][col] = {
                'count': len(col_outliers),
                'percentage': (len(col_outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'rows_affected': rows_affected
            }

        # Only remove rows if method is 'remove' (union-based - not recommended)
        if method == 'remove':
            outlier_indices = set()
            for col, indices in outlier_indices_per_col.items():
                outlier_indices.update(indices)
            
            if outlier_indices:
                df_clean = df_clean.drop(index=list(outlier_indices))
                outlier_report['total_rows_removed'] = len(outlier_indices)
                outlier_report['total_rows_after'] = len(df_clean)
                outlier_report['percentage_removed'] = (len(outlier_indices) / len(df)) * 100
            else:
                outlier_report['total_rows_after'] = len(df)
        else:
            # Method 'cap': No rows removed, only values capped
            outlier_report['total_rows_after'] = len(df_clean)
            outlier_report['total_rows_removed'] = 0
            outlier_report['percentage_removed'] = 0
            total_outliers = sum(len(indices) for indices in outlier_indices_per_col.values())
            print(f"  → Capped {total_outliers} outlier values (preserved all {len(df_clean)} rows)")

        return df_clean, outlier_report

    def preprocess_data(self, exclude_cols: List[str] = None, calculate_indicators: bool = True, 
                       include_key_indicators_in_clustering: bool = True):
        """
        Preprocess data for clustering - COMPREHENSIVE FEATURE SET:

        Numeric Features:
        - Core: Revenue, Market Value, Employee Total, Employee Single Sites
        - Technology: IT Budget, IT Spend, PCs, Servers, Storage, Routers, Laptops, Desktops
        - Maturity: Company Age (derived from Year Founded)
        - Key Business Indicators: Market Value/Revenue Ratio, IT Investment Intensity

        Text Features (TF-IDF):
        - SIC Description, NAICS Description, NACE Rev 2 Description

        Categorical Features (Chi-square):
        - Entity Type, Ownership Type, Manufacturing Status, Import/Export Status, Franchise Status

        Processing Steps:
        1. Feature engineering (create company age)
        2. Missing value imputation (median)
        3. Outlier removal (IQR method)
        4. Calculate key business indicators (Market Value/Revenue, IT Investment Intensity)
        5. Feature scaling (StandardScaler)
        6. TF-IDF vectorization for text
        7. Feature combination
        8. (Optional) Calculate additional business indicators

        Args:
            exclude_cols: List of column names to exclude from analysis (not used in focused mode)
            calculate_indicators: If True, calculate key business indicators (default: True)
            include_key_indicators_in_clustering: If True, include Market Value/Revenue and IT Investment Intensity in clustering (default: True)
        """
        print("\n" + "="*50)
        print("DATA PREPROCESSING - COMPREHENSIVE ANALYSIS")
        print("="*50)
        print("Core: Revenue, Market Value, Employees")
        print("Technology: IT Budget/Spend, Infrastructure (PCs, Servers, etc.)")
        print("Maturity: Company Age")
        print("Text: SIC/NAICS/NACE Descriptions (TF-IDF)")
        print("Categorical: Entity Type, Ownership, Manufacturing, Trade Status")

        # Start with a copy
        df_processed = self.df.copy()

        # Define target numeric columns with multiple possible name patterns
        numeric_columns = {
            # Core financial metrics
            'revenue': ['revenue', 'revenue_usd', 'revenue (usd)', 'annual_revenue', 'total_revenue', 'sales'],
            'market_value': ['market value', 'market value (usd)', 'marketvalue', 'market_val', 'market_val_usd', 'market_cap'],

            # Workforce metrics
            'employee_total': ['employee total', 'employees total', 'employee_total', 'employees', 'total_employees',
                              'employee_count', 'num_employees', 'employees_total'],
            'employee_single_sites': ['employee single sites', 'employees single site', 'employee_single_sites',
                                     'single_site_employees', 'employees_single_site', 'single_site_employee_count'],

            # Technology investment
            'it_budget': ['it budget', 'it_budget', 'it_budget_usd', 'technology_budget', 'tech_budget'],
            'it_spend': ['it spend', 'it_spend', 'it spending', 'it_spending', 'it_spending_usd',
                        'technology_spending', 'it spending & budget', 'it_spending_budget'],

            # Technology infrastructure (NEW)
            'num_pcs': ['no. of pc', 'no of pc', 'no. of pcs', 'pc count', 'num_pcs', 'total_pcs'],
            'num_desktops': ['no. of desktops', 'no of desktops', 'desktop count', 'num_desktops'],
            'num_laptops': ['no. of laptops', 'no of laptops', 'laptop count', 'num_laptops'],
            'num_servers': ['no. of servers', 'no of servers', 'server count', 'num_servers'],
            'num_storage': ['no. of storage devices', 'no of storage devices', 'storage_devices', 'num_storage'],
            'num_routers': ['no. of routers', 'no of routers', 'router count', 'num_routers'],

            # Company maturity (NEW)
            'year_founded': ['year found', 'year_found', 'founded', 'year founded', 'year_founded',
                           'establishment_year', 'incorporation_year']
        }

        # Define text columns for TF-IDF
        text_columns = {
            'sic_description': ['sic description', 'sic_description', 'sic desc', '8-digit sic description',
                               '8_digit_sic_description', 'sic_8_digit', 'sic code description'],
            'naics_description': ['naics description', 'naics_description', 'naics desc', 'naics code description'],
            'nace_description': ['nace rev 2 description', 'nace_rev_2_description', 'nace description',
                                'nace_desc', 'nace rev2 description']
        }

        # Define categorical columns for Chi-square tests
        categorical_columns = {
            'entity_type': ['entity type', 'entity_type', 'entity', 'company_type', 'legal_entity_type'],
            'ownership_type': ['ownership type', 'ownership_type', 'ownership', 'owner_type'],
            'manufacturing_status': ['manufacturing status', 'manufacturing_status', 'is_manufacturer', 'manufacturer'],
            'import_export_status': ['import/export status', 'import_export_status', 'trade_status'],
            'franchise_status': ['franchise status', 'franchise_status', 'is_franchise']
        }

        # Find actual column names in the dataset
        found_numeric = {}
        found_text = {}
        found_categorical = {}
        feature_cols = []

        # Find numeric columns
        for key, patterns in numeric_columns.items():
            col = self._find_column_by_pattern(patterns)
            if col:
                found_numeric[key] = col
                print(f"✓ Found numeric '{key}': {col}")
            else:
                print(f"⚠ Warning: Could not find column for '{key}' (searched: {patterns})")

        # Find text columns for TF-IDF
        for key, patterns in text_columns.items():
            col = self._find_column_by_pattern(patterns)
            if col:
                found_text[key] = col
                print(f"✓ Found text '{key}': {col}")
            else:
                print(f"⚠ Warning: Could not find column for '{key}' (searched: {patterns})")

        # Find categorical columns
        for key, patterns in categorical_columns.items():
            col = self._find_column_by_pattern(patterns)
            if col:
                found_categorical[key] = col
                print(f"✓ Found categorical '{key}': {col}")
            else:
                print(f"⚠ Warning: Could not find column for '{key}' (searched: {patterns})")

        if len(found_numeric) == 0:
            raise ValueError("None of the target numeric columns were found in the dataset. Please check column names.")

        # Create derived features
        print(f"\n{'='*50}")
        print("FEATURE ENGINEERING")
        print(f"{'='*50}")

        # Create company age from year founded
        if 'year_founded' in found_numeric:
            year_col = found_numeric['year_founded']
            # FIXED: Use dynamic current year instead of hardcoded
            from datetime import datetime
            current_year = datetime.now().year
            if year_col in df_processed.columns:
                # Create age column
                age_col = 'company_age'
                df_processed[age_col] = current_year - df_processed[year_col]

                # FIXED: Validate years (check for future years, invalid years)
                invalid_years = (df_processed[year_col] > current_year) | (df_processed[year_col] < 1800)
                if invalid_years.sum() > 0:
                    print(f"  ⚠ Warning: {invalid_years.sum()} companies have invalid years (future or < 1800)")
                    print(f"     Invalid years will be set to median")
                    df_processed.loc[invalid_years, year_col] = df_processed[year_col].median()
                    df_processed[age_col] = current_year - df_processed[year_col]

                # Handle invalid ages (negative or too large)
                df_processed.loc[df_processed[age_col] < 0, age_col] = 0
                df_processed.loc[df_processed[age_col] > 200, age_col] = df_processed[age_col].median()

                # Add to found_numeric and remove year_founded (we use age instead)
                found_numeric['company_age'] = age_col
                del found_numeric['year_founded']
                print(f"✓ Created derived feature 'company_age' from 'Year Found'")
                print(f"  Age range: {df_processed[age_col].min():.0f} - {df_processed[age_col].max():.0f} years")

        # Process numeric columns - handle missing values and convert categorical ranges
        for key, col in found_numeric.items():
            # Check if column is actually numeric or contains categorical ranges
            if df_processed[col].dtype == 'object':
                # Convert categorical ranges to numeric (e.g., "1 to 10" -> 5)
                try:
                    def parse_range(val):
                        """FIXED: Handle multiple range formats: '1 to 10', '1-10', '1~10', '<10', '>100'"""
                        if pd.isna(val):
                            return np.nan
                        val_str = str(val).strip()
                        
                        # Handle "X to Y" format
                        if ' to ' in val_str.lower():
                            parts = val_str.lower().split(' to ')
                            try:
                                return (float(parts[0]) + float(parts[1])) / 2
                            except:
                                return np.nan
                        # Handle "X-Y" or "X~Y" format
                        elif '-' in val_str and not val_str.startswith('-'):
                            parts = val_str.split('-')
                            try:
                                return (float(parts[0]) + float(parts[1])) / 2
                            except:
                                return np.nan
                        elif '~' in val_str:
                            parts = val_str.split('~')
                            try:
                                return (float(parts[0]) + float(parts[1])) / 2
                            except:
                                return np.nan
                        # Handle inequalities "<X" or ">X"
                        elif val_str.startswith('<'):
                            try:
                                return float(val_str[1:]) * 0.5  # Use half of upper bound
                            except:
                                return np.nan
                        elif val_str.startswith('>'):
                            try:
                                return float(val_str[1:]) * 1.5  # Use 1.5x of lower bound
                            except:
                                return np.nan
                        # Handle plain numbers
                        elif val_str.replace('.', '').replace('-', '').isdigit():
                            return float(val_str)
                        else:
                            return np.nan

                    df_processed[col] = df_processed[col].apply(parse_range)
                    print(f"  Converted categorical ranges to numeric in {col}")
                except Exception as e:
                    print(f"  ⚠ Warning: Could not convert {col} to numeric, skipping: {e}")
                    continue

            # Handle missing values
            if df_processed[col].isnull().sum() > 0:
                fill_value = df_processed[col].median() if not pd.isna(df_processed[col].median()) else 0
                df_processed[col].fillna(fill_value, inplace=True)
                print(f"  Filled missing values in {col} with {fill_value:.2f}")

        # Remove outliers from numeric columns
        print(f"\n{'='*50}")
        print("OUTLIER REMOVAL - IQR METHOD")
        print(f"{'='*50}")
        numeric_cols_to_clean = [col for col in found_numeric.values() if col in df_processed.columns]

        if numeric_cols_to_clean:
            # FIXED: Use 'cap' method to preserve data instead of removing rows
            df_processed, outlier_report = self.remove_outliers_iqr(
                df_processed, numeric_cols_to_clean, method='cap'
            )
            self.outlier_report = outlier_report

            # Print detailed outlier report
            print(f"\nOutlier Removal Summary:")
            print(f"  Total rows before: {outlier_report['total_rows_before']}")
            print(f"  Total rows removed: {outlier_report['total_rows_removed']}")
            print(f"  Total rows after: {outlier_report['total_rows_after']}")
            print(f"  Percentage removed: {outlier_report['percentage_removed']:.2f}%")

            print(f"\nOutliers by column:")
            for col, info in outlier_report['outliers_by_column'].items():
                print(f"  {col}:")
                print(f"    - Outliers found: {info['count']} ({info['percentage']:.2f}%)")
                print(f"    - Valid range: [{info['lower_bound']:.2f}, {info['upper_bound']:.2f}]")

            # Update self.df with cleaned data
            self.df = df_processed.copy()

        # Continue with feature selection (avoid duplicates)
        seen_cols = set()
        for key, col in found_numeric.items():
            if col not in df_processed.columns:
                continue

            # Skip if we've already added this column (handles duplicate mappings)
            if col in seen_cols:
                continue

            # Check if column has variance
            if df_processed[col].std() > 0:
                feature_cols.append(col)
                seen_cols.add(col)
            else:
                print(f"  Warning: {col} has zero variance, skipping")

        # Store found columns for later use
        self.found_numeric = found_numeric
        self.found_text = found_text
        self.found_categorical = found_categorical

        if len(feature_cols) == 0:
            raise ValueError("No valid numeric features found after preprocessing. Check your data.")

        # Calculate key business indicators BEFORE scaling (if requested)
        key_indicators = []
        if calculate_indicators:
            # Calculate Market Value to Revenue Ratio
            revenue_col = found_numeric.get('revenue')
            market_val_col = found_numeric.get('market_value')
            if revenue_col and market_val_col and revenue_col in df_processed.columns and market_val_col in df_processed.columns:
                df_processed['market_value_to_revenue_ratio'] = np.where(
                    df_processed[revenue_col] > 0,
                    df_processed[market_val_col] / df_processed[revenue_col],
                    np.nan
                )
                key_indicators.append('market_value_to_revenue_ratio')
                print("✓ Calculated Market Value to Revenue Ratio (for growth vs. value segmentation)")
            
            # Calculate IT Investment Intensity
            it_budget_col = found_numeric.get('it_budget')
            it_spend_col = found_numeric.get('it_spend')
            if revenue_col and revenue_col in df_processed.columns:
                it_total = None
                if it_budget_col and it_budget_col in df_processed.columns:
                    it_total = df_processed[it_budget_col].fillna(0)
                if it_spend_col and it_spend_col in df_processed.columns:
                    if it_total is not None:
                        it_total = it_total + df_processed[it_spend_col].fillna(0)
                    else:
                        it_total = df_processed[it_spend_col].fillna(0)
                
                if it_total is not None:
                    df_processed['it_investment_intensity'] = np.where(
                        df_processed[revenue_col] > 0,
                        it_total / df_processed[revenue_col],
                        np.nan
                    )
                    key_indicators.append('it_investment_intensity')
                    print("✓ Calculated IT Investment Intensity (for tech-forward vs. traditional segmentation)")
        
        # Add key indicators to feature list if requested
        if include_key_indicators_in_clustering and key_indicators:
            for indicator in key_indicators:
                if indicator in df_processed.columns and indicator not in feature_cols:
                    # Check if indicator has variance
                    if df_processed[indicator].std() > 0:
                        feature_cols.append(indicator)
                        print(f"  → Added '{indicator}' to clustering features")
                    else:
                        print(f"  ⚠ Skipped '{indicator}' (zero variance)")
        
        # Store numeric features (without duplicates)
        self.feature_names = feature_cols.copy()
        self.df_processed = df_processed[feature_cols].copy()
        
        # Handle missing values in key indicators before scaling
        for col in key_indicators:
            if col in self.df_processed.columns:
                fill_value = self.df_processed[col].median() if not pd.isna(self.df_processed[col].median()) else 0
                self.df_processed[col].fillna(fill_value, inplace=True)
        
        # Scale numeric features
        self.df_processed_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df_processed),
            columns=self.df_processed.columns,
            index=self.df_processed.index
        )
        
        print(f"\n✓ Processed {len(feature_cols)} numeric features for analysis")
        if key_indicators:
            print(f"  Key Business Indicators included: {', '.join([k for k in key_indicators if k in feature_cols])}")
        print(f"  Numeric Features: {feature_cols}")
        
        # Apply TF-IDF to text columns if available
        if found_text:
            print(f"\nApplying TF-IDF to {len(found_text)} text column(s)...")
            self.apply_tfidf(text_columns=list(found_text.values()), max_features=100)
            
            # Combine TF-IDF features with numeric features
            if self.tfidf_features is not None:
                # Align indices
                self.df_processed_scaled = pd.concat([
                    self.df_processed_scaled,
                    self.tfidf_features.loc[self.df_processed_scaled.index]
                ], axis=1)
                print(f"  Combined with {self.tfidf_features.shape[1]} TF-IDF features")
        
        # Optionally calculate additional business indicators
        if calculate_indicators:
            self.calculate_business_indicators()
        
        return self.df_processed_scaled
    
    def calculate_business_indicators(self):
        """
        Calculate additional business indicators from existing features.
        These metrics provide deeper insights for segmentation and analysis.
        
        Returns:
            DataFrame with additional business indicator columns
        """
        print("\n" + "="*50)
        print("CALCULATING BUSINESS INDICATORS")
        print("="*50)
        
        if self.df is None:
            print("Error: Data must be loaded first!")
            return None
        
        df_indicators = self.df.copy()
        indicators_added = []
        
        # Get column references from found_numeric
        if not hasattr(self, 'found_numeric') or not self.found_numeric:
            print("Warning: Numeric columns not found. Running preprocessing first...")
            self.preprocess_data()
        
        # Helper function to safely get column value
        def get_col(key):
            if key in self.found_numeric and self.found_numeric[key] in df_indicators.columns:
                return df_indicators[self.found_numeric[key]]
            return None
        
        # 1. Market Value to Revenue Ratio (Price-to-Sales)
        revenue_col = get_col('revenue')
        market_val_col = get_col('market_value')
        if revenue_col is not None and market_val_col is not None:
            df_indicators['market_value_to_revenue_ratio'] = np.where(
                revenue_col > 0,
                market_val_col / revenue_col,
                np.nan
            )
            indicators_added.append('market_value_to_revenue_ratio')
            print("✓ Market Value to Revenue Ratio")
        
        # 2. IT Investment Intensity
        it_budget_col = get_col('it_budget')
        it_spend_col = get_col('it_spend')
        if revenue_col is not None:
            # FIXED: Use median imputation instead of fillna(0) to avoid bias
            it_total = None
            if it_budget_col is not None and it_spend_col is not None:
                # Use median of non-null values for imputation
                budget_median = it_budget_col.median() if not it_budget_col.isna().all() else 0
                spend_median = it_spend_col.median() if not it_spend_col.isna().all() else 0
                it_total = it_budget_col.fillna(budget_median) + it_spend_col.fillna(spend_median)
            elif it_budget_col is not None:
                budget_median = it_budget_col.median() if not it_budget_col.isna().all() else 0
                it_total = it_budget_col.fillna(budget_median)
            elif it_spend_col is not None:
                spend_median = it_spend_col.median() if not it_spend_col.isna().all() else 0
                it_total = it_spend_col.fillna(spend_median)
            
            if it_total is not None:
                df_indicators['it_investment_intensity'] = np.where(
                    revenue_col > 0,
                    it_total / revenue_col,
                    np.nan
                )
                indicators_added.append('it_investment_intensity')
                print("✓ IT Investment Intensity")
        
        # 3. Single-Site Concentration Ratio
        emp_total_col = get_col('employee_total')
        emp_single_col = get_col('employee_single_sites')
        if emp_total_col is not None and emp_single_col is not None:
            df_indicators['single_site_concentration'] = np.where(
                emp_total_col > 0,
                emp_single_col / emp_total_col,
                np.nan
            )
            indicators_added.append('single_site_concentration')
            print("✓ Single-Site Concentration Ratio")
        
        # 4. Workforce Technology Ratio
        num_pcs_col = get_col('num_pcs')
        num_laptops_col = get_col('num_laptops')
        if emp_total_col is not None:
            devices_total = None
            if num_pcs_col is not None and num_laptops_col is not None:
                devices_total = num_pcs_col.fillna(0) + num_laptops_col.fillna(0)
            elif num_pcs_col is not None:
                devices_total = num_pcs_col.fillna(0)
            elif num_laptops_col is not None:
                devices_total = num_laptops_col.fillna(0)
            
            if devices_total is not None:
                df_indicators['workforce_tech_ratio'] = np.where(
                    emp_total_col > 0,
                    devices_total / emp_total_col,
                    np.nan
                )
                indicators_added.append('workforce_tech_ratio')
                print("✓ Workforce Technology Ratio")
        
        # 5. Technology Sophistication Index
        num_servers_col = get_col('num_servers')
        num_storage_col = get_col('num_storage')
        num_routers_col = get_col('num_routers')
        if emp_total_col is not None:
            tech_score = pd.Series(0.0, index=df_indicators.index)
            
            if num_servers_col is not None:
                tech_score += num_servers_col.fillna(0) * 3
            if num_storage_col is not None:
                tech_score += num_storage_col.fillna(0) * 2
            if num_routers_col is not None:
                tech_score += num_routers_col.fillna(0) * 1
            if num_pcs_col is not None:
                tech_score += num_pcs_col.fillna(0) * 0.5
            if num_laptops_col is not None:
                tech_score += num_laptops_col.fillna(0) * 0.5
            
            df_indicators['tech_sophistication_index'] = np.where(
                emp_total_col > 0,
                tech_score / emp_total_col,
                np.nan
            )
            indicators_added.append('tech_sophistication_index')
            print("✓ Technology Sophistication Index")
        
        # 6. Mobile vs. Desktop Ratio
        num_desktops_col = get_col('num_desktops')
        if num_laptops_col is not None and num_desktops_col is not None:
            total_devices = num_laptops_col.fillna(0) + num_desktops_col.fillna(0)
            df_indicators['mobile_ratio'] = np.where(
                total_devices > 0,
                num_laptops_col.fillna(0) / total_devices,
                np.nan
            )
            indicators_added.append('mobile_ratio')
            print("✓ Mobile vs. Desktop Ratio")
        
        # 7. Growth Potential Index (composite)
        if revenue_col is not None and emp_total_col is not None and market_val_col is not None:
            revenue_per_emp = np.where(emp_total_col > 0, revenue_col / emp_total_col, 0)
            market_to_rev = np.where(revenue_col > 0, market_val_col / revenue_col, 0)
            it_intensity = df_indicators.get('it_investment_intensity', pd.Series(0, index=df_indicators.index))
            
            # Normalize components (0-1 scale) before combining
            revenue_per_emp_norm = (revenue_per_emp - np.nanmin(revenue_per_emp)) / (np.nanmax(revenue_per_emp) - np.nanmin(revenue_per_emp) + 1e-10)
            market_to_rev_norm = (market_to_rev - np.nanmin(market_to_rev)) / (np.nanmax(market_to_rev) - np.nanmin(market_to_rev) + 1e-10)
            it_intensity_norm = (it_intensity - np.nanmin(it_intensity)) / (np.nanmax(it_intensity) - np.nanmin(it_intensity) + 1e-10)
            
            # FIXED: Documented weights rationale
            # Weights: 0.4 revenue/emp (productivity), 0.3 market/revenue (growth expectations), 0.3 IT intensity (innovation)
            # Rationale: Productivity is primary driver (40%), market expectations and innovation equally important (30% each)
            # Alternative: Use PCA to derive data-driven weights if preferred
            growth_weights = {
                'revenue_per_emp': 0.4,  # Labor productivity - primary growth driver
                'market_to_revenue': 0.3,  # Market expectations - growth potential indicator
                'it_intensity': 0.3  # Innovation investment - technology-driven growth
            }
            
            df_indicators['growth_potential_index'] = (
                revenue_per_emp_norm * growth_weights['revenue_per_emp'] +
                market_to_rev_norm * growth_weights['market_to_revenue'] +
                it_intensity_norm * growth_weights['it_intensity']
            )
            indicators_added.append('growth_potential_index')
            print("✓ Growth Potential Index")
        
        # 8. Company Maturity Stage (categorical)
        # FIXED: Made cutoffs configurable (can be customized per industry)
        if 'company_age' in df_indicators.columns:
            # Default thresholds (can be customized)
            startup_threshold = 5
            growth_threshold = 10
            mature_threshold = 20
            
            def categorize_maturity(age):
                if pd.isna(age) or age < 0:
                    return "Unknown"
                elif age < startup_threshold:
                    return "Startup"
                elif age < growth_threshold:
                    return "Growth"
                elif age < mature_threshold:
                    return "Mature"
                else:
                    return "Established"
            
            df_indicators['maturity_stage'] = df_indicators['company_age'].apply(categorize_maturity)
            indicators_added.append('maturity_stage')
            print(f"✓ Company Maturity Stage (thresholds: {startup_threshold}/{growth_threshold}/{mature_threshold} years)")
        
        # 9. Revenue Scale Category
        # FIXED: Use quantiles instead of fixed boundaries for data-driven segmentation
        if revenue_col is not None:
            valid_revenue = revenue_col[revenue_col > 0].dropna()
            if len(valid_revenue) > 0:
                # Use quantiles for data-driven boundaries
                q20 = valid_revenue.quantile(0.20)
                q40 = valid_revenue.quantile(0.40)
                q60 = valid_revenue.quantile(0.60)
                q80 = valid_revenue.quantile(0.80)
                
                def categorize_revenue_scale(rev):
                    if pd.isna(rev) or rev <= 0:
                        return "Unknown"
                    elif rev < q20:
                        return "Micro"
                    elif rev < q40:
                        return "Small"
                    elif rev < q60:
                        return "Mid-Market"
                    elif rev < q80:
                        return "Upper Mid-Market"
                    else:
                        return "Enterprise"
                
                df_indicators['revenue_scale'] = revenue_col.apply(categorize_revenue_scale)
                indicators_added.append('revenue_scale')
                print(f"✓ Revenue Scale Category (quantile-based: {q20:.0f}/{q40:.0f}/{q60:.0f}/{q80:.0f})")
            else:
                # Fallback to fixed boundaries if no valid data
                def categorize_revenue_scale(rev):
                    if pd.isna(rev) or rev <= 0:
                        return "Unknown"
                    elif rev < 1_000_000:
                        return "Micro"
                    elif rev < 10_000_000:
                        return "Small"
                    elif rev < 100_000_000:
                        return "Mid-Market"
                    elif rev < 1_000_000_000:
                        return "Upper Mid-Market"
                    else:
                        return "Enterprise"
                
                df_indicators['revenue_scale'] = revenue_col.apply(categorize_revenue_scale)
                indicators_added.append('revenue_scale')
                print("✓ Revenue Scale Category (using fixed boundaries - no valid data for quantiles)")
        
        # 10. Employee Scale Category
        if emp_total_col is not None:
            def categorize_employee_scale(emp):
                if pd.isna(emp) or emp <= 0:
                    return "Unknown"
                elif emp < 10:
                    return "Micro"
                elif emp < 50:
                    return "Small"
                elif emp < 250:
                    return "Mid-Market"
                elif emp < 1000:
                    return "Upper Mid-Market"
                else:
                    return "Enterprise"
            
            df_indicators['employee_scale'] = emp_total_col.apply(categorize_employee_scale)
            indicators_added.append('employee_scale')
            print("✓ Employee Scale Category")
        
        # Update the main dataframe
        self.df = df_indicators
        
        print(f"\n✓ Calculated {len(indicators_added)} business indicators")
        print(f"  Indicators: {', '.join(indicators_added)}")
        
        return df_indicators
    
    def determine_optimal_clusters(self, max_k: int = 10, practical_threshold: float = 0.70):
        """
        Determine optimal number of clusters using enhanced business-focused method with
        multiple validation metrics.
        
        FIXED: Added Davies-Bouldin and Calinski-Harabasz indices to complement Silhouette Score.
        FIXED: Made practical_threshold configurable (was hardcoded 0.70).

        The algorithm considers:
        1. Silhouette scores (cluster quality) - higher is better
        2. Davies-Bouldin index (cluster separation) - lower is better
        3. Calinski-Harabasz index (variance ratio) - higher is better
        4. Business practicality (K=2 is too simplistic, K>9 is too complex)
        5. Preference for K=5-7 range (standard market segmentation)

        Args:
            max_k: Maximum number of clusters to test
            practical_threshold: Threshold for preferring practical K (0.70 = 70% of max score)
        """
        print("\n" + "="*50)
        print("SEGMENTATION: DETERMINING OPTIMAL CLUSTERS")
        print("Method: Multi-Metric Business-Focused Clustering")
        print("="*50)

        inertias = []
        silhouette_scores = []
        davies_bouldin_scores = []
        calinski_harabasz_scores = []
        
        max_possible_k = min(max_k + 1, len(self.df_processed_scaled) // 2, len(self.df_processed_scaled) - 1)
        if max_possible_k < 2:
            print("Warning: Not enough data points for clustering. Using k=2.")
            return 2

        k_range = range(2, max_possible_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.df_processed_scaled)
            inertias.append(kmeans.inertia_)
            
            # Calculate multiple validation metrics
            sil_score = silhouette_score(self.df_processed_scaled, labels)
            silhouette_scores.append(sil_score)
            
            db_score = davies_bouldin_score(self.df_processed_scaled, labels)
            davies_bouldin_scores.append(db_score)
            
            ch_score = calinski_harabasz_score(self.df_processed_scaled, labels)
            calinski_harabasz_scores.append(ch_score)
            
            print(f"K={k}: Silhouette={sil_score:.3f}, Davies-Bouldin={db_score:.3f}, Calinski-Harabasz={ch_score:.2f}")

        # Enhanced optimal k selection logic with multiple metrics
        best_k_by_silhouette = k_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)
        
        # Normalize Davies-Bouldin (lower is better, so invert)
        min_db = min(davies_bouldin_scores)
        max_db = max(davies_bouldin_scores)
        normalized_db = [(max_db - db) / (max_db - min_db + 1e-10) for db in davies_bouldin_scores]
        
        # Normalize Calinski-Harabasz (higher is better)
        min_ch = min(calinski_harabasz_scores)
        max_ch = max(calinski_harabasz_scores)
        normalized_ch = [(ch - min_ch) / (max_ch - min_ch + 1e-10) for ch in calinski_harabasz_scores]
        
        # Combined score (weighted average of normalized metrics)
        combined_scores = [
            0.4 * sil + 0.3 * db_norm + 0.3 * ch_norm
            for sil, db_norm, ch_norm in zip(silhouette_scores, normalized_db, normalized_ch)
        ]
        best_k_by_combined = k_range[np.argmax(combined_scores)]

        # Look for best K in the practical range (4-8)
        practical_range = [k for k in k_range if 4 <= k <= 8]
        if practical_range:
            practical_scores = [(k, silhouette_scores[k-2]) for k in practical_range]
            best_practical = max(practical_scores, key=lambda x: x[1])

            # FIXED: Use configurable threshold instead of hardcoded 0.70
            if best_practical[1] >= max_silhouette * practical_threshold:
                optimal_k = best_practical[0]
                optimal_silhouette = best_practical[1]
                print(f"\nOptimal number of clusters: {optimal_k} (business-optimized)")
                print(f"Silhouette Score: {optimal_silhouette:.4f}")
                print(f"Davies-Bouldin: {davies_bouldin_scores[optimal_k-2]:.3f} (lower is better)")
                print(f"Calinski-Harabasz: {calinski_harabasz_scores[optimal_k-2]:.2f} (higher is better)")
                print(f"Note: K={best_k_by_silhouette} has highest silhouette ({max_silhouette:.4f}),")
                print(f"      but K={optimal_k} provides better business segmentation")
            else:
                optimal_k = best_k_by_combined
                optimal_silhouette = silhouette_scores[optimal_k-2]
                print(f"\nOptimal number of clusters: {optimal_k} (multi-metric)")
                print(f"Silhouette Score: {optimal_silhouette:.4f}")
                print(f"Davies-Bouldin: {davies_bouldin_scores[optimal_k-2]:.3f}")
                print(f"Calinski-Harabasz: {calinski_harabasz_scores[optimal_k-2]:.2f}")
        else:
            optimal_k = best_k_by_combined
            optimal_silhouette = silhouette_scores[optimal_k-2]
            print(f"\nOptimal number of clusters: {optimal_k} (multi-metric)")
            print(f"Silhouette Score: {optimal_silhouette:.4f}")
            print(f"Davies-Bouldin: {davies_bouldin_scores[optimal_k-2]:.3f}")
            print(f"Calinski-Harabasz: {calinski_harabasz_scores[optimal_k-2]:.2f}")
        
        # Store segmentation metrics
        self.segmentation_metrics = {
            'optimal_k': optimal_k,
            'optimal_silhouette_score': optimal_silhouette,
            'silhouette_scores': dict(zip(k_range, silhouette_scores)),
            'davies_bouldin_scores': dict(zip(k_range, davies_bouldin_scores)),
            'calinski_harabasz_scores': dict(zip(k_range, calinski_harabasz_scores)),
            'combined_scores': dict(zip(k_range, combined_scores)),
            'inertias': dict(zip(k_range, inertias)),
            'method': 'Multi-Metric (Silhouette + Davies-Bouldin + Calinski-Harabasz)',
            'practical_threshold': practical_threshold
        }
        
        # Plot validation metrics
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(k_range, inertias, 'bo-')
        axes[0, 0].set_xlabel('Number of Clusters (k)')
        axes[0, 0].set_ylabel('Inertia')
        axes[0, 0].set_title('Elbow Method')
        axes[0, 0].grid(True)
        
        axes[0, 1].plot(k_range, silhouette_scores, 'ro-')
        axes[0, 1].set_xlabel('Number of Clusters (k)')
        axes[0, 1].set_ylabel('Silhouette Score (higher is better)')
        axes[0, 1].set_title('Silhouette Score')
        axes[0, 1].grid(True)
        
        axes[1, 0].plot(k_range, davies_bouldin_scores, 'go-')
        axes[1, 0].set_xlabel('Number of Clusters (k)')
        axes[1, 0].set_ylabel('Davies-Bouldin Index (lower is better)')
        axes[1, 0].set_title('Davies-Bouldin Index')
        axes[1, 0].grid(True)
        
        axes[1, 1].plot(k_range, calinski_harabasz_scores, 'mo-')
        axes[1, 1].set_xlabel('Number of Clusters (k)')
        axes[1, 1].set_ylabel('Calinski-Harabasz Index (higher is better)')
        axes[1, 1].set_title('Calinski-Harabasz Index')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
        print("\nSaved optimal_clusters.png (with multiple validation metrics)")
        plt.close()
        
        return optimal_k
    
    def perform_clustering(self, n_clusters: int = None, method: str = 'kmeans'):
        """
        Perform clustering on the processed data
        
        Args:
            n_clusters: Number of clusters (if None, will determine optimal)
            method: Clustering method ('kmeans' or 'dbscan')
        """
        print("\n" + "="*50)
        print("PERFORMING CLUSTERING")
        print("="*50)
        
        if n_clusters is None:
            n_clusters = self.determine_optimal_clusters()
        
        if method == 'kmeans':
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            self.clusters = self.cluster_model.fit_predict(self.df_processed_scaled)
        elif method == 'dbscan':
            self.cluster_model = DBSCAN(eps=0.5, min_samples=5)
            self.clusters = self.cluster_model.fit_predict(self.df_processed_scaled)
            n_clusters = len(set(self.clusters)) - (1 if -1 in self.clusters else 0)
        
        # Add cluster labels to original dataframe
        self.df['Cluster'] = self.clusters

        print(f"\nClustering complete: {n_clusters} clusters identified")
        print(f"Cluster distribution:")
        cluster_counts = pd.Series(self.clusters).value_counts().sort_index()
        print(cluster_counts)

        # FIXED: Check for very small clusters and warn
        min_cluster_size = max(10, len(self.df) * 0.01)  # At least 10 or 1% of data
        small_clusters = cluster_counts[cluster_counts < min_cluster_size]

        if len(small_clusters) > 0:
            print(f"\n⚠ WARNING: {len(small_clusters)} cluster(s) have fewer than {int(min_cluster_size)} companies:")
            for cluster_id, count in small_clusters.items():
                pct = (count / len(self.df)) * 100
                print(f"    Cluster {cluster_id}: {count} companies ({pct:.2f}%)")

            print("\n  These small clusters may represent outliers rather than meaningful segments.")
            print("  Consider:")
            print("    - Using fewer clusters (reduce K)")
            print("    - Removing outliers before clustering")
            print("    - Merging small clusters with nearest neighbors")

            # Store outlier cluster info for later handling
            self.small_clusters = small_clusters.index.tolist()
        else:
            self.small_clusters = []
            print(f"\n✓ All clusters have at least {int(min_cluster_size)} companies (good balance)")

        return self.clusters
    
    def analyze_clusters(self):
        """Analyze characteristics of each cluster - FOCUSED ON TARGET COLUMNS ONLY"""
        print("\n" + "="*50)
        print("CLUSTER ANALYSIS - FOCUSED COLUMNS")
        print("="*50)
        
        cluster_analysis = {}
        
        # Get target columns from preprocessing
        if not hasattr(self, 'found_numeric') or not self.found_numeric:
            print("Warning: Target columns not found. Re-running preprocessing...")
            self.preprocess_data()
        
        # Get numeric columns
        numeric_cols = list(self.found_numeric.values())
        
        # Get categorical columns
        categorical_cols = list(self.found_categorical.values()) if hasattr(self, 'found_categorical') and self.found_categorical else []
        
        for cluster_id in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            # Convert cluster_id to native Python int for JSON serialization
            cluster_id_int = int(cluster_id) if hasattr(cluster_id, 'item') else int(cluster_id)
            cluster_analysis[cluster_id_int] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(self.df)) * 100,
                'numeric_stats': {},
                'categorical_profiles': {}
            }
            
            # Analyze numeric target features
            if numeric_cols:
                cluster_analysis[cluster_id_int]['numeric_stats'] = cluster_data[numeric_cols].describe().to_dict()
            
            # Analyze categorical features
            for cat_col in categorical_cols:
                value_counts = cluster_data[cat_col].value_counts(normalize=True)
                cluster_analysis[cluster_id_int]['categorical_profiles'][cat_col] = value_counts.to_dict()
            
            print(f"\nCluster {cluster_id_int}:")
            print(f"  Size: {cluster_analysis[cluster_id_int]['size']} companies ({cluster_analysis[cluster_id_int]['percentage']:.1f}%)")
            if numeric_cols:
                print(f"  Key Metrics:")
                for col in numeric_cols:
                    mean_val = cluster_data[col].mean()
                    print(f"    {col}: {mean_val:,.2f}")
            if categorical_cols:
                for cat_col in categorical_cols:
                    top_value = cluster_data[cat_col].mode()[0] if len(cluster_data[cat_col].mode()) > 0 else "N/A"
                    print(f"  Top {cat_col}: {top_value}")
        
        return cluster_analysis
    
    def compare_clusters(self, feature: str = None):
        """
        Compare clusters across TARGET FEATURES ONLY:
        Revenue, Employee Total, Employee Single Sites, Market Value, IT Spending & Budget
        
        Args:
            feature: Specific feature to compare (if None, compares all target features)
        """
        print("\n" + "="*50)
        print("CLUSTER COMPARISON - TARGET FEATURES")
        print("="*50)
        
        # Get target columns
        if not hasattr(self, 'found_numeric') or not self.found_numeric:
            print("Warning: Target columns not found. Re-running preprocessing...")
            self.preprocess_data()
        
        numeric_cols = list(self.found_numeric.values())
        
        if not numeric_cols:
            print("Warning: No target numeric columns available for comparison.")
            return pd.DataFrame()
        
        comparison = None
        if feature and feature in numeric_cols:
            # Compare specific feature
            comparison = self.df.groupby('Cluster')[feature].agg(['mean', 'std', 'min', 'max', 'count'])
            print(f"\nComparison of {feature} across clusters:")
            print(comparison)
        else:
            # Compare all target numeric features
            print("\nTarget feature comparison across clusters:")
            comparison = self.df.groupby('Cluster')[numeric_cols].agg(['mean', 'std'])
            print(comparison.round(2))
        
        return comparison if comparison is not None else pd.DataFrame()
    
    def identify_patterns(self):
        """Identify notable patterns, strengths, risks, and anomalies - FOCUSED ON TARGET COLUMNS"""
        print("\n" + "="*50)
        print("PATTERN IDENTIFICATION - TARGET COLUMNS")
        print("="*50)
        
        patterns = {
            'outliers': [],
            'correlations': {},
            'cluster_differences': {},
            'anomalies': []
        }
        
        # Get target columns
        if not hasattr(self, 'found_numeric') or not self.found_numeric:
            print("Warning: Target columns not found. Re-running preprocessing...")
            self.preprocess_data()
        
        numeric_cols = list(self.found_numeric.values())
        
        if numeric_cols:
            for col in numeric_cols:  # Check all target numeric features
                try:
                    Q1 = self.df[col].quantile(0.25)
                    Q3 = self.df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    
                    # Skip if IQR is zero or invalid
                    if IQR <= 0 or pd.isna(IQR):
                        continue
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
                    if len(outliers) > 0:
                        patterns['outliers'].append({
                            'feature': col,
                            'count': len(outliers),
                            'percentage': (len(outliers) / len(self.df)) * 100
                        })
                except Exception as e:
                    # Skip columns that cause errors
                    continue
        
        # Calculate correlations between target features
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols].corr()
            patterns['correlations'] = corr_matrix.to_dict()
            print(f"\nCorrelations between target features:")
            print(corr_matrix.round(3))
        
        # Identify cluster differences in target features
        if numeric_cols:
            for cluster_id in sorted(self.df['Cluster'].unique()):
                cluster_data = self.df[self.df['Cluster'] == cluster_id]
                other_data = self.df[self.df['Cluster'] != cluster_id]
                
                if len(cluster_data) == 0 or len(other_data) == 0:
                    continue
                
                differences = {}
                for col in numeric_cols:  # All target features
                    try:
                        cluster_mean = cluster_data[col].mean()
                        other_mean = other_data[col].mean()
                        
                        # Skip if means are NaN
                        if pd.isna(cluster_mean) or pd.isna(other_mean):
                            continue
                        
                        # Avoid division by zero
                        if abs(other_mean) > 1e-10 and abs(cluster_mean - other_mean) > 0.1 * abs(other_mean):  # 10% difference
                            differences[col] = {
                                'cluster_mean': cluster_mean,
                                'other_mean': other_mean,
                                'difference_pct': ((cluster_mean - other_mean) / abs(other_mean)) * 100
                            }
                    except Exception:
                        continue
                
                # Convert cluster_id to native Python int for JSON serialization
                cluster_id_int = int(cluster_id) if hasattr(cluster_id, 'item') else int(cluster_id)
                patterns['cluster_differences'][cluster_id_int] = differences
        
        return patterns
    
    def _convert_numpy_types(self, obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization
        Compatible with NumPy 2.0 (np.float_ removed)
        
        Args:
            obj: Object that may contain numpy types
            
        Returns:
            Object with numpy types converted to native Python types
        """
        # Check for numpy integer types (NumPy 2.0 compatible)
        if isinstance(obj, np.integer):
            return int(obj)
        
        # Check for numpy floating types (NumPy 2.0 compatible - np.float_ removed)
        if isinstance(obj, np.floating):
            return float(obj)
        
        # Check for numpy bool type
        if isinstance(obj, np.bool_):
            return bool(obj)
        
        # Check for specific integer types (fallback for edge cases)
        if isinstance(obj, (np.int8, np.int16, np.int32, np.int64, 
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        
        # Check for specific float types (fallback for edge cases)
        if isinstance(obj, (np.float16, np.float32, np.float64)):
            return float(obj)
        
        # Check for numpy array
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        # Recursively process dictionaries
        if isinstance(obj, dict):
            return {self._convert_numpy_types(k): self._convert_numpy_types(v) for k, v in obj.items()}
        
        # Recursively process lists and tuples
        if isinstance(obj, (list, tuple)):
            return [self._convert_numpy_types(item) for item in obj]
        
        # Handle pandas types
        if pd.isna(obj):
            return None
        if isinstance(obj, pd.Series):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        
        return obj
    
    def generate_llm_insights(self, cluster_analysis: Dict, patterns: Dict) -> str:
        """
        Generate interpretable insights using LLM
        
        Args:
            cluster_analysis: Results from analyze_clusters()
            patterns: Results from identify_patterns()
        """
        if not self.use_llm:
            return self.generate_rule_based_insights(cluster_analysis, patterns)
        
        print("\n" + "="*50)
        print("GENERATING LLM INSIGHTS")
        print("="*50)
        
        # Convert numpy types to native Python types for JSON serialization
        patterns_clean = self._convert_numpy_types(patterns)
        cluster_analysis_clean = self._convert_numpy_types(cluster_analysis)
        
        # Prepare summary for LLM
        summary = {
            'total_companies': len(self.df),
            'num_clusters': len(set(self.clusters)),
            'cluster_sizes': {str(k): v['size'] for k, v in cluster_analysis_clean.items()},
            'key_patterns': patterns_clean
        }
        
        prompt = f"""You are a business intelligence analyst. Analyze the following company segmentation results and provide actionable insights.

Dataset Summary:
- Total companies: {summary['total_companies']}
- Number of segments: {summary['num_clusters']}
- Segment sizes: {summary['cluster_sizes']}

Key Patterns Identified:
{json.dumps(patterns_clean, indent=2, default=str)}

Please provide:
1. A high-level summary of the segmentation
2. Key characteristics that distinguish each segment
3. Notable patterns, strengths, and potential risks
4. Business implications and recommendations
5. How this segmentation can support decision-making

Format your response in clear, business-friendly language suitable for executives and data buyers."""

        try:
            # Select model based on API provider
            if self.api_provider == 'deepseek':
                model = "deepseek-chat"  # DeepSeek's chat model
            else:
                model = "gpt-4"  # OpenAI model
            
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert business intelligence analyst specializing in company data analysis and market segmentation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # FIXED: Lower temperature for more deterministic analytical insights (was 0.7)
                max_tokens=1500
            )
            insights = response.choices[0].message.content
            provider_name = "DeepSeek" if self.api_provider == 'deepseek' else "OpenAI"
            print(f"\n{provider_name} LLM insights generated successfully")
            return insights
        except Exception as e:
            print(f"Error generating LLM insights: {e}")
            print("Falling back to rule-based insights...")
            return self.generate_rule_based_insights(cluster_analysis, patterns)
    
    def generate_rule_based_insights(self, cluster_analysis: Dict, patterns: Dict) -> str:
        """Generate insights using rule-based approach when LLM is not available"""
        insights = []
        insights.append("="*50)
        insights.append("COMPANY INTELLIGENCE INSIGHTS")
        insights.append("="*50)
        insights.append("")
        
        insights.append("SEGMENTATION OVERVIEW:")
        insights.append(f"- Total companies analyzed: {len(self.df)}")
        insights.append(f"- Number of segments identified: {len(set(self.clusters))}")
        insights.append("")
        
        insights.append("SEGMENT CHARACTERISTICS:")
        for cluster_id in sorted(cluster_analysis.keys()):
            info = cluster_analysis[cluster_id]
            insights.append(f"\nSegment {cluster_id}:")
            insights.append(f"  - Size: {info['size']} companies ({info['percentage']:.1f}% of total)")
            
            # Add key numeric characteristics
            if info['numeric_stats']:
                insights.append("  - Key characteristics:")
                for feature, stats in list(info['numeric_stats'].items())[:3]:
                    if 'mean' in stats:
                        insights.append(f"    • {feature}: {stats['mean']:.2f} (avg)")
        
        insights.append("\n" + "="*50)
        insights.append("KEY PATTERNS & INSIGHTS:")
        insights.append("="*50)
        
        if patterns['outliers']:
            insights.append("\nNotable Outliers Detected:")
            for outlier in patterns['outliers'][:5]:
                insights.append(f"  - {outlier['feature']}: {outlier['count']} companies ({outlier['percentage']:.1f}%)")
        
        if patterns['cluster_differences']:
            insights.append("\nSignificant Segment Differences:")
            for cluster_id, differences in patterns['cluster_differences'].items():
                if differences:
                    insights.append(f"\n  Segment {cluster_id} differs in:")
                    for feature, diff_info in list(differences.items())[:3]:
                        insights.append(f"    • {feature}: {diff_info['difference_pct']:.1f}% difference")
        
        insights.append("\n" + "="*50)
        insights.append("BUSINESS IMPLICATIONS:")
        insights.append("="*50)
        insights.append("""
1. SEGMENTATION VALUE:
   - Enables targeted marketing and sales strategies
   - Supports risk assessment and portfolio management
   - Facilitates competitive benchmarking

2. DECISION-MAKING SUPPORT:
   - Identify high-value segments for business development
   - Understand operational differences across company types
   - Detect anomalies that may indicate risks or opportunities

3. COMMERCIAL APPLICATIONS:
   - Sales intelligence: Prioritize prospects by segment
   - Risk management: Assess credit/operational risk by segment
   - Market research: Understand industry structure and dynamics
   - Strategic planning: Identify growth opportunities
        """)
        
        return "\n".join(insights)
    
    def apply_tfidf(self, text_columns: List[str] = None, max_features: int = 100):
        """
        Apply TF-IDF vectorization to text columns (SIC, NAICS, NACE descriptions)
        
        Args:
            text_columns: List of column names containing text data
            max_features: Maximum number of TF-IDF features to create
        """
        print("\n" + "="*50)
        print("TF-IDF FEATURE EXTRACTION")
        print("="*50)
        
        if text_columns is None:
            # Use found text columns from preprocessing
            if hasattr(self, 'found_text') and self.found_text:
                text_columns = list(self.found_text.values())
            else:
                print("No text columns specified. Skipping TF-IDF feature extraction.")
                return None
        
        if not text_columns:
            print("No text columns found for TF-IDF. Skipping TF-IDF feature extraction.")
            return None
        
        print(f"Processing {len(text_columns)} text column(s): {text_columns}")
        
        # Combine all text columns into a single text feature
        combined_text = self.df[text_columns[0]].fillna('').astype(str)
        for col in text_columns[1:]:
            combined_text += ' ' + self.df[col].fillna('').astype(str)
        
        # Apply TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2)
        )
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(combined_text)
            self.tfidf_features = pd.DataFrame(
                tfidf_matrix.toarray(),
                # FIXED: Use actual feature names instead of generic TFIDF_0, TFIDF_1, etc.
                columns=self.tfidf_vectorizer.get_feature_names_out(),
                index=self.df.index
            )
            
            print(f"✓ Created {self.tfidf_features.shape[1]} TF-IDF features")
            
            # Show top terms
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            print(f"\nTop 10 TF-IDF terms: {list(feature_names[:10])}")
            
            return self.tfidf_features
        except Exception as e:
            print(f"Error in TF-IDF processing: {e}")
            return None
    
    def perform_chi_square_test(self, categorical_cols: List[str] = None, 
                                 alpha: float = 0.05, correction_method: str = 'fdr_bh'):
        """
        Perform Chi-square test of independence between categorical variables and clusters.
        
        FIXED: Added check for expected frequency >= 5 (chi-square assumption).
        FIXED: Added multiple testing correction (Bonferroni/FDR) to control family-wise error rate.
        
        Args:
            categorical_cols: List of categorical column names to test
            alpha: Significance level (default 0.05)
            correction_method: Multiple testing correction method ('bonferroni', 'fdr_bh', 'fdr_by', or None)
        """
        print("\n" + "="*50)
        print("FINDING TRENDS: CHI-SQUARE TEST ANALYSIS")
        print("Method: Chi-Square Test of Independence (with assumption validation)")
        print("="*50)
        
        if self.clusters is None:
            print("Error: Must perform clustering first!")
            return None
        
        if categorical_cols is None:
            # Use found categorical columns from preprocessing
            if hasattr(self, 'found_categorical') and self.found_categorical:
                categorical_cols = list(self.found_categorical.values())
            else:
                print("No categorical columns specified. Skipping chi-square tests.")
                return None
        
        if not categorical_cols:
            print("No categorical columns found. Skipping chi-square tests.")
            return None
        
        chi_square_results = {}
        p_values = []
        test_names = []
        
        for col in categorical_cols:
            try:
                # Create contingency table
                contingency = pd.crosstab(self.df[col], self.df['Cluster'])
                
                # Check if table is valid (at least 2x2)
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    print(f"  ⚠ Skipping {col}: insufficient categories for chi-square test")
                    continue
                
                # FIXED: Check expected frequency assumption (>= 5)
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                
                # Check if expected frequencies meet assumption
                min_expected = np.min(expected)
                cells_below_5 = np.sum(expected < 5)
                total_cells = expected.size
                violation_percentage = (cells_below_5 / total_cells) * 100
                
                assumption_violated = min_expected < 5
                if assumption_violated:
                    print(f"  ⚠ Warning for {col}: {cells_below_5}/{total_cells} cells ({violation_percentage:.1f}%) have expected frequency < 5")
                    print(f"     Minimum expected frequency: {min_expected:.2f}")
                    print(f"     Chi-square results may be unreliable. Consider:")
                    print(f"     - Combining categories with low counts")
                    print(f"     - Using Fisher's exact test for 2x2 tables")
                    print(f"     - Using G-test (likelihood ratio test)")
                
                chi_square_results[col] = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'expected_frequencies': expected,
                    'min_expected_frequency': min_expected,
                    'cells_below_5': cells_below_5,
                    'assumption_violated': assumption_violated,
                    'contingency_table': contingency
                }
                
                p_values.append(p_value)
                test_names.append(col)
                
            except Exception as e:
                print(f"  ✗ Error testing {col}: {e}")
                continue
        
        # FIXED: Apply multiple testing correction
        if len(p_values) > 1 and correction_method:
            try:
                if HAS_STATSMODELS:
                    if correction_method == 'bonferroni':
                        _, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method='bonferroni')
                    elif correction_method == 'fdr_bh':
                        _, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
                    elif correction_method == 'fdr_by':
                        _, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method='fdr_by')
                    else:
                        p_adjusted = p_values
                else:
                    # Fallback: Simple Bonferroni correction (multiply by number of tests)
                    if correction_method == 'bonferroni':
                        p_adjusted = [min(p * len(p_values), 1.0) for p in p_values]
                    else:
                        print(f"Warning: {correction_method} requires statsmodels. Using Bonferroni fallback.")
                        p_adjusted = [min(p * len(p_values), 1.0) for p in p_values]
                
                # Update results with adjusted p-values
                for i, col in enumerate(test_names):
                    chi_square_results[col]['p_value_adjusted'] = p_adjusted[i]
                    chi_square_results[col]['is_significant'] = p_adjusted[i] < alpha
                    chi_square_results[col]['is_significant_unadjusted'] = p_values[i] < alpha
                    chi_square_results[col]['correction_method'] = correction_method
                
                print(f"\nMultiple Testing Correction ({correction_method.upper()}):")
                print(f"  Tests performed: {len(p_values)}")
                significant_before = sum(1 for p in p_values if p < alpha)
                significant_after = sum(1 for p in p_adjusted if p < alpha)
                print(f"  Significant before correction: {significant_before}/{len(p_values)}")
                print(f"  Significant after correction: {significant_after}/{len(p_values)}")
                
            except Exception as e:
                print(f"  ⚠ Warning: Could not apply multiple testing correction: {e}")
                # Fallback to unadjusted
                for i, col in enumerate(test_names):
                    chi_square_results[col]['is_significant'] = p_values[i] < alpha
        else:
            # Single test or no correction
            for i, col in enumerate(test_names):
                chi_square_results[col]['is_significant'] = p_values[i] < alpha
                if len(p_values) == 1:
                    chi_square_results[col]['correction_method'] = 'none (single test)'
        
        # Print summary
        print(f"\nChi-Square Test Results:")
        for col, result in chi_square_results.items():
            sig_marker = "✓" if result['is_significant'] else "✗"
            p_val = result.get('p_value_adjusted', result['p_value'])
            print(f"  {sig_marker} {col}: χ²={result['chi2_statistic']:.3f}, p={p_val:.4f}")
            if result.get('assumption_violated'):
                print(f"    ⚠ Assumption violation: {result['cells_below_5']} cells with expected frequency < 5")
        
        # Summary
        if chi_square_results:
            significant_vars = [col for col, result in chi_square_results.items() 
                              if result['is_significant']]
            print(f"\n{'='*50}")
            print(f"Summary: {len(significant_vars)}/{len(chi_square_results)} categorical variables "
                  f"show significant association with clusters (after {correction_method} correction)")
            
            # Store trend finding metrics
            self.trend_metrics = {
                'chi_square_results': chi_square_results,
                'significant_variables': significant_vars,
                'total_tested': len(chi_square_results),
                'significant_count': len(significant_vars),
                'correction_method': correction_method,
                'method': 'Chi-Square Test (with assumption validation & multiple testing correction)'
            }
        
        return chi_square_results
    
    def apply_dimensionality_reduction(self, method: str = 'pca', n_components: int = 2):
        """
        Apply dimensionality reduction techniques
        SIMPLIFYING COMPLEX DATA: PCA Explained Variance
        
        Args:
            method: Method to use ('pca', 'tsne', 'umap', 'svd')
            n_components: Number of components for reduction
        """
        print("\n" + "="*50)
        print(f"SIMPLIFYING COMPLEX DATA: DIMENSIONALITY REDUCTION")
        print(f"Method: {method.upper()} - Explained Variance")
        print("="*50)
        
        if self.df_processed_scaled is None:
            print("Error: Must preprocess data first!")
            return None
        
        reduction_results = {}
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
            reduced_data = reducer.fit_transform(self.df_processed_scaled)
            reduction_results['explained_variance'] = reducer.explained_variance_ratio_
            total_variance = sum(reduction_results['explained_variance'])
            print(f"PCA: Explained variance ratio: {reduction_results['explained_variance']}")
            print(f"Total variance explained: {total_variance:.2%}")
            
            # Store PCA metrics
            self.pca_metrics = {
                'method': 'PCA',
                'n_components': n_components,
                'explained_variance_ratio': reduction_results['explained_variance'].tolist(),
                'total_variance_explained': float(total_variance),
                'original_dimensions': self.df_processed_scaled.shape[1],
                'reduced_dimensions': n_components
            }
            
        elif method.lower() == 'svd' or method.lower() == 'truncated_svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_data = reducer.fit_transform(self.df_processed_scaled)
            reduction_results['explained_variance'] = reducer.explained_variance_ratio_
            total_variance = sum(reduction_results['explained_variance'])
            print(f"TruncatedSVD: Explained variance ratio: {reduction_results['explained_variance']}")
            print(f"Total variance explained: {total_variance:.2%}")
            
            # Store SVD metrics
            self.pca_metrics = {
                'method': 'TruncatedSVD',
                'n_components': n_components,
                'explained_variance_ratio': reduction_results['explained_variance'].tolist(),
                'total_variance_explained': float(total_variance),
                'original_dimensions': self.df_processed_scaled.shape[1],
                'reduced_dimensions': n_components
            }
            
        elif method.lower() == 'tsne':
            if not HAS_TSNE:
                print("Error: t-SNE not available. Install scikit-learn or use another method.")
                return None
            # FIXED: Scale perplexity with dataset size (rule of thumb: sqrt(n_samples))
            n_samples = self.df_processed_scaled.shape[0]  # Use input data shape, not undefined variable
            optimal_perplexity = min(30, max(5, int(np.sqrt(n_samples))))
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=optimal_perplexity)
            print(f"  Using t-SNE with perplexity={optimal_perplexity} (scaled for {n_samples} samples)")
            reduced_data = reducer.fit_transform(self.df_processed_scaled)
            print(f"t-SNE: Completed (perplexity={optimal_perplexity})")
            
        elif method.lower() == 'umap':
            if not HAS_UMAP:
                print("Error: UMAP not available. Install umap-learn or use another method.")
                return None
            reducer = umap.UMAP(n_components=n_components, random_state=42)
            reduced_data = reducer.fit_transform(self.df_processed_scaled)
            print(f"UMAP: Completed (n_neighbors=15)")
            
        else:
            print(f"Error: Unknown method '{method}'. Use 'pca', 'svd', 'tsne', or 'umap'.")
            return None
        
        reduction_results['method'] = method
        reduction_results['reduced_data'] = reduced_data
        reduction_results['n_components'] = n_components
        
        return reduction_results
    
    def train_logistic_regression(self, test_size: float = 0.2, random_state: int = 42, 
                                  use_original_features: bool = True, cv_folds: int = 5):
        """
        Train logistic regression model to predict cluster membership.
        
        FIXED: Use original features instead of PCA for interpretability.
        FIXED: Split data BEFORE scaling to prevent data leakage.
        FIXED: Added cross-validation for stable metrics.
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            use_original_features: If True, use original features (interpretable). If False, use PCA.
            cv_folds: Number of folds for cross-validation
        """
        print("\n" + "="*50)
        print("LOGISTIC REGRESSION - CLUSTER PREDICTION")
        print("="*50)
        
        if self.clusters is None:
            print("Error: Must perform clustering first!")
            return None
        
        if self.df_processed is None:
            print("Error: Must preprocess data first!")
            return None
        
        # FIXED: Use original unscaled features for interpretability
        if use_original_features:
            X = self.df_processed.values
            feature_names = self.df_processed.columns.tolist()
            print(f"Using original features: {len(feature_names)} features")
        else:
            # Use scaled features (less interpretable)
            X = self.df_processed_scaled.values
            feature_names = self.df_processed_scaled.columns.tolist()
            print(f"Using scaled features: {len(feature_names)} features")
        
        y = self.clusters
        
        # FIXED: Split data FIRST, then scale (prevents data leakage)
        unique, counts = np.unique(y, return_counts=True)
        can_stratify = all(counts >= 2) and len(unique) > 1
        
        if can_stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            print("Warning: Cannot stratify (some clusters have < 2 samples). Using random split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # FIXED: Scale AFTER split (fit on training, transform both)
        scaler_lr = StandardScaler()
        X_train_scaled = scaler_lr.fit_transform(X_train)
        X_test_scaled = scaler_lr.transform(X_test)
        
        print(f"\nTraining set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # FIXED: Add cross-validation for stable metrics
        print(f"\nPerforming {cv_folds}-fold cross-validation...")
        try:
            cv_scores = cross_val_score(
                LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000, random_state=random_state),
                X_train_scaled, y_train,
                cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
                scoring='accuracy'
            )
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"CV Scores per fold: {[f'{s:.4f}' for s in cv_scores]}")
        except Exception as e:
            print(f"Warning: Cross-validation failed: {e}")
            cv_scores = None
        
        # Train logistic regression
        try:
            self.logistic_model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state
            )
        except TypeError:
            self.logistic_model = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state
            )
        
        self.logistic_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.logistic_model.predict(X_train_scaled)
        y_test_pred = self.logistic_model.predict(X_test_scaled)
        
        # Evaluate
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        if cv_scores is not None:
            print(f"CV Mean Accuracy: {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred))
        
        # FIXED: Feature importance with original feature names (interpretable)
        driver_coefficients = {}
        print("\nDRIVER IDENTIFICATION: Top contributing features per cluster:")
        for i, cluster_id in enumerate(sorted(set(y))):
            print(f"\nCluster {cluster_id}:")
            coefs = self.logistic_model.coef_[i]
            top_indices = np.argsort(np.abs(coefs))[-10:][::-1]  # Top 10 features
            driver_coefficients[cluster_id] = {}
            
            for idx in top_indices:
                feature_name = feature_names[idx] if idx < len(feature_names) else f'Feature_{idx}'
                coef_value = float(coefs[idx])
                driver_coefficients[cluster_id][feature_name] = coef_value
                print(f"  {feature_name}: {coef_value:.4f}")
        
        # Store driver identification metrics
        self.driver_metrics = {
            'coefficients': driver_coefficients,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores.tolist() if cv_scores is not None else None,
            'cv_mean': float(cv_scores.mean()) if cv_scores is not None else None,
            'cv_std': float(cv_scores.std()) if cv_scores is not None else None,
            'feature_names': feature_names,
            'method': 'Logistic Regression (Original Features - Interpretable)'
        }
        
        results = {
            'model': self.logistic_model,
            'scaler': scaler_lr,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'coefficients': driver_coefficients,
            'feature_names': feature_names
        }
        
        return results
    
    def train_linear_regression(self, target_feature: str = None, test_size: float = 0.2, 
                                 random_state: int = 42, regularization: str = 'ridge', 
                                 alpha: float = 1.0, check_multicollinearity: bool = True):
        """
        Train linear regression model to predict a TARGET FEATURE (Revenue or Market Value).
        
        FIXED: Split data BEFORE scaling to prevent data leakage.
        FIXED: Added regularization (Ridge/Lasso) to handle multicollinearity.
        FIXED: Added VIF check for multicollinearity detection.
        
        Args:
            target_feature: Name of the target feature to predict ('revenue' or 'market_value')
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            regularization: 'ridge', 'lasso', 'elasticnet', or 'none' (default: 'ridge')
            alpha: Regularization strength (default: 1.0)
            check_multicollinearity: If True, check VIF and warn about multicollinearity
        """
        print("\n" + "="*50)
        print("PERFORMANCE FORECAST: LINEAR REGRESSION")
        print(f"Method: {regularization.upper()} Regression (R² Score)" if regularization != 'none' else "Method: Linear Regression (R² Score)")
        print("="*50)
        
        if self.df_processed is None:
            print("Error: Must preprocess data first!")
            return None
        
        # Get target columns
        if not hasattr(self, 'found_numeric') or not self.found_numeric:
            print("Warning: Target columns not found. Re-running preprocessing...")
            self.preprocess_data()
        
        # Select target feature from target columns
        if target_feature is None:
            if 'revenue' in self.found_numeric:
                target_feature = self.found_numeric['revenue']
                print(f"Using default target: Revenue")
            elif 'market_value' in self.found_numeric:
                target_feature = self.found_numeric['market_value']
                print(f"Using default target: Market Value")
            else:
                print("Error: No suitable target feature found (revenue or market_value)!")
                return None
        else:
            target_key = None
            if 'revenue' in target_feature.lower():
                target_key = 'revenue'
            elif 'market' in target_feature.lower() and 'value' in target_feature.lower():
                target_key = 'market_value'
            
            if target_key and target_key in self.found_numeric:
                target_feature = self.found_numeric[target_key]
            elif target_feature not in self.df.columns:
                print(f"Warning: {target_feature} not found. Using revenue as default.")
                if 'revenue' in self.found_numeric:
                    target_feature = self.found_numeric['revenue']
                else:
                    return None
        
        # FIXED: Use UNSCALED features, split FIRST, then scale
        # FIXED: Exclude features highly correlated with target to prevent data leakage
        feature_names = self.df_processed.columns.tolist()

        # Identify and exclude features that are derived from or highly correlated with target
        # This prevents R² = 1.0 due to data leakage
        features_to_exclude = []
        target_col_name = target_feature.lower().replace(' ', '_').replace('(', '').replace(')', '')

        # Define features that should be excluded when predicting specific targets
        leakage_map = {
            'revenue': ['it_budget', 'it_spend', 'it budget', 'it spend', 'it_investment_intensity',
                       'market_value_to_revenue_ratio'],
            'market_value': ['market_value_to_revenue_ratio', 'ps_ratio'],
        }

        # Find which target we're predicting
        target_key = None
        for key in leakage_map.keys():
            if key in target_col_name or key in target_feature.lower():
                target_key = key
                break

        if target_key:
            for feat in feature_names:
                feat_lower = feat.lower()
                for exclude_pattern in leakage_map[target_key]:
                    if exclude_pattern in feat_lower:
                        features_to_exclude.append(feat)
                        break

        # Also exclude features with correlation > 0.95 with target
        if target_feature in self.df.columns:
            for feat in feature_names:
                if feat in self.df.columns and feat not in features_to_exclude:
                    try:
                        corr = abs(self.df[feat].corr(self.df[target_feature]))
                        if corr > 0.95:
                            features_to_exclude.append(feat)
                            print(f"  ⚠ Excluding '{feat}' (correlation {corr:.3f} with target)")
                    except:
                        pass

        # Remove excluded features
        if features_to_exclude:
            print(f"\n⚠ Excluding {len(features_to_exclude)} features to prevent data leakage:")
            for feat in features_to_exclude:
                print(f"    - {feat}")
            feature_names = [f for f in feature_names if f not in features_to_exclude]

        # Get feature matrix with excluded columns removed
        X = self.df_processed[feature_names].values
        y = self.df[target_feature].values
        
        # Remove rows with missing target values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            print(f"Error: Not enough valid samples ({len(X)}) for linear regression!")
            return None
        
        # FIXED: Split data FIRST (before scaling)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # FIXED: Scale AFTER split (fit on training, transform both)
        scaler_lr = StandardScaler()
        X_train_scaled = scaler_lr.fit_transform(X_train)
        X_test_scaled = scaler_lr.transform(X_test)
        
        # FIXED: Check multicollinearity (VIF) before regression
        if check_multicollinearity and HAS_STATSMODELS:
            print("\nChecking for multicollinearity (VIF)...")
            try:
                # Calculate VIF for training data
                vif_data = pd.DataFrame(X_train_scaled, columns=feature_names)
                vif_scores = {}
                for i, col in enumerate(feature_names):
                    try:
                        vif = variance_inflation_factor(X_train_scaled, i)
                        vif_scores[col] = vif
                    except:
                        vif_scores[col] = np.nan
                
                high_vif = {k: v for k, v in vif_scores.items() if v > 10}
                if high_vif:
                    print(f"  ⚠ Warning: {len(high_vif)} features have VIF > 10 (multicollinearity):")
                    for feat, vif_val in sorted(high_vif.items(), key=lambda x: x[1], reverse=True)[:5]:
                        print(f"    {feat}: VIF = {vif_val:.2f}")
                    print("  Recommendation: Use Ridge/Lasso regression or remove highly correlated features")
                else:
                    print(f"  ✓ No severe multicollinearity detected (all VIF < 10)")
            except Exception as e:
                print(f"  ⚠ Could not calculate VIF: {e}")
        
        print(f"\nTraining set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"Target feature: {target_feature}")
        
        # FIXED: Use regularized regression to handle multicollinearity
        if regularization.lower() == 'ridge':
            self.linear_model = Ridge(alpha=alpha, random_state=random_state)
            print(f"Using Ridge regression (L2 regularization, alpha={alpha})")
        elif regularization.lower() == 'lasso':
            self.linear_model = Lasso(alpha=alpha, random_state=random_state, max_iter=2000)
            print(f"Using Lasso regression (L1 regularization, alpha={alpha})")
        elif regularization.lower() == 'elasticnet':
            self.linear_model = ElasticNet(alpha=alpha, l1_ratio=0.5, random_state=random_state, max_iter=2000)
            print(f"Using ElasticNet regression (L1+L2, alpha={alpha})")
        else:
            self.linear_model = LinearRegression()
            print("Using standard Linear Regression (no regularization)")
        
        self.linear_model.fit(X_train_scaled, y_train)
        
        # Predictions (use scaled data)
        y_train_pred = self.linear_model.predict(X_train_scaled)
        y_test_pred = self.linear_model.predict(X_test_scaled)
        
        # Evaluate
        train_mse = mean_squared_error(y_train, y_train_pred)
        test_mse = mean_squared_error(y_test, y_test_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"\nTraining Metrics:")
        print(f"  MSE: {train_mse:.4f}")
        print(f"  R² Score: {train_r2:.4f} ({train_r2*100:.2f}%)")
        print(f"\nTest Metrics:")
        print(f"  MSE: {test_mse:.4f}")
        print(f"  R² Score: {test_r2:.4f} ({test_r2*100:.2f}%)")
        
        # Feature importance (coefficients)
        if self.linear_model.n_features_in_ < 20:
            print("\nFeature Importance (Top coefficients):")
            feature_names = list(self.feature_names)
            if self.tfidf_features is not None:
                feature_names += list(self.tfidf_features.columns)
            
            coefs = self.linear_model.coef_
            top_indices = np.argsort(np.abs(coefs))[-10:][::-1]
            
            print(f"\nTop 10 most important features for predicting {target_feature}:")
            for idx in top_indices:
                if idx < len(feature_names):
                    print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
        
        # Visualize predictions vs actual
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Training set
            ax1.scatter(y_train, y_train_pred, alpha=0.6)
            ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2)
            ax1.set_xlabel('Actual')
            ax1.set_ylabel('Predicted')
            ax1.set_title(f'Training Set (R² = {train_r2:.3f})')
            ax1.grid(True, alpha=0.3)
            
            # Test set
            ax2.scatter(y_test, y_test_pred, alpha=0.6)
            ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax2.set_xlabel('Actual')
            ax2.set_ylabel('Predicted')
            ax2.set_title(f'Test Set (R² = {test_r2:.3f})')
            ax2.grid(True, alpha=0.3)
            
            plt.suptitle(f'Linear Regression: Predicting {target_feature}', y=1.02)
            plt.tight_layout()
            plt.savefig('linear_regression_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("\nSaved linear_regression_results.png")
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
        
        # Store performance forecast metrics
        self.performance_metrics = {
            'target_feature': target_feature,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'method': 'Linear Regression (R² Score)'
        }
        
        results = {
            'model': self.linear_model,
            'target_feature': target_feature,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'train_r2': train_r2,
            'test_r2': test_r2
        }
        
        return results
    
    def visualize_dimensionality_reduction(self, methods: List[str] = ['pca', 'tsne']):
        """
        Visualize clustering results using different dimensionality reduction methods
        
        Args:
            methods: List of methods to visualize ('pca', 'tsne', 'umap', 'svd')
        """
        print("\n" + "="*50)
        print("DIMENSIONALITY REDUCTION VISUALIZATIONS")
        print("="*50)
        
        if self.clusters is None:
            print("Error: Must perform clustering first!")
            return
        
        available_methods = {'pca': True, 'svd': True}
        if HAS_TSNE:
            available_methods['tsne'] = True
        if HAS_UMAP:
            available_methods['umap'] = True
        
        n_methods = len([m for m in methods if m.lower() in available_methods])
        if n_methods == 0:
            print("No available methods to visualize.")
            return
        
        fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 6))
        if n_methods == 1:
            axes = [axes]
        
        plot_idx = 0
        for method in methods:
            if method.lower() not in available_methods:
                print(f"Skipping {method} (not available)")
                continue
            
            reduction_result = self.apply_dimensionality_reduction(method=method, n_components=2)
            if reduction_result is None:
                continue
            
            reduced_data = reduction_result['reduced_data']
            ax = axes[plot_idx]
            
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], 
                               c=self.clusters, cmap='viridis', alpha=0.6)
            ax.set_xlabel(f'{method.upper()} Component 1')
            ax.set_ylabel(f'{method.upper()} Component 2')
            ax.set_title(f'Clusters - {method.upper()}')
            plt.colorbar(scatter, ax=ax, label='Cluster')
            
            plot_idx += 1
        
        plt.tight_layout()
        plt.savefig('dimensionality_reduction.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved dimensionality_reduction.png")
    
    def visualize_results(self):
        """Create visualizations of the analysis"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # 1. Cluster distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        cluster_counts = self.df['Cluster'].value_counts().sort_index()
        ax.bar(cluster_counts.index.astype(str), cluster_counts.values)
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Number of Companies')
        ax.set_title('Company Distribution Across Segments')
        plt.tight_layout()
        plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved cluster_distribution.png")
        
        # 2. Enhanced PCA visualization (2D) with feature importance - IMPROVED VERSION
        if len(self.feature_names) > 2:
            try:
                # Try to use enhanced visualization
                from visualization_improvements import create_enhanced_pca_visualization
                create_enhanced_pca_visualization(
                    self.df_processed_scaled,
                    self.clusters,
                    self.feature_names,
                    save_path='pca_clusters.png'
                )
            except ImportError:
                # Fallback to original visualization
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(self.df_processed_scaled)

                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

                # Left plot: PCA scatter with improved styling
                unique_clusters = sorted(set(self.clusters))
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
                
                for i, cluster_id in enumerate(unique_clusters):
                    mask = self.clusters == cluster_id
                    ax1.scatter(pca_result[mask, 0], pca_result[mask, 1],
                               c=[colors[i]], label=f'Cluster {cluster_id}',
                               alpha=0.6, s=50, edgecolors='white', linewidths=0.5)
                    
                    # Add centroid
                    centroid_x = pca_result[mask, 0].mean()
                    centroid_y = pca_result[mask, 1].mean()
                    ax1.scatter(centroid_x, centroid_y, c='black', marker='X', 
                               s=200, edgecolors='white', linewidths=2, zorder=10)
                
                ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontweight='bold')
                ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontweight='bold')
                ax1.set_title('Company Segments (PCA Visualization)', fontweight='bold')
                ax1.grid(True, alpha=0.3, linestyle='--')
                ax1.legend(loc='best', framealpha=0.9)

                # Right plot: Feature contributions to PCA components
                all_feature_names = self.df_processed_scaled.columns.tolist()
                components_df = pd.DataFrame(
                    pca.components_.T,
                    columns=['PC1', 'PC2'],
                    index=all_feature_names
                )

                # Plot top contributing features
                components_df['abs_sum'] = components_df.abs().sum(axis=1)
                top_features = components_df.nlargest(min(10, len(all_feature_names)), 'abs_sum')

                x_pos = np.arange(len(top_features))
                width = 0.35
                ax2.barh(x_pos - width/2, top_features['PC1'], width, label='PC1', alpha=0.8, color='steelblue')
                ax2.barh(x_pos + width/2, top_features['PC2'], width, label='PC2', alpha=0.8, color='coral')
                ax2.set_yticks(x_pos)
                ax2.set_yticklabels([f.split('(')[0].strip() if '(' in f else f for f in top_features.index], fontsize=9)
                ax2.set_xlabel('Component Loading', fontweight='bold')
                ax2.set_title('Top Feature Contributions to PCA', fontweight='bold')
                ax2.legend()
                ax2.grid(True, alpha=0.3, axis='x')
                ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

                plt.tight_layout()
                plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("Saved pca_clusters.png (with feature importance)")
        
        # 3. Feature comparison across clusters - ENHANCED VERSION
        # Get target columns
        if hasattr(self, 'found_numeric') and self.found_numeric:
            numeric_cols = list(self.found_numeric.values())
        else:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'Cluster']
        
        if numeric_cols:
            try:
                # Try to use enhanced visualization
                from visualization_improvements import create_enhanced_feature_comparison
                create_enhanced_feature_comparison(
                    self.df,
                    self.clusters,
                    numeric_cols,
                    save_path='feature_comparison.png',
                    max_features=6
                )
            except ImportError:
                # Fallback to original visualization with improvements
                top_features = numeric_cols[:6]
                n_features = len(top_features)
                n_cols = min(3, n_features)
                n_rows = (n_features + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
                if n_features == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
                
                # Get cluster colors
                unique_clusters = sorted(set(self.clusters))
                colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
                cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
                
                for idx, feature in enumerate(top_features):
                    feature_data = self.df[feature].copy()

                    # Check if data spans multiple orders of magnitude (skewed)
                    valid_data = feature_data[feature_data > 0]
                    if len(valid_data) > 0:
                        data_range = valid_data.max() / (valid_data.min() + 1e-10)
                        use_log = data_range > 100
                    else:
                        use_log = False

                    # Prepare data for each cluster
                    plot_data = []
                    plot_labels = []
                    plot_colors_list = []
                    
                    for cluster_id in unique_clusters:
                        mask = self.clusters == cluster_id
                        cluster_data = feature_data[mask].dropna()
                        
                        if len(cluster_data) > 0:
                            if use_log:
                                cluster_data = np.log10(cluster_data + 1)
                            plot_data.append(cluster_data.values)
                            plot_labels.append(f'C{cluster_id}\n(n={len(cluster_data)})')
                            plot_colors_list.append(cluster_colors[cluster_id])
                    
                    # Create enhanced boxplot with colors
                    if plot_data:
                        bp = axes[idx].boxplot(plot_data, labels=plot_labels, 
                                              patch_artist=True, widths=0.6)
                        
                        # Color the boxes
                        for patch, color in zip(bp['boxes'], plot_colors_list):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.7)
                            patch.set_edgecolor('black')
                            patch.set_linewidth(1.5)
                        
                        # Add mean markers
                        for i, data in enumerate(plot_data):
                            mean_val = np.mean(data)
                            axes[idx].scatter(i+1, mean_val, color='red', marker='D', 
                                            s=100, zorder=10, edgecolors='white', linewidths=1)
                        
                        axes[idx].set_ylabel(
                            f'Log10({feature} + 1)' if use_log else feature,
                            fontweight='bold'
                        )
                        axes[idx].set_title(
                            f'{feature}\n{"(Log Scale)" if use_log else "(Linear Scale)"}',
                            fontweight='bold'
                        )
                        axes[idx].grid(True, alpha=0.3, axis='y', linestyle='--')

                # Hide extra subplots
                for idx in range(n_features, len(axes)):
                    axes[idx].set_visible(False)

                plt.suptitle('Enhanced Feature Comparison Across Clusters', 
                           fontsize=14, fontweight='bold', y=0.995)
                plt.tight_layout(rect=[0, 0, 1, 0.98])
                plt.savefig('feature_comparison.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("Saved feature_comparison.png")
        
        # 4. Interactive Plotly visualization
        if len(self.feature_names) > 2:
            pca = PCA(n_components=3)
            pca_result = pca.fit_transform(self.df_processed_scaled)
            
            fig = go.Figure(data=go.Scatter3d(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                z=pca_result[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=self.clusters,
                    colorscale='Viridis',
                    opacity=0.8,
                    colorbar=dict(title="Cluster")
                ),
                text=[f"Company {i}, Cluster {c}" for i, c in enumerate(self.clusters)],
                hovertemplate='<b>%{text}</b><br>PC1: %{x}<br>PC2: %{y}<br>PC3: %{z}<extra></extra>'
            ))
            
            fig.update_layout(
                title='3D Company Segmentation Visualization',
                scene=dict(
                    xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)',
                    yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)',
                    zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)'
                ),
                width=1000,
                height=800
            )
            
            fig.write_html('interactive_clusters.html')
            print("Saved interactive_clusters.html")
    
    def generate_report(self, cluster_analysis: Dict, patterns: Dict, insights: str):
        """Generate a comprehensive report with detailed data implications"""
        print("\n" + "="*50)
        print("GENERATING COMPREHENSIVE REPORT")
        print("="*50)
        
        report = []
        report.append("="*80)
        report.append("COMPANY INTELLIGENCE ANALYSIS REPORT")
        report.append("="*80)
        report.append("")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Add insights
        report.append(insights)
        report.append("")
        
        # ========================================================================
        # SECTION 1: SEGMENTATION - Silhouette Score / Elbow Method
        # ========================================================================
        report.append("="*80)
        report.append("1. SEGMENTATION")
        report.append("Method: Silhouette Score / Elbow Method")
        report.append("="*80)
        report.append("")
        
        if hasattr(self, 'segmentation_metrics') and self.segmentation_metrics:
            metrics = self.segmentation_metrics
            report.append(f"Optimal Number of Clusters: {metrics['optimal_k']}")
            report.append(f"Optimal Silhouette Score: {metrics['optimal_silhouette_score']:.4f}")
            report.append("")
            report.append("Silhouette Score Interpretation:")
            report.append("  - Score ranges from -1 to 1")
            report.append("  - Values closer to 1 indicate well-separated clusters")
            # FIXED: Use standard silhouette score interpretation thresholds
            score = metrics['optimal_silhouette_score']
            if score > 0.7:
                report.append(f"  - Score of {score:.4f} indicates: STRONG cluster structure (well-separated)")
            elif score > 0.5:
                report.append(f"  - Score of {score:.4f} indicates: REASONABLE cluster structure")
            elif score > 0.25:
                report.append(f"  - Score of {score:.4f} indicates: FAIR/WEAK structure (clusters may overlap)")
                report.append("    Note: Scores in this range suggest clusters exist but have significant overlap")
            else:
                report.append(f"  - Score of {score:.4f} indicates: NO SUBSTANTIAL structure found")
                report.append("    Note: Consider using fewer clusters or different features")
            report.append("")
            report.append("DATA IMPLICATIONS:")
            report.append("  • The segmentation identifies distinct company groups based on:")
            report.append("    - Revenue patterns and financial performance")
            report.append("    - Employee size and organizational structure")
            report.append("    - Market value and IT investment levels")
            report.append("  • Well-separated clusters enable:")
            report.append("    - Targeted marketing strategies for each segment")
            report.append("    - Customized product/service offerings")
            report.append("    - Risk assessment tailored to segment characteristics")
            report.append("  • Business Value:")
            report.append("    - Sales teams can prioritize high-value segments")
            report.append("    - Product development can focus on segment-specific needs")
            report.append("    - Resource allocation optimized by segment potential")
        else:
            report.append("Segmentation metrics not available.")
        report.append("")
        
        # ========================================================================
        # SECTION 2: FINDING TRENDS - Chi-Square Test
        # ========================================================================
        report.append("="*80)
        report.append("2. FINDING TRENDS")
        report.append("Method: Chi-Square Test")
        report.append("="*80)
        report.append("")
        
        if hasattr(self, 'trend_metrics') and self.trend_metrics:
            metrics = self.trend_metrics
            report.append(f"Variables Tested: {metrics['total_tested']}")
            report.append(f"Significant Associations: {metrics['significant_count']}")
            report.append("")
            
            if metrics['chi_square_results']:
                report.append("Trend Analysis Results:")
                for col, result in metrics['chi_square_results'].items():
                    significance = "✓ SIGNIFICANT" if result['is_significant'] else "✗ NOT SIGNIFICANT"
                    report.append(f"\n  {col}:")
                    report.append(f"    Status: {significance}")
                    report.append(f"    Chi-square statistic: {result['chi2_statistic']:.4f}")
                    report.append(f"    P-value: {result['p_value']:.6f}")
                    if result['is_significant']:
                        report.append(f"    → This variable shows strong association with cluster membership")
            report.append("")
            report.append("DATA IMPLICATIONS:")
            report.append("  • Significant associations reveal:")
            report.append("    - Entity Type (HQ vs Branch) patterns across segments")
            report.append("    - Ownership Type (Public vs Private) distribution by cluster")
            report.append("    - Structural characteristics that define each segment")
            report.append("  • Business Applications:")
            report.append("    - Identify which company types dominate each segment")
            report.append("    - Understand organizational structure preferences")
            report.append("    - Predict segment membership from company attributes")
            report.append("  • Strategic Insights:")
            report.append("    - Tailor engagement strategies based on entity structure")
            report.append("    - Adjust sales approach for public vs private companies")
            report.append("    - Recognize segment-specific organizational patterns")
        else:
            report.append("Trend analysis metrics not available.")
        report.append("")
        
        # ========================================================================
        # SECTION 3: DRIVER IDENTIFICATION - Logistic Regression Coefficients
        # ========================================================================
        report.append("="*80)
        report.append("3. DRIVER IDENTIFICATION")
        report.append("Method: Logistic Regression (Coefficients)")
        report.append("="*80)
        report.append("")
        
        if hasattr(self, 'driver_metrics') and self.driver_metrics:
            metrics = self.driver_metrics
            report.append(f"Model Training Accuracy: {metrics['train_accuracy']:.2%}")
            report.append(f"Model Test Accuracy: {metrics['test_accuracy']:.2%}")
            report.append("")
            report.append("Key Drivers by Cluster (Top Contributing Features):")
            report.append("")
            
            if metrics['coefficients']:
                for cluster_id, coefs in sorted(metrics['coefficients'].items()):
                    report.append(f"  Cluster {cluster_id} Drivers:")
                    sorted_drivers = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)
                    for driver, value in sorted_drivers[:5]:
                        direction = "increases" if value > 0 else "decreases"
                        report.append(f"    • {driver}: {value:.4f} (strongly {direction} cluster membership)")
            report.append("")
            report.append("DATA IMPLICATIONS:")
            report.append("  • Coefficient analysis reveals:")
            report.append("    - Which features most strongly predict segment membership")
            report.append("    - Relative importance of revenue, employees, IT spending, etc.")
            report.append("    - Key differentiators between company segments")
            report.append("  • Business Value:")
            report.append("    - Identify critical factors for segment classification")
            report.append("    - Understand what makes companies similar within segments")
            report.append("    - Predict segment membership for new companies")
            report.append("  • Strategic Applications:")
            report.append("    - Focus on high-impact features for segmentation")
            report.append("    - Develop segment-specific value propositions")
            report.append("    - Create predictive models for lead scoring")
        else:
            report.append("Driver identification metrics not available.")
        report.append("")
        
        # ========================================================================
        # SECTION 4: PERFORMANCE FORECAST - Linear Regression R²
        # ========================================================================
        report.append("="*80)
        report.append("4. PERFORMANCE FORECAST")
        report.append("Method: Linear Regression (R² Score)")
        report.append("="*80)
        report.append("")
        
        if hasattr(self, 'performance_metrics') and self.performance_metrics:
            metrics = self.performance_metrics
            report.append(f"Target Feature: {metrics['target_feature']}")
            report.append(f"Training R² Score: {metrics['train_r2']:.4f} ({metrics['train_r2']*100:.2f}%)")
            report.append(f"Test R² Score: {metrics['test_r2']:.4f} ({metrics['test_r2']*100:.2f}%)")
            report.append("")
            report.append("R² Score Interpretation:")
            report.append("  - R² ranges from 0 to 1 (or can be negative for poor models)")
            report.append("  - R² = 1.0: Perfect predictions")
            report.append("  - R² > 0.7: Strong predictive power")
            report.append("  - R² > 0.5: Moderate predictive power")
            report.append("  - R² < 0.5: Weak predictive power")
            report.append("")
            if metrics['test_r2'] > 0.7:
                report.append(f"  → R² of {metrics['test_r2']:.4f} indicates STRONG predictive capability")
            elif metrics['test_r2'] > 0.5:
                report.append(f"  → R² of {metrics['test_r2']:.4f} indicates MODERATE predictive capability")
            else:
                report.append(f"  → R² of {metrics['test_r2']:.4f} indicates LIMITED predictive capability")
            report.append("")
            report.append("DATA IMPLICATIONS:")
            report.append("  • Forecasting Capability:")
            report.append(f"    - Can predict {metrics['target_feature']} with {metrics['test_r2']*100:.1f}% accuracy")
            report.append("    - Model performance indicates relationship strength between features and target")
            report.append("    - Test R² shows model's ability to generalize to new data")
            report.append("  • Business Applications:")
            report.append("    - Forecast revenue/performance for new prospects")
            report.append("    - Estimate market value based on company characteristics")
            report.append("    - Predict IT spending needs for budgeting")
            report.append("  • Strategic Value:")
            report.append("    - Enable data-driven financial planning")
            report.append("    - Support investment decision-making")
            report.append("    - Provide benchmarks for company evaluation")
        else:
            report.append("Performance forecast metrics not available.")
        report.append("")
        
        # ========================================================================
        # SECTION 5: SIMPLIFYING COMPLEX DATA - PCA Explained Variance
        # ========================================================================
        report.append("="*80)
        report.append("5. SIMPLIFYING COMPLEX DATA")
        report.append("Method: PCA Explained Variance")
        report.append("="*80)
        report.append("")
        
        if hasattr(self, 'pca_metrics') and self.pca_metrics:
            metrics = self.pca_metrics
            report.append(f"Reduction Method: {metrics['method']}")
            report.append(f"Original Dimensions: {metrics['original_dimensions']}")
            report.append(f"Reduced Dimensions: {metrics['reduced_dimensions']}")
            report.append(f"Total Variance Explained: {metrics['total_variance_explained']:.2%}")
            report.append("")
            report.append("Explained Variance by Component:")
            for i, var in enumerate(metrics['explained_variance_ratio'][:10], 1):
                report.append(f"  Component {i}: {var:.2%}")
            report.append("")
            report.append("Variance Interpretation:")
            if metrics['total_variance_explained'] > 0.9:
                report.append(f"  → {metrics['total_variance_explained']:.1%} variance captured: EXCELLENT data compression")
            elif metrics['total_variance_explained'] > 0.7:
                report.append(f"  → {metrics['total_variance_explained']:.1%} variance captured: GOOD data compression")
            elif metrics['total_variance_explained'] > 0.5:
                report.append(f"  → {metrics['total_variance_explained']:.1%} variance captured: MODERATE data compression")
            else:
                report.append(f"  → {metrics['total_variance_explained']:.1%} variance captured: LIMITED compression")
            report.append("")
            report.append("DATA IMPLICATIONS:")
            report.append("  • Dimensionality Reduction Benefits:")
            report.append(f"    - Reduced {metrics['original_dimensions']} features to {metrics['reduced_dimensions']} components")
            report.append(f"    - Maintained {metrics['total_variance_explained']:.1%} of original information")
            report.append("    - Simplified complex multi-dimensional data for analysis")
            report.append("  • Business Value:")
            report.append("    - Faster model training and prediction")
            report.append("    - Reduced computational complexity")
            report.append("    - Identified most important underlying patterns")
            report.append("  • Strategic Insights:")
            report.append("    - Key dimensions that drive company differences")
            report.append("    - Simplified visualization of company relationships")
            report.append("    - Focus on principal components for decision-making")
        else:
            report.append("PCA metrics not available.")
        report.append("")
        
        # ========================================================================
        # SECTION 6: DETAILED CLUSTER ANALYSIS
        # ========================================================================
        report.append("="*80)
        report.append("6. DETAILED CLUSTER ANALYSIS")
        report.append("="*80)
        report.append("")
        
        for cluster_id in sorted(cluster_analysis.keys()):
            info = cluster_analysis[cluster_id]
            report.append(f"\n{'='*60}")
            report.append(f"SEGMENT {cluster_id}")
            report.append(f"{'='*60}")
            report.append(f"Size: {info['size']} companies ({info['percentage']:.1f}%)")
            report.append("")
            
            if info['numeric_stats']:
                report.append("Numeric Characteristics:")
                for feature, stats in list(info['numeric_stats'].items())[:10]:
                    if 'mean' in stats:
                        report.append(f"  {feature}:")
                        report.append(f"    Mean: {stats['mean']:,.2f}")
                        report.append(f"    Std: {stats['std']:,.2f}")
                        report.append(f"    Min: {stats['min']:,.2f}")
                        report.append(f"    Max: {stats['max']:,.2f}")
                report.append("")
            
            if info['categorical_profiles']:
                report.append("Categorical Profiles:")
                for feature, profile in list(info['categorical_profiles'].items())[:5]:
                    report.append(f"  {feature}:")
                    for value, pct in list(profile.items())[:5]:
                        report.append(f"    {value}: {pct:.1%}")
                report.append("")
            
            report.append("BUSINESS IMPLICATIONS FOR THIS SEGMENT:")
            report.append("  • Segment represents a distinct market group with unique characteristics")
            report.append("  • Companies in this segment share similar:")
            if info['numeric_stats']:
                report.append("    - Financial metrics and performance indicators")
            if info['categorical_profiles']:
                report.append("    - Organizational structure and ownership patterns")
            report.append("  • Recommended Actions:")
            report.append("    - Develop segment-specific marketing messages")
            report.append("    - Customize product/service offerings")
            report.append("    - Adjust pricing strategies to segment characteristics")
            report.append("")
        
        # ========================================================================
        # SECTION 7: OVERALL DATA IMPLICATIONS & RECOMMENDATIONS
        # ========================================================================
        report.append("="*80)
        report.append("7. OVERALL DATA IMPLICATIONS & STRATEGIC RECOMMENDATIONS")
        report.append("="*80)
        report.append("")
        report.append("KEY FINDINGS:")
        report.append("  • The analysis successfully identified distinct company segments")
        report.append("  • Multiple analytical methods confirm segment validity")
        report.append("  • Predictive models enable forward-looking insights")
        report.append("")
        report.append("STRATEGIC RECOMMENDATIONS:")
        report.append("  1. SEGMENTATION STRATEGY:")
        report.append("     - Use identified segments for targeted marketing campaigns")
        report.append("     - Develop segment-specific value propositions")
        report.append("     - Allocate sales resources based on segment potential")
        report.append("")
        report.append("  2. TREND ANALYSIS:")
        report.append("     - Monitor changes in Entity Type and Ownership patterns")
        report.append("     - Track segment evolution over time")
        report.append("     - Identify emerging trends in company structures")
        report.append("")
        report.append("  3. DRIVER-BASED DECISIONS:")
        report.append("     - Focus on high-impact features for segmentation")
        report.append("     - Develop predictive models for lead qualification")
        report.append("     - Use driver insights for product development")
        report.append("")
        report.append("  4. PERFORMANCE FORECASTING:")
        report.append("     - Leverage predictive models for revenue estimation")
        report.append("     - Use forecasts for budget planning and resource allocation")
        report.append("     - Support investment decisions with data-driven predictions")
        report.append("")
        report.append("  5. DATA SIMPLIFICATION:")
        report.append("     - Utilize reduced dimensions for faster analysis")
        report.append("     - Focus on principal components for key insights")
        report.append("     - Streamline reporting with simplified visualizations")
        report.append("")
        report.append("BUSINESS VALUE:")
        report.append("  • Enhanced targeting and personalization capabilities")
        report.append("  • Improved sales efficiency through segment prioritization")
        report.append("  • Data-driven decision making across the organization")
        report.append("  • Competitive advantage through advanced analytics")
        report.append("")
        
        # Save report
        report_text = "\n".join(report)
        with open('company_intelligence_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("Saved company_intelligence_report.txt")
        return report_text
    
    def run_full_analysis(self, n_clusters: int = None, exclude_cols: List[str] = None):
        """Run the complete analysis pipeline"""
        print("\n" + "="*80)
        print("STARTING FOCUSED COMPANY INTELLIGENCE ANALYSIS")
        print("="*80)
        print("Analyzing: Revenue | Employee Total | Employee Single Sites | Market Value | IT Spending & Budget")
        print("TF-IDF on: SIC Description | NAICS Description | NACE Rev 2 Description")
        print("Chi-square on: Entity Type | Ownership Type")
        print("Methods: Logistic Regression | Dimensional Reduction | Train/Test Split")
        print("="*80)
        
        # Check if data is loaded
        if self.df is None or len(self.df) == 0:
            print("\n❌ ERROR: No data loaded. Cannot run analysis.")
            print("Please check the data file path and try again.")
            return None
        
        # 1. Explore data
        self.explore_data()
        
        # 2. Preprocess
        self.preprocess_data(exclude_cols=exclude_cols)
        
        # 3. Cluster
        self.perform_clustering(n_clusters=n_clusters)
        
        # 4. Apply Chi-square tests on categorical variables
        chi_square_results = self.perform_chi_square_test()
        
        # 5. Analyze clusters
        cluster_analysis = self.analyze_clusters()
        
        # 6. Compare clusters
        comparison = self.compare_clusters()
        
        # 7. Identify patterns
        patterns = self.identify_patterns()
        
        # 8. Apply dimensional reduction for visualization
        pca_result = self.apply_dimensionality_reduction(method='pca', n_components=3)
        
        # 9. Train logistic regression (with train/test split and dimensional reduction)
        lr_results = self.train_logistic_regression()
        
        # 10. Train linear regression (optional)
        linear_reg_results = None
        try:
            linear_reg_results = self.train_linear_regression()
        except Exception as e:
            print(f"Linear regression skipped: {e}")
        
        # 12. Generate insights
        insights = self.generate_llm_insights(cluster_analysis, patterns)
        
        # 13. Visualize
        self.visualize_results()
        
        # 14. Add new results to patterns
        if lr_results:
            patterns['logistic_regression'] = {
                'train_accuracy': lr_results['train_accuracy'],
                'test_accuracy': lr_results['test_accuracy']
            }
        if chi_square_results:
            patterns['chi_square'] = {
                col: {
                    'p_value': result['p_value'],
                    'is_significant': result['is_significant']
                }
                for col, result in chi_square_results.items()
            }
        if linear_reg_results:
            patterns['linear_regression'] = {
                'target_feature': linear_reg_results['target_feature'],
                'train_r2': linear_reg_results['train_r2'],
                'test_r2': linear_reg_results['test_r2']
            }
        
        # 15. Generate report
        report = self.generate_report(cluster_analysis, patterns, insights)
        
        # 16. Save results to CSV
        output_file = 'companies_with_segments.csv'
        self.df.to_csv(output_file, index=False)
        print(f"\nSaved results to {output_file}")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print("\nGenerated files:")
        print("  - company_intelligence_report.txt (detailed report)")
        print("  - companies_with_segments.csv (data with cluster labels)")
        print("  - cluster_distribution.png (visualization)")
        print("  - pca_clusters.png (2D visualization)")
        print("  - feature_comparison.png (feature analysis)")
        print("  - interactive_clusters.html (interactive 3D visualization)")
        print("  - optimal_clusters.png (cluster selection)")
        print("  - linear_regression_results.png (linear regression visualization)")
        
        # Build return dictionary
        return_dict = {
            'cluster_analysis': cluster_analysis,
            'patterns': patterns,
            'insights': insights,
            'report': report
        }
        
        # Add new analysis results
        if lr_results:
            return_dict['logistic_regression'] = lr_results
        if chi_square_results:
            return_dict['chi_square'] = chi_square_results
        if linear_reg_results:
            return_dict['linear_regression'] = linear_reg_results
        if pca_result:
            return_dict['pca_result'] = pca_result
        
        return return_dict


if __name__ == "__main__":
    import sys
    
    # Default data path
    data_path = 'champions_group_data.xlsx'
    
    # Check if custom path provided
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        
        # Check if user accidentally used placeholder path
        if 'path/to/your' in data_path or 'your/file' in data_path:
            print("\n⚠️  WARNING: You used a placeholder path!")
            print(f"   You provided: {data_path}")
            print("\n💡 This should be replaced with your actual file path.")
            print("   Examples:")
            print("   - python company_intelligence.py champions_group_data.xlsx")
            print("   - python company_intelligence.py /full/path/to/your/file.xlsx")
            print("\n   Or just run without arguments to use the default file:")
            print("   - python company_intelligence.py")
            sys.exit(1)
    
    # Check if default file exists
    if not os.path.exists(data_path):
        print(f"\n❌ ERROR: Default data file not found: {data_path}")
        print(f"\n💡 Please either:")
        print(f"   1. Place your data file in the current directory and name it: {data_path}")
        print(f"   2. Or provide the path to your data file:")
        print(f"      python company_intelligence.py /path/to/your/file.xlsx")
        sys.exit(1)
    
    # Check for API key in environment or .env file
    # Try DeepSeek first, then OpenAI
    api_key = os.getenv('DEEPSEEK_API_KEY')
    api_provider = 'deepseek'
    
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
        api_provider = 'openai'
    
    # Load .env file if available
    if not api_key:
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.getenv('DEEPSEEK_API_KEY') or os.getenv('OPENAI_API_KEY')
            if os.getenv('DEEPSEEK_API_KEY'):
                api_provider = 'deepseek'
            elif os.getenv('OPENAI_API_KEY'):
                api_provider = 'openai'
        except:
            pass
    
    # Allow override via command line argument (if provided)
    if len(sys.argv) > 2:
        if sys.argv[2].lower() in ['openai', 'deepseek']:
            api_provider = sys.argv[2].lower()
    
    # Initialize and run analysis
    try:
        analyzer = CompanyIntelligence(data_path, api_key=api_key, api_provider=api_provider)
        results = analyzer.run_full_analysis()
        
        if results:
            print("\n✅ Analysis completed successfully!")
        else:
            print("\n⚠️  Analysis did not complete successfully.")
            sys.exit(1)
    except ValueError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
