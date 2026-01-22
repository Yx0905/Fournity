"""
AI-Driven Company Intelligence System
Analyzes company data to generate actionable insights through data-driven segmentation
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix, accuracy_score
from scipy.stats import chi2_contingency
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
    
    def remove_outliers_iqr(self, df: pd.DataFrame, columns: List[str], multiplier: float = 1.5) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove outliers using the Interquartile Range (IQR) method.

        Args:
            df: DataFrame to clean
            columns: List of column names to check for outliers
            multiplier: IQR multiplier (default 1.5 for standard outlier detection)

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
            'percentage_removed': 0
        }

        # Track rows to remove (union of all outliers across columns)
        outlier_indices = set()

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
            outlier_indices.update(col_outliers)

            # Store report data
            outlier_report['columns_processed'].append(col)
            outlier_report['outliers_by_column'][col] = {
                'count': len(col_outliers),
                'percentage': (len(col_outliers) / len(df)) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR
            }

        # Remove all outlier rows
        if outlier_indices:
            df_clean = df_clean.drop(index=list(outlier_indices))
            outlier_report['total_rows_removed'] = len(outlier_indices)
            outlier_report['total_rows_after'] = len(df_clean)
            outlier_report['percentage_removed'] = (len(outlier_indices) / len(df)) * 100
        else:
            outlier_report['total_rows_after'] = len(df)

        return df_clean, outlier_report

    def preprocess_data(self, exclude_cols: List[str] = None):
        """
        Preprocess data for clustering - COMPREHENSIVE FEATURE SET:

        Numeric Features:
        - Core: Revenue, Market Value, Employee Total, Employee Single Sites
        - Technology: IT Budget, IT Spend, PCs, Servers, Storage, Routers, Laptops, Desktops
        - Maturity: Company Age (derived from Year Founded)

        Text Features (TF-IDF):
        - SIC Description, NAICS Description, NACE Rev 2 Description

        Categorical Features (Chi-square):
        - Entity Type, Ownership Type, Manufacturing Status, Import/Export Status, Franchise Status

        Processing Steps:
        1. Feature engineering (create company age)
        2. Missing value imputation (median)
        3. Outlier removal (IQR method)
        4. Feature scaling (StandardScaler)
        5. TF-IDF vectorization for text
        6. Feature combination

        Args:
            exclude_cols: List of column names to exclude from analysis (not used in focused mode)
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
            current_year = 2026  # Update this as needed
            if year_col in df_processed.columns:
                # Create age column
                age_col = 'company_age'
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
                        if pd.isna(val):
                            return np.nan
                        val_str = str(val).strip()
                        if ' to ' in val_str.lower():
                            parts = val_str.lower().split(' to ')
                            try:
                                return (float(parts[0]) + float(parts[1])) / 2
                            except:
                                return np.nan
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
            df_processed, outlier_report = self.remove_outliers_iqr(df_processed, numeric_cols_to_clean)
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

        # Store numeric features (without duplicates)
        self.feature_names = feature_cols.copy()
        self.df_processed = df_processed[feature_cols].copy()
        
        # Scale numeric features
        self.df_processed_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df_processed),
            columns=self.df_processed.columns,
            index=self.df_processed.index
        )
        
        print(f"\n✓ Processed {len(feature_cols)} numeric features for analysis")
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
        
        return self.df_processed_scaled
    
    def determine_optimal_clusters(self, max_k: int = 10):
        """
        Determine optimal number of clusters using enhanced business-focused method
        SEGMENTATION: Balanced Silhouette Score + Business Practicality

        The algorithm considers:
        1. Silhouette scores (cluster quality)
        2. Business practicality (K=2 is too simplistic, K>9 is too complex)
        3. Preference for K=5-7 range (standard market segmentation)

        Args:
            max_k: Maximum number of clusters to test
        """
        print("\n" + "="*50)
        print("SEGMENTATION: DETERMINING OPTIMAL CLUSTERS")
        print("Method: Enhanced Business-Focused Clustering")
        print("="*50)

        inertias = []
        silhouette_scores = []
        max_possible_k = min(max_k + 1, len(self.df_processed_scaled) // 2, len(self.df_processed_scaled) - 1)
        if max_possible_k < 2:
            print("Warning: Not enough data points for clustering. Using k=2.")
            return 2

        k_range = range(2, max_possible_k + 1)

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.df_processed_scaled)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.df_processed_scaled, labels))
            print(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")

        # Enhanced optimal k selection logic
        # Prioritize K=5-7 if silhouette scores are reasonable (>0.30)
        # This provides actionable business segmentation instead of oversimplified K=2

        best_k_by_silhouette = k_range[np.argmax(silhouette_scores)]
        max_silhouette = max(silhouette_scores)

        # Look for best K in the practical range (4-8)
        practical_range = [k for k in k_range if 4 <= k <= 8]
        if practical_range:
            practical_scores = [(k, silhouette_scores[k-2]) for k in practical_range]
            best_practical = max(practical_scores, key=lambda x: x[1])

            # If practical K has reasonable score (within 15% of max), prefer it
            if best_practical[1] >= max_silhouette * 0.70:
                optimal_k = best_practical[0]
                optimal_silhouette = best_practical[1]
                print(f"\nOptimal number of clusters: {optimal_k} (business-optimized)")
                print(f"Silhouette Score: {optimal_silhouette:.4f}")
                print(f"Note: K={best_k_by_silhouette} has highest silhouette ({max_silhouette:.4f}),")
                print(f"      but K={optimal_k} provides better business segmentation")
            else:
                optimal_k = best_k_by_silhouette
                optimal_silhouette = max_silhouette
                print(f"\nOptimal number of clusters: {optimal_k} (silhouette-based)")
                print(f"Silhouette Score: {optimal_silhouette:.4f}")
        else:
            optimal_k = best_k_by_silhouette
            optimal_silhouette = max_silhouette
            print(f"\nOptimal number of clusters: {optimal_k} (silhouette-based)")
            print(f"Silhouette Score: {optimal_silhouette:.4f}")
        
        # Store segmentation metrics
        self.segmentation_metrics = {
            'optimal_k': optimal_k,
            'optimal_silhouette_score': optimal_silhouette,
            'silhouette_scores': dict(zip(k_range, silhouette_scores)),
            'inertias': dict(zip(k_range, inertias)),
            'method': 'Silhouette Score / Elbow Method'
        }
        
        # Plot elbow curve
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.plot(k_range, inertias, 'bo-')
        ax1.set_xlabel('Number of Clusters (k)')
        ax1.set_ylabel('Inertia')
        ax1.set_title('Elbow Method')
        ax1.grid(True)
        
        ax2.plot(k_range, silhouette_scores, 'ro-')
        ax2.set_xlabel('Number of Clusters (k)')
        ax2.set_ylabel('Silhouette Score')
        ax2.set_title('Silhouette Score')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('optimal_clusters.png', dpi=300, bbox_inches='tight')
        print("\nSaved optimal_clusters.png")
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
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(self.df)) * 100,
                'numeric_stats': {},
                'categorical_profiles': {}
            }
            
            # Analyze numeric target features
            if numeric_cols:
                cluster_analysis[cluster_id]['numeric_stats'] = cluster_data[numeric_cols].describe().to_dict()
            
            # Analyze categorical features
            for cat_col in categorical_cols:
                value_counts = cluster_data[cat_col].value_counts(normalize=True)
                cluster_analysis[cluster_id]['categorical_profiles'][cat_col] = value_counts.to_dict()
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {cluster_analysis[cluster_id]['size']} companies ({cluster_analysis[cluster_id]['percentage']:.1f}%)")
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
                
                patterns['cluster_differences'][cluster_id] = differences
        
        return patterns
    
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
        
        # Prepare summary for LLM
        summary = {
            'total_companies': len(self.df),
            'num_clusters': len(set(self.clusters)),
            'cluster_sizes': {str(k): v['size'] for k, v in cluster_analysis.items()},
            'key_patterns': patterns
        }
        
        prompt = f"""You are a business intelligence analyst. Analyze the following company segmentation results and provide actionable insights.

Dataset Summary:
- Total companies: {summary['total_companies']}
- Number of segments: {summary['num_clusters']}
- Segment sizes: {summary['cluster_sizes']}

Key Patterns Identified:
{json.dumps(patterns, indent=2, default=str)}

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
                temperature=0.7,
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
                columns=[f'TFIDF_{i}' for i in range(tfidf_matrix.shape[1])],
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
    
    def perform_chi_square_test(self, categorical_cols: List[str] = None):
        """
        Perform Chi-square test of independence between categorical variables (Entity Type, Ownership Type) and clusters
        FINDING TRENDS: Chi-Square Test
        
        Args:
            categorical_cols: List of categorical column names to test (defaults to Entity Type and Ownership Type)
        """
        print("\n" + "="*50)
        print("FINDING TRENDS: CHI-SQUARE TEST ANALYSIS")
        print("Method: Chi-Square Test of Independence")
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
        
        for col in categorical_cols:
            try:
                # Create contingency table
                contingency = pd.crosstab(self.df[col], self.df['Cluster'])
                
                # Check if table is valid (at least 2x2)
                if contingency.shape[0] < 2 or contingency.shape[1] < 2:
                    print(f"  Skipping {col}: insufficient categories for chi-square test")
                    continue
                
                # Perform chi-square test
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                
                chi_square_results[col] = {
                    'chi2_statistic': chi2,
                    'p_value': p_value,
                    'degrees_of_freedom': dof,
                    'is_significant': p_value < 0.05,
                    'contingency_table': contingency
                }
                
                significance = "SIGNIFICANT" if p_value < 0.05 else "NOT SIGNIFICANT"
                print(f"\n{col}:")
                print(f"  Chi-square statistic: {chi2:.4f}")
                print(f"  P-value: {p_value:.6f}")
                print(f"  Degrees of freedom: {dof}")
                print(f"  Result: {significance} (α=0.05)")
                
            except Exception as e:
                print(f"Error testing {col}: {e}")
                continue
        
        # Summary
        if chi_square_results:
            significant_vars = [col for col, result in chi_square_results.items() 
                              if result['is_significant']]
            print(f"\n{'='*50}")
            print(f"Summary: {len(significant_vars)}/{len(chi_square_results)} categorical variables "
                  f"show significant association with clusters")
            
            # Store trend finding metrics
            self.trend_metrics = {
                'chi_square_results': chi_square_results,
                'significant_variables': significant_vars,
                'total_tested': len(chi_square_results),
                'significant_count': len(significant_vars),
                'method': 'Chi-Square Test'
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
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
            reduced_data = reducer.fit_transform(self.df_processed_scaled)
            print(f"t-SNE: Completed (perplexity=30)")
            
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
    
    def train_logistic_regression(self, test_size: float = 0.2, random_state: int = 42):
        """
        Train logistic regression model to predict cluster membership using:
        - Numeric features: Revenue, Employee Total, Employee Single Sites, Market Value, IT Spending & Budget
        - TF-IDF features: From SIC, NAICS, NACE descriptions
        - Dimensional reduction applied to combined features
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        print("\n" + "="*50)
        print("LOGISTIC REGRESSION - CLUSTER PREDICTION")
        print("="*50)
        print("Using: Numeric features + TF-IDF features + Dimensional reduction")
        
        if self.clusters is None:
            print("Error: Must perform clustering first!")
            return None
        
        if self.df_processed_scaled is None:
            print("Error: Must preprocess data first!")
            return None
        
        # Apply dimensional reduction to combined features
        print("\nApplying dimensional reduction to features...")
        reduction_result = self.apply_dimensionality_reduction(method='pca', n_components=min(50, self.df_processed_scaled.shape[1]))
        
        if reduction_result and 'reduced_data' in reduction_result:
            X = reduction_result['reduced_data']
            print(f"Reduced features from {self.df_processed_scaled.shape[1]} to {X.shape[1]} dimensions")
            print(f"Explained variance: {sum(reduction_result['explained_variance']):.2%}")
        else:
            # Fallback to original features if reduction fails
            X = self.df_processed_scaled.values
            print("Using original features (dimensional reduction not applied)")
        
        y = self.clusters
        
        # Split data with train/test split
        # Check if stratification is possible (each class needs at least 2 samples)
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
        
        print(f"\nTraining set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        print(f"Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        
        # Train logistic regression
        # Use solver='lbfgs' which supports multinomial by default for multi-class
        try:
            # Try with multi_class parameter (for newer scikit-learn versions)
            self.logistic_model = LogisticRegression(
                multi_class='multinomial',
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state
            )
        except TypeError:
            # Fallback for older scikit-learn versions that don't support multi_class parameter
            # 'lbfgs' solver automatically handles multinomial for multi-class problems
            self.logistic_model = LogisticRegression(
                solver='lbfgs',
                max_iter=1000,
                random_state=random_state
            )
        
        self.logistic_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.logistic_model.predict(X_train)
        y_test_pred = self.logistic_model.predict(X_test)
        
        # Evaluate
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
        
        # Classification report
        print("\nClassification Report (Test Set):")
        print(classification_report(y_test, y_test_pred))
        
        # Feature importance (coefficients) - for reduced dimensions
        driver_coefficients = {}
        if hasattr(self.logistic_model, 'n_features_in_') and self.logistic_model.n_features_in_ < 50:
            print("\nDRIVER IDENTIFICATION: Top contributing features per cluster:")
            for i, cluster_id in enumerate(sorted(set(y))):
                print(f"\nCluster {cluster_id}:")
                coefs = self.logistic_model.coef_[i]
                top_indices = np.argsort(np.abs(coefs))[-5:][::-1]
                driver_coefficients[cluster_id] = {}
                for idx in top_indices:
                    driver_coefficients[cluster_id][f'Component_{idx}'] = float(coefs[idx])
                    print(f"  Component {idx}: {coefs[idx]:.4f}")
        
        # Store driver identification metrics
        self.driver_metrics = {
            'coefficients': driver_coefficients,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'method': 'Logistic Regression (Coefficients)'
        }
        
        results = {
            'model': self.logistic_model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'dimensional_reduction': reduction_result,
            'coefficients': driver_coefficients
        }
        
        return results
    
    def train_linear_regression(self, target_feature: str = None, test_size: float = 0.2, random_state: int = 42):
        """
        Train linear regression model to predict a TARGET FEATURE (Revenue or Market Value)
        
        Args:
            target_feature: Name of the target feature to predict ('revenue' or 'market_value')
                          If None, uses 'revenue' as default
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        print("\n" + "="*50)
        print("PERFORMANCE FORECAST: LINEAR REGRESSION")
        print("Method: Linear Regression (R² Score)")
        print("="*50)
        
        if self.df_processed_scaled is None:
            print("Error: Must preprocess data first!")
            return None
        
        # Get target columns
        if not hasattr(self, 'found_numeric') or not self.found_numeric:
            print("Warning: Target columns not found. Re-running preprocessing...")
            self.preprocess_data()
        
        # Select target feature from target columns
        if target_feature is None:
            # Default to revenue
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
            # Find the actual column name
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
        
        # Prepare features and target
        X = self.df_processed_scaled.values
        y = self.df[target_feature].values
        
        # Remove rows with missing target values
        valid_mask = ~np.isnan(y)
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) < 10:
            print(f"Error: Not enough valid samples ({len(X)}) for linear regression!")
            return None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Target feature: {target_feature}")
        
        # Train linear regression
        self.linear_model = LinearRegression()
        self.linear_model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = self.linear_model.predict(X_train)
        y_test_pred = self.linear_model.predict(X_test)
        
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
        
        # 2. Enhanced PCA visualization (2D) with feature importance
        if len(self.feature_names) > 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(self.df_processed_scaled)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

            # Left plot: PCA scatter
            scatter = ax1.scatter(pca_result[:, 0], pca_result[:, 1],
                               c=self.clusters, cmap='viridis', alpha=0.6, s=50)
            ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax1.set_title('Company Segments (PCA Visualization)')
            ax1.grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=ax1, label='Cluster')

            # Right plot: Feature contributions to PCA components
            # Get all feature names (numeric + TF-IDF if combined)
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
            ax2.barh(x_pos - width/2, top_features['PC1'], width, label='PC1', alpha=0.8)
            ax2.barh(x_pos + width/2, top_features['PC2'], width, label='PC2', alpha=0.8)
            ax2.set_yticks(x_pos)
            ax2.set_yticklabels([f.split('(')[0].strip() if '(' in f else f for f in top_features.index], fontsize=9)
            ax2.set_xlabel('Component Loading')
            ax2.set_title('Top Feature Contributions to PCA')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='x')
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

            plt.tight_layout()
            plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved pca_clusters.png (with feature importance)")
        
        # 3. Feature comparison across clusters - TARGET FEATURES ONLY
        # Get target columns
        if hasattr(self, 'found_numeric') and self.found_numeric:
            numeric_cols = list(self.found_numeric.values())
        else:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [col for col in numeric_cols if col != 'Cluster']
        
        if numeric_cols:
            # Use all target features (up to 6)
            top_features = numeric_cols[:6]
            n_features = len(top_features)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 4*n_rows))
            if n_features == 1:
                axes = [axes]
            else:
                axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
            
            for idx, feature in enumerate(top_features):
                self.df.boxplot(column=feature, by='Cluster', ax=axes[idx])
                axes[idx].set_title(f'{feature} by Cluster')
                axes[idx].set_xlabel('Cluster')
            
            # Hide extra subplots
            for idx in range(n_features, len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Target Feature Comparison Across Clusters', y=1.02)
            plt.tight_layout()
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
            if metrics['optimal_silhouette_score'] > 0.5:
                report.append(f"  - Score of {metrics['optimal_silhouette_score']:.4f} indicates: EXCELLENT cluster separation and cohesion")
            elif metrics['optimal_silhouette_score'] > 0.3:
                report.append(f"  - Score of {metrics['optimal_silhouette_score']:.4f} indicates: GOOD cluster structure with reasonable separation")
            elif metrics['optimal_silhouette_score'] > 0.1:
                report.append(f"  - Score of {metrics['optimal_silhouette_score']:.4f} indicates: FAIR clustering with some overlap between segments")
            else:
                report.append(f"  - Score of {metrics['optimal_silhouette_score']:.4f} indicates: WEAK clustering - segments may not be well-defined")
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
