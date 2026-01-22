#!/usr/bin/env python3
"""
Latent-Sparse Clustering Analysis for CHAMPIONS GROUP Dataset
Implements a comprehensive 4-phase clustering workflow with interpretability
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    from prince import FAMD
    FAMD_AVAILABLE = True
except ImportError:
    FAMD_AVAILABLE = False
    print("Warning: prince (FAMD) not available. Install with: pip install prince")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: umap-learn not available. Install with: pip install umap-learn")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: shap not available. Install with: pip install shap")

try:
    from scipy.spatial.distance import cdist
    from scipy import stats
    THEIL_AVAILABLE = True
except ImportError:
    THEIL_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")

try:
    from sklearn_extra.cluster import KMedoids
    KMEDOIDS_AVAILABLE = True
except ImportError:
    KMEDOIDS_AVAILABLE = False
    print("Warning: sklearn_extra (K-Medoids) not available. Install with: pip install scikit-learn-extra")

try:
    import gower
    GOWER_AVAILABLE = True
except ImportError:
    GOWER_AVAILABLE = False
    print("Warning: gower not available. Install with: pip install gower")

try:
    from sklearn.mixture import GaussianMixture
    from sklearn.cluster import AgglomerativeClustering
    GMM_AVAILABLE = True
except ImportError:
    GMM_AVAILABLE = False
    print("Warning: sklearn clustering modules not fully available")

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("Warning: hdbscan not available. Install with: pip install hdbscan")

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Install with: pip install plotly")

# Set style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


class LatentSparseClustering:
    """
    Implements the Latent-Sparse Clustering workflow
    """
    
    def __init__(self, data_file='champions_group_processed.xlsx', strict_validation_gate=False, use_enhanced_pipeline=True):
        self.data_file = data_file
        self.strict_validation_gate = strict_validation_gate
        self.use_enhanced_pipeline = use_enhanced_pipeline
        self.df = None
        self.df_processed = None
        self.feature_info = {}
        self.results = {}
        self.cluster_labels = None
        self.best_model = None
        self.best_params = {}
        self.personas = None
        
    def phase1_context_analysis(self):
        """
        Phase 1: Context & Meta-Data Analysis
        """
        print("="*80)
        print("PHASE 1: CONTEXT & META-DATA ANALYSIS")
        print("="*80)
        
        # Load data
        print("\nStep 1.1: Feature Profiling...")
        self.df = pd.read_excel(self.data_file)
        print(f"Dataset shape: {self.df.shape}")
        
        # Select relevant columns for clustering
        clustering_cols = [
            'Revenue (USD)', 'Market Value (USD)', 'IT Spend', 'Employees Total',
            'Company Sites', 'No. of Servers', 'No. of Storage Devices',
            'Revenue per Employee', 'Market Value per Employee', 'IT Spend Ratio',
            'Employees per Site', 'Technology Density Index', 'Company Age',
            'Geographic Dispersion'
        ]
        
        # Filter to available columns
        available_cols = [col for col in clustering_cols if col in self.df.columns]
        self.df_work = self.df[available_cols].copy()
        
        # Add categorical columns if available
        categorical_cols = ['Country', 'Region', 'Parent Country/Region']
        for col in categorical_cols:
            if col in self.df.columns:
                self.df_work[col] = self.df[col]
        
        print(f"Selected {len(available_cols)} numerical features for clustering")
        
        # Feature profiling
        numerical_cols = self.df_work.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df_work.select_dtypes(include=['object', 'category']).columns.tolist()
        
        self.feature_info['numerical'] = numerical_cols
        self.feature_info['categorical'] = categorical_cols
        self.feature_info['n_samples'] = len(self.df_work)
        self.feature_info['n_features'] = len(numerical_cols) + len(categorical_cols)
        
        print(f"\nNumerical features: {len(numerical_cols)}")
        print(f"Categorical features: {len(categorical_cols)}")
        
        # Calculate cardinality for categoricals
        for col in categorical_cols:
            cardinality = self.df_work[col].nunique()
            print(f"  {col}: {cardinality} unique values")
            self.feature_info[f'{col}_cardinality'] = cardinality
        
        # Step 1.2: Sparsity Check
        print("\nStep 1.2: Sparsity Check...")
        missing_pct = (self.df_work.isnull().sum() / len(self.df_work)) * 100
        high_missing = missing_pct[missing_pct > 20]
        
        if len(high_missing) > 0:
            print("Features with >20% missing values:")
            for col, pct in high_missing.items():
                print(f"  {col}: {pct:.2f}%")
                # Decision: drop if >50% missing, otherwise impute
                if pct > 50:
                    print(f"    -> Dropping {col} (>50% missing)")
                    self.df_work = self.df_work.drop(columns=[col])
                    if col in numerical_cols:
                        numerical_cols.remove(col)
                    if col in categorical_cols:
                        categorical_cols.remove(col)
                else:
                    print(f"    -> Will impute {col}")
                    if col in numerical_cols:
                        self.df_work[col] = self.df_work[col].fillna(self.df_work[col].median())
                    else:
                        self.df_work[col] = self.df_work[col].fillna(self.df_work[col].mode()[0] if len(self.df_work[col].mode()) > 0 else 'Unknown')
        else:
            print("No features with >20% missing values")
        
        # Fill remaining missing values
        for col in numerical_cols:
            if col in self.df_work.columns and self.df_work[col].isnull().any():
                median_val = self.df_work[col].median()
                if pd.notna(median_val):
                    self.df_work[col] = self.df_work[col].fillna(median_val)
                else:
                    # If all values are NaN, fill with 0
                    self.df_work[col] = self.df_work[col].fillna(0)
        
        for col in categorical_cols:
            if col in self.df_work.columns and self.df_work[col].isnull().any():
                mode_val = self.df_work[col].mode()[0] if len(self.df_work[col].mode()) > 0 else 'Unknown'
                self.df_work[col] = self.df_work[col].fillna(mode_val)
        
        # Final check - drop any rows that still have NaN in critical columns
        critical_cols = [col for col in numerical_cols if col in self.df_work.columns]
        if len(critical_cols) > 0:
            rows_before = len(self.df_work)
            self.df_work = self.df_work.dropna(subset=critical_cols)
            rows_after = len(self.df_work)
            if rows_before != rows_after:
                print(f"  -> Dropped {rows_before - rows_after} rows with NaN in critical columns")
                self.feature_info['n_samples'] = len(self.df_work)
        
        # Step 1.3: Metric Mapping
        print("\nStep 1.3: Metric Mapping...")
        total_features = len(numerical_cols) + len(categorical_cols)
        cat_ratio = len(categorical_cols) / total_features if total_features > 0 else 0
        num_ratio = len(numerical_cols) / total_features if total_features > 0 else 0
        
        print(f"Categorical ratio: {cat_ratio:.2%}")
        print(f"Numerical ratio: {num_ratio:.2%}")
        
        if cat_ratio > 0.5:
            self.feature_info['distance_metric'] = 'Gower'
            self.feature_info['use_famd'] = True
            self.feature_info['use_gower_kmedoids'] = True
            print("-> Selected: Gower Distance + K-Medoids (when deps available)")
        elif num_ratio > 0.8:
            self.feature_info['distance_metric'] = 'Euclidean/Cosine'
            self.feature_info['use_famd'] = False
            self.feature_info['use_gower_kmedoids'] = False
            print("-> Selected: Euclidean/Cosine Distance")
        else:
            self.feature_info['distance_metric'] = 'Mixed'
            self.feature_info['use_famd'] = True
            self.feature_info['use_gower_kmedoids'] = False
            print("-> Selected: Mixed (FAMD preprocessing)")
        
        self.feature_info['numerical_cols'] = numerical_cols
        self.feature_info['categorical_cols'] = categorical_cols
        
        return self.df_work
    
    def phase2_filtering_encoding(self):
        """
        Phase 2: Filtering & Encoding Engine
        """
        print("\n" + "="*80)
        print("PHASE 2: FILTERING & ENCODING ENGINE")
        print("="*80)
        
        numerical_cols = self.feature_info['numerical_cols']
        categorical_cols = self.feature_info['categorical_cols']
        
        # Step 2.1: Redundancy Pruning
        print("\nStep 2.1: Redundancy Pruning...")
        
        # Filter to only columns that exist in df_work
        numerical_cols = [col for col in numerical_cols if col in self.df_work.columns]
        categorical_cols = [col for col in categorical_cols if col in self.df_work.columns]
        
        # Correlation matrix for numericals
        if len(numerical_cols) > 1:
            corr_matrix = self.df_work[numerical_cols].corr().abs()
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > 0.9:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
            
            if high_corr_pairs:
                print("Highly correlated numerical pairs (|ρ| > 0.9):")
                cols_to_drop = set()
                for col1, col2, corr_val in high_corr_pairs:
                    print(f"  {col1} <-> {col2}: {corr_val:.3f}")
                    # Drop the one with more missing values or lower variance
                    if self.df_work[col1].var() < self.df_work[col2].var():
                        cols_to_drop.add(col1)
                    else:
                        cols_to_drop.add(col2)
                
                for col in cols_to_drop:
                    if col in self.df_work.columns:
                        print(f"  -> Dropping {col}")
                        if col in numerical_cols:
                            numerical_cols.remove(col)
                        self.df_work = self.df_work.drop(columns=[col])
            else:
                print("No highly correlated numerical pairs found")
        
        # Theil's U for categoricals (simplified - using Cramér's V as proxy)
        if len(categorical_cols) > 1 and THEIL_AVAILABLE:
            print("\nChecking categorical redundancy...")
            # Simplified approach: drop low cardinality categoricals if they're redundant
            for col in categorical_cols:
                if col in self.df_work.columns and self.df_work[col].nunique() == 1:
                    print(f"  -> Dropping {col} (single value)")
                    categorical_cols.remove(col)
                    self.df_work = self.df_work.drop(columns=[col])
        
        # Update feature_info with cleaned column lists
        self.feature_info['numerical_cols'] = numerical_cols
        self.feature_info['categorical_cols'] = categorical_cols
        
        # Step 2.2: Automated Scaling
        print("\nStep 2.2: Automated Scaling...")
        
        # Check for outliers using IQR method
        has_outliers = False
        for col in numerical_cols:
            Q1 = self.df_work[col].quantile(0.25)
            Q3 = self.df_work[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((self.df_work[col] < (Q1 - 1.5 * IQR)) | (self.df_work[col] > (Q3 + 1.5 * IQR))).sum()
            if outliers > len(self.df_work) * 0.05:  # More than 5% outliers
                has_outliers = True
                break
        
        if has_outliers:
            print("Outliers detected -> Using RobustScaler")
            scaler = RobustScaler()
            self.scaler = scaler
        else:
            print("No significant outliers -> Using StandardScaler")
            scaler = StandardScaler()
            self.scaler = scaler
        
        # Scale numerical features
        self.df_scaled = self.df_work.copy()
        if len(numerical_cols) > 0:
            # Filter to only existing columns
            available_num_for_scale = [col for col in numerical_cols if col in self.df_work.columns]
            if len(available_num_for_scale) > 0:
                self.df_scaled[available_num_for_scale] = scaler.fit_transform(self.df_work[available_num_for_scale])
        
        # Step 2.3: Dimensionality Reduction
        print("\nStep 2.3: Dimensionality Reduction...")
        
        n_samples = len(self.df_scaled)
        n_features = len(numerical_cols) + len(categorical_cols)
        
        # Decision logic
        use_famd = self.feature_info.get('use_famd', False) and (len(categorical_cols) > 0)
        is_mixed = len(categorical_cols) > 0 and len(numerical_cols) > 0
        n_less_than_p = n_samples < n_features
        
        if use_famd and FAMD_AVAILABLE and is_mixed:
            print("Using FAMD (Factor Analysis of Mixed Data)...")
            
            # Filter to only columns that actually exist in df_work
            available_num_cols = [col for col in numerical_cols if col in self.df_work.columns]
            available_cat_cols = [col for col in categorical_cols if col in self.df_work.columns]
            
            print(f"  Preparing FAMD with {len(available_num_cols)} numerical and {len(available_cat_cols)} categorical columns")
            
            # Prepare data for FAMD
            famd_data = self.df_work[available_num_cols + available_cat_cols].copy()
            
            # Check initial NaN count
            nan_count_before = famd_data.isnull().sum().sum()
            if nan_count_before > 0:
                print(f"  Found {nan_count_before} NaN values before cleaning")
            
            # Ensure no NaN values remain
            # Fill numerical NaN with median (or 0 if all NaN)
            for col in available_num_cols:
                if col in famd_data.columns:
                    if famd_data[col].isnull().any():
                        median_val = famd_data[col].median()
                        if pd.isna(median_val) or pd.isnull(median_val):
                            # All values are NaN, fill with 0
                            print(f"    Warning: {col} has all NaN, filling with 0")
                            famd_data[col] = famd_data[col].fillna(0)
                        else:
                            famd_data[col] = famd_data[col].fillna(median_val)
            
            # Fill categorical NaN with mode or 'Unknown'
            for col in available_cat_cols:
                if col in famd_data.columns:
                    if famd_data[col].isnull().any():
                        mode_vals = famd_data[col].mode()
                        if len(mode_vals) > 0:
                            mode_val = mode_vals[0]
                        else:
                            mode_val = 'Unknown'
                        famd_data[col] = famd_data[col].fillna(mode_val)
            
            # Check for infinite values in numerical columns
            for col in available_num_cols:
                if col in famd_data.columns:
                    if np.isinf(famd_data[col]).any():
                        print(f"    Warning: {col} contains infinite values, replacing with NaN then median")
                        famd_data[col] = famd_data[col].replace([np.inf, -np.inf], np.nan)
                        median_val = famd_data[col].median()
                        if pd.isna(median_val):
                            famd_data[col] = famd_data[col].fillna(0)
                        else:
                            famd_data[col] = famd_data[col].fillna(median_val)
            
            # Double check for any remaining NaN
            nan_count_after = famd_data.isnull().sum().sum()
            if nan_count_after > 0:
                print(f"  Warning: Still have {nan_count_after} NaN values after filling. Dropping rows...")
                print("  Columns with NaN:")
                nan_cols = famd_data.isnull().sum()[famd_data.isnull().sum() > 0]
                for col, count in nan_cols.items():
                    print(f"    {col}: {count} NaN")
                
                rows_before = len(famd_data)
                famd_data = famd_data.dropna()
                rows_after = len(famd_data)
                print(f"  -> Dropped {rows_before - rows_after} rows with NaN")
                
                # Update df_work and df_scaled to match
                self.df_work = self.df_work.loc[famd_data.index].copy()
                self.df_scaled = self.df_scaled.loc[famd_data.index].copy()
            
            # Final verification - check for NaN and infinite values
            if famd_data.isnull().any().any():
                print("ERROR: NaN values still present after all cleaning!")
                print("Columns with NaN:")
                print(famd_data.isnull().sum()[famd_data.isnull().sum() > 0])
                # Force fill any remaining NaN
                print("Force filling remaining NaN...")
                for col in famd_data.columns:
                    if famd_data[col].isnull().any():
                        if pd.api.types.is_numeric_dtype(famd_data[col]):
                            famd_data[col] = famd_data[col].fillna(0)
                        else:
                            famd_data[col] = famd_data[col].fillna('Unknown')
            
            # Check for infinite values one more time
            num_cols = famd_data.select_dtypes(include=[np.number]).columns
            if len(num_cols) > 0:
                inf_mask = np.isinf(famd_data[num_cols]).any()
                if inf_mask.any():
                    print("Warning: Infinite values found, replacing...")
                    for col in num_cols:
                        if np.isinf(famd_data[col]).any():
                            famd_data[col] = famd_data[col].replace([np.inf, -np.inf], 0)
            
            # Final check
            if famd_data.isnull().any().any():
                raise ValueError("Cannot proceed - NaN values persist after all cleaning")
            
            print(f"  ✓ FAMD data ready: {len(famd_data)} rows, {len(famd_data.columns)} columns, no NaN, no Inf")

            # Update sample count if rows were dropped
            if len(famd_data) != self.feature_info['n_samples']:
                self.feature_info['n_samples'] = len(famd_data)
                print(f"  Updated sample count: {len(famd_data)}")

            # Remove zero-variance numerical columns (causes NaN during FAMD internal standardization)
            num_cols_in_famd = [col for col in available_num_cols if col in famd_data.columns]
            zero_var_cols = []
            for col in num_cols_in_famd:
                if famd_data[col].std() == 0 or famd_data[col].nunique() <= 1:
                    zero_var_cols.append(col)
            if zero_var_cols:
                print(f"  Removing {len(zero_var_cols)} zero-variance numerical columns: {zero_var_cols}")
                famd_data = famd_data.drop(columns=zero_var_cols)
                available_num_cols = [c for c in available_num_cols if c not in zero_var_cols]

            # Remove single-category categorical columns (causes issues in FAMD)
            cat_cols_in_famd = [col for col in available_cat_cols if col in famd_data.columns]
            single_cat_cols = []
            for col in cat_cols_in_famd:
                if famd_data[col].nunique() <= 1:
                    single_cat_cols.append(col)
            if single_cat_cols:
                print(f"  Removing {len(single_cat_cols)} single-category categorical columns: {single_cat_cols}")
                famd_data = famd_data.drop(columns=single_cat_cols)
                available_cat_cols = [c for c in available_cat_cols if c not in single_cat_cols]

            # Verify we still have columns to work with
            if len(famd_data.columns) < 2:
                raise ValueError("Not enough columns remaining after removing zero-variance columns")

            n_components_famd = max(1, min(20, len(available_num_cols) + len(available_cat_cols) - 1))
            print(f"  Fitting FAMD with {n_components_famd} components...")

            try:
                famd = FAMD(n_components=n_components_famd)
                famd.fit(famd_data)
                self.df_reduced = pd.DataFrame(
                    famd.transform(famd_data),
                    index=famd_data.index,
                    columns=[f'FAMD_{i+1}' for i in range(famd.n_components)]
                )
                self.reducer = famd

                # Keep components explaining 80-90% variance
                cumsum_var = np.cumsum(famd.explained_inertia_)
                n_components = np.where(cumsum_var >= 0.85)[0][0] + 1
                print(f"Variance explained by {n_components} components: {cumsum_var[n_components-1]:.2%}")
                self.df_reduced = self.df_reduced.iloc[:, :n_components]
            except (ValueError, Exception) as e:
                print(f"  FAMD failed with error: {e}")
                print("  Falling back to PCA on numerical features only...")
                use_famd = False  # Force fallback to PCA below

        if not use_famd or not hasattr(self, 'df_reduced') or self.df_reduced is None:
            print("Using PCA...")
            # Use only numerical features for PCA
            # Filter to only existing columns
            available_num_for_pca = [col for col in numerical_cols if col in self.df_scaled.columns]
            if len(available_num_for_pca) == 0:
                raise ValueError("No numerical columns available for PCA")
            
            # Ensure no NaN in scaled data
            pca_data = self.df_scaled[available_num_for_pca].copy()
            if pca_data.isnull().any().any():
                print("Warning: NaN in scaled data. Filling with 0...")
                pca_data = pca_data.fillna(0)
            
            # Check for infinite values
            if np.isinf(pca_data.values).any():
                print("Warning: Infinite values in PCA data. Replacing with 0...")
                pca_data = pca_data.replace([np.inf, -np.inf], 0)
            
            pca = PCA(n_components=min(20, len(available_num_for_pca)))
            self.df_reduced = pd.DataFrame(
                pca.fit_transform(pca_data),
                index=pca_data.index,
                columns=[f'PC_{i+1}' for i in range(pca.n_components_)]
            )
            self.reducer = pca
            
            # Keep components explaining 80-90% variance
            cumsum_var = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.where(cumsum_var >= 0.85)[0][0] + 1
            print(f"Variance explained by {n_components} components: {cumsum_var[n_components-1]:.2%}")
            self.df_reduced = self.df_reduced.iloc[:, :n_components]
        
        self.feature_info['n_components'] = n_components
        print(f"Reduced from {n_features} features to {n_components} components")
        
        # Step 2.4: Gower distance matrix (when >50% categorical → K-Medoids path)
        use_gower = self.feature_info.get('use_gower_kmedoids', False)
        if use_gower and GOWER_AVAILABLE and KMEDOIDS_AVAILABLE:
            print("\nStep 2.4: Computing Gower distance matrix (for K-Medoids)...")
            num_cols = [c for c in self.feature_info['numerical_cols'] if c in self.df_work.columns]
            cat_cols = [c for c in self.feature_info['categorical_cols'] if c in self.df_work.columns]
            gower_data = self.df_work[num_cols + cat_cols].copy()
            self.gower_matrix = gower.gower_matrix(gower_data)
            print(f"  Gower matrix shape: {self.gower_matrix.shape}")
        else:
            self.gower_matrix = None
        
        return self.df_reduced
    
    def phase3_clustering_loop(self):
        """
        Phase 3: Iterative Clustering Loop
        """
        print("\n" + "="*80)
        print("PHASE 3: ITERATIVE CLUSTERING LOOP")
        print("="*80)
        
        n_samples = len(self.df_reduced)
        use_gower = self.feature_info.get('use_gower_kmedoids', False) and getattr(self, 'gower_matrix', None) is not None
        
        # Decision logic: Gower + K-Medoids when >50% categorical; else K-Means
        if use_gower:
            print("Using K-Medoids with Gower distance (>50% categorical)")
            self._use_kmedoids = True
            cluster_class = None
        else:
            self._use_kmedoids = False
            is_pure_numerical = len(self.feature_info['categorical_cols']) == 0
            n_large = n_samples > 10000
            if is_pure_numerical and n_large:
                print("Using MiniBatch K-Means (large dataset, pure numerical)")
                cluster_class = MiniBatchKMeans
            else:
                print("Using Standard K-Means")
                cluster_class = KMeans
        
        # Step 3.1: Hyperparameter Search
        print("\nStep 3.1: Hyperparameter Search...")
        max_k = min(15, n_samples // 100)
        k_range = range(2, max(3, max_k))
        
        best_score = -1
        best_params = None
        results_grid = []
        
        if self._use_kmedoids:
            param_grid = [{'n_clusters': k, 'random_state': 42} for k in k_range]
            print(f"Testing {len(param_grid)} values of k (K-Medoids + Gower)...")
            for params in param_grid:
                try:
                    model = KMedoids(metric='precomputed', **params)
                    labels = model.fit_predict(self.gower_matrix)
                    if len(np.unique(labels)) < 2:
                        continue
                    sil_score = silhouette_score(self.gower_matrix, labels, metric='precomputed')
                    results_grid.append({
                        'params': params,
                        'silhouette': sil_score,
                        'calinski_harabasz': np.nan,
                        'n_clusters': params['n_clusters']
                    })
                    if sil_score > best_score:
                        best_score = sil_score
                        best_params = params.copy()
                        self.best_model = model
                        self.cluster_labels = labels
                except Exception as e:
                    print(f"  Error with k={params['n_clusters']}: {e}")
                    continue
        else:
            param_grid = {
                'n_clusters': list(k_range),
                'random_state': [42],
                'n_init': [10]
            }
            if cluster_class == MiniBatchKMeans:
                param_grid['batch_size'] = [256, 512]
            print(f"Testing {len(list(ParameterGrid(param_grid)))} parameter combinations...")
            for params in ParameterGrid(param_grid):
                try:
                    model = cluster_class(**params)
                    labels = model.fit_predict(self.df_reduced)
                    if len(np.unique(labels)) < 2:
                        continue
                    sil_score = silhouette_score(self.df_reduced, labels)
                    ch_score = calinski_harabasz_score(self.df_reduced, labels)
                    results_grid.append({
                        'params': params,
                        'silhouette': sil_score,
                        'calinski_harabasz': ch_score,
                        'n_clusters': params['n_clusters']
                    })
                    combined_score = 0.6 * sil_score + 0.4 * (ch_score / 1000)
                    if combined_score > best_score:
                        best_score = combined_score
                        best_params = params.copy()
                        self.best_model = model
                        self.cluster_labels = labels
                except Exception as e:
                    print(f"  Error with params {params}: {e}")
                    continue
        
        if best_params is None:
            raise ValueError("No valid clustering solution found")
        
        print(f"\nBest parameters: {best_params}")
        best_result = next(r for r in results_grid if r['params'] == best_params)
        print(f"Best silhouette score: {best_result['silhouette']:.3f}")
        self.best_params = best_params
        self.results['grid_results'] = results_grid
        
        # Step 3.3: Stability Testing (5-fold shuffle for K-Means; 5-seed ARI for K-Medoids)
        print("\nStep 3.3: Stability Testing (5-fold)...")
        n_folds = 5
        
        if self._use_kmedoids:
            # K-Medoids: run same k five times with different random_state; use ARI vs reference
            reference_labels = self.cluster_labels.copy()
            ari_scores = []
            for fold in range(n_folds):
                rs = 42 + fold
                try:
                    m = KMedoids(metric='precomputed', n_clusters=best_params['n_clusters'], random_state=rs)
                    labels_fold = m.fit_predict(self.gower_matrix)
                    ari = adjusted_rand_score(reference_labels, labels_fold)
                    ari_scores.append(ari)
                except Exception:
                    ari_scores.append(0.0)
            mean_ari = np.mean(ari_scores)
            stability_threshold = 0.85
            is_stable = mean_ari >= stability_threshold
            print(f"Mean ARI across 5 seeds: {mean_ari:.4f} (threshold {stability_threshold})")
            if not is_stable:
                print("WARNING: K-Medoids may be unstable (low label consistency across seeds)")
            else:
                print("Model stability: PASSED")
            self.results['stability'] = {'mean_ari': mean_ari, 'is_stable': is_stable}
        else:
            # K-Means: run same k five times with different shuffles; check centroid stability
            centers_original = self.best_model.cluster_centers_
            stability_scores = []
            for fold in range(n_folds):
                np.random.seed(42 + fold)
                indices = np.random.permutation(len(self.df_reduced))
                df_shuffled = self.df_reduced.iloc[indices].reset_index(drop=True)
                fold_params = best_params.copy()
                model_fold = cluster_class(**fold_params)
                model_fold.fit_predict(df_shuffled)
                centers_fold = model_fold.cluster_centers_
                if len(centers_fold) == len(centers_original):
                    dists = cdist(centers_fold, centers_original)
                    shift = np.mean([np.min(dists[i, :]) for i in range(len(centers_fold))])
                    stability_scores.append(shift)
            avg_shift = np.mean(stability_scores) if stability_scores else 0.0
            stability_threshold = 0.1
            is_stable = avg_shift <= stability_threshold
            print(f"Average cluster center shift: {avg_shift:.4f}")
            if avg_shift > stability_threshold:
                print(f"WARNING: Model may be unstable (shift > {stability_threshold})")
            else:
                print("Model stability: PASSED")
            self.results['stability'] = {'avg_shift': avg_shift, 'is_stable': is_stable}
        
        # Validation Gate: explicit check; halt if strict and Silhouette < 0.5
        sil_score = best_result['silhouette']
        print(f"\nValidation Gate: Silhouette Score = {sil_score:.3f}")
        if sil_score > 0.5:
            print("✓ Validation PASSED - Proceeding to Profiling")
            self.results['validation_passed'] = True
        else:
            print("✗ Validation FAILED - Weak cluster separation (Silhouette < 0.5)")
            print("  Consider re-running with Sparse Autoencoder or different feature set.")
            self.results['validation_passed'] = False
            if self.strict_validation_gate:
                raise ValueError(
                    "Strict validation gate enabled: clustering quality insufficient "
                    f"(Silhouette={sil_score:.3f} < 0.5). Refuse to proceed."
                )
        
        return self.cluster_labels
    
    def phase4_interpretability(self):
        """
        Phase 4: Interpretability & "The Why"
        """
        print("\n" + "="*80)
        print("PHASE 4: INTERPRETABILITY & 'THE WHY'")
        print("="*80)
        
        # Step 4.1: Feature Importance Extraction
        print("\nStep 4.1: Feature Importance Extraction...")
        
        # Use Random Forest to predict cluster labels
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.df_reduced, self.cluster_labels)
        
        feature_importance = pd.DataFrame({
            'feature': self.df_reduced.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Top 5 most important features for clustering:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        self.results['feature_importance'] = feature_importance
        
        # Step 4.2: Cluster Profiling
        print("\nStep 4.2: Cluster Profiling...")
        
        # Add cluster labels to original data
        self.df_work['Cluster'] = self.cluster_labels
        
        # Get top features from original dataset
        top_features = feature_importance.head(5)['feature'].tolist()
        # Map back to original features if possible
        original_features = []
        for feat in top_features:
            # Try to find corresponding original feature
            if 'FAMD' in feat or 'PC' in feat:
                # For reduced dimensions, use all original numerical features
                original_features.extend(self.feature_info['numerical_cols'][:5])
            else:
                original_features.append(feat)
        
        # Remove duplicates while preserving order
        original_features = list(dict.fromkeys(original_features))[:5]
        original_features = [f for f in original_features if f in self.df_work.columns]
        
        cluster_profiles = []
        for cluster_id in sorted(np.unique(self.cluster_labels)):
            cluster_data = self.df_work[self.df_work['Cluster'] == cluster_id]
            profile = {'Cluster': cluster_id, 'Size': len(cluster_data)}
            
            for feat in original_features:
                if feat in self.df_work.columns:
                    if self.df_work[feat].dtype in [np.number]:
                        profile[f'{feat}_mean'] = cluster_data[feat].mean()
                        profile[f'{feat}_median'] = cluster_data[feat].median()
                    else:
                        profile[f'{feat}_mode'] = cluster_data[feat].mode()[0] if len(cluster_data[feat].mode()) > 0 else 'N/A'
            
            cluster_profiles.append(profile)
        
        self.cluster_profiles_df = pd.DataFrame(cluster_profiles)
        print("\nCluster Profiles:")
        print(self.cluster_profiles_df.to_string())
        
        # Step 4.3: Visual Synthesis
        print("\nStep 4.3: Visual Synthesis...")
        
        if UMAP_AVAILABLE:
            print("Generating UMAP projection...")
            reducer_umap = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer_umap.fit_transform(self.df_reduced)
            
            plt.figure(figsize=(12, 8))
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                                c=self.cluster_labels, cmap='tab10', 
                                alpha=0.6, s=50)
            plt.colorbar(scatter, label='Cluster')
            plt.title('UMAP Projection of Clusters', fontsize=16, fontweight='bold')
            plt.xlabel('UMAP Dimension 1', fontsize=12)
            plt.ylabel('UMAP Dimension 2', fontsize=12)
            plt.tight_layout()
            plt.savefig('umap_clusters.png', dpi=300, bbox_inches='tight')
            print("  -> Saved: umap_clusters.png")
            plt.close()
        else:
            print("UMAP not available - skipping visualization")
        
        # Additional visualizations
        self._create_visualizations()
        
        return self.cluster_profiles_df
    
    def _create_visualizations(self):
        """Create additional visualizations"""
        print("\nCreating additional visualizations...")
        
        # 1. Cluster size distribution
        plt.figure(figsize=(10, 6))
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        plt.bar(cluster_counts.index, cluster_counts.values, color=sns.color_palette("husl", len(cluster_counts)))
        plt.xlabel('Cluster ID', fontsize=12)
        plt.ylabel('Number of Companies', fontsize=12)
        plt.title('Cluster Size Distribution', fontsize=16, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig('cluster_sizes.png', dpi=300, bbox_inches='tight')
        print("  -> Saved: cluster_sizes.png")
        plt.close()
        
        # 2. Feature importance plot
        plt.figure(figsize=(10, 8))
        top_10 = self.results['feature_importance'].head(10)
        plt.barh(range(len(top_10)), top_10['importance'].values, color=sns.color_palette("husl", len(top_10)))
        plt.yticks(range(len(top_10)), top_10['feature'].values)
        plt.xlabel('Importance Score', fontsize=12)
        plt.title('Top 10 Feature Importance for Clustering', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("  -> Saved: feature_importance.png")
        plt.close()
        
        # 3. Cluster profiles heatmap (for numerical features)
        numerical_cols = self.feature_info['numerical_cols']
        if len(numerical_cols) > 0:
            # Select top numerical features
            top_num_features = numerical_cols[:8]  # Top 8 numerical features
            
            cluster_means = []
            for cluster_id in sorted(np.unique(self.cluster_labels)):
                cluster_data = self.df_work[self.df_work['Cluster'] == cluster_id]
                means = [cluster_data[feat].mean() for feat in top_num_features]
                cluster_means.append(means)
            
            cluster_means_df = pd.DataFrame(cluster_means, 
                                          index=[f'Cluster {i}' for i in sorted(np.unique(self.cluster_labels))],
                                          columns=top_num_features)
            
            # Normalize for better visualization
            cluster_means_df_norm = cluster_means_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() != x.min() else x, axis=0)
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(cluster_means_df_norm, annot=True, fmt='.2f', cmap='YlOrRd', 
                       cbar_kws={'label': 'Normalized Value'})
            plt.title('Cluster Profiles Heatmap (Normalized)', fontsize=16, fontweight='bold')
            plt.ylabel('Cluster', fontsize=12)
            plt.xlabel('Feature', fontsize=12)
            plt.tight_layout()
            plt.savefig('cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
            print("  -> Saved: cluster_profiles_heatmap.png")
            plt.close()
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        print("\n" + "="*80)
        print("GENERATING SUMMARY REPORT")
        print("="*80)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("LATENT-SPARSE CLUSTERING ANALYSIS SUMMARY REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Phase 1 Summary
        report_lines.append("PHASE 1: CONTEXT & META-DATA ANALYSIS")
        report_lines.append("-"*80)
        report_lines.append(f"Total Samples: {self.feature_info['n_samples']}")
        report_lines.append(f"Total Features: {self.feature_info['n_features']}")
        report_lines.append(f"  - Numerical: {len(self.feature_info['numerical_cols'])}")
        report_lines.append(f"  - Categorical: {len(self.feature_info['categorical_cols'])}")
        report_lines.append(f"Distance Metric Selected: {self.feature_info['distance_metric']}")
        report_lines.append("")
        
        # Phase 2 Summary
        report_lines.append("PHASE 2: FILTERING & ENCODING ENGINE")
        report_lines.append("-"*80)
        report_lines.append(f"Features after redundancy pruning: {len(self.feature_info['numerical_cols']) + len(self.feature_info['categorical_cols'])}")
        report_lines.append(f"Reduced dimensions: {self.feature_info['n_components']} components")
        report_lines.append(f"Scaler used: {type(self.scaler).__name__}")
        report_lines.append("")
        
        # Phase 3 Summary
        report_lines.append("PHASE 3: ITERATIVE CLUSTERING LOOP")
        report_lines.append("-"*80)
        report_lines.append(f"Best Algorithm: {type(self.best_model).__name__}")
        report_lines.append(f"Best Parameters: {self.best_params}")
        
        # Get validation scores
        best_result = next((r for r in self.results.get('grid_results', []) if r['params'] == self.best_params), None)
        if best_result:
            report_lines.append(f"Silhouette Score: {best_result['silhouette']:.4f}")
            ch = best_result.get('calinski_harabasz')
            if ch is not None and not (isinstance(ch, float) and np.isnan(ch)):
                report_lines.append(f"Calinski-Harabasz Score: {ch:.4f}")
            else:
                report_lines.append("Calinski-Harabasz Score: N/A (K-Medoids + Gower)")
        
        report_lines.append(f"Stability Check: {'PASSED' if self.results.get('stability', {}).get('is_stable', False) else 'FAILED'}")
        if 'stability' in self.results:
            s = self.results['stability']
            if 'avg_shift' in s:
                report_lines.append(f"  Average cluster center shift: {s['avg_shift']:.4f}")
            if 'mean_ari' in s:
                report_lines.append(f"  Mean ARI (5 seeds): {s['mean_ari']:.4f}")
        report_lines.append(f"Validation Gate: {'PASSED' if self.results.get('validation_passed', False) else 'FAILED'}")
        report_lines.append("")
        
        # Phase 4 Summary
        report_lines.append("PHASE 4: INTERPRETABILITY & 'THE WHY'")
        report_lines.append("-"*80)
        report_lines.append("Top 5 Most Important Features:")
        for idx, row in self.results['feature_importance'].head(5).iterrows():
            report_lines.append(f"  {row['feature']}: {row['importance']:.4f}")
        report_lines.append("")
        
        # Cluster Profiles
        report_lines.append("CLUSTER PROFILES")
        report_lines.append("-"*80)
        report_lines.append(self.cluster_profiles_df.to_string())
        report_lines.append("")
        
        # Cluster Statistics
        report_lines.append("CLUSTER STATISTICS")
        report_lines.append("-"*80)
        cluster_counts = pd.Series(self.cluster_labels).value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            pct = (count / len(self.cluster_labels)) * 100
            report_lines.append(f"Cluster {cluster_id}: {count} companies ({pct:.2f}%)")
        report_lines.append("")
        
        # Key Insights
        report_lines.append("KEY INSIGHTS")
        report_lines.append("-"*80)
        
        # Find most distinct clusters
        if len(self.cluster_profiles_df) > 1:
            report_lines.append(f"Identified {len(np.unique(self.cluster_labels))} distinct company clusters")
            report_lines.append("")
            report_lines.append("Cluster Characteristics:")
            
            numerical_cols = self.feature_info['numerical_cols']
            if len(numerical_cols) > 0:
                key_feature = numerical_cols[0]  # Use first numerical feature as example
                if key_feature in self.df_work.columns:
                    for cluster_id in sorted(np.unique(self.cluster_labels)):
                        cluster_data = self.df_work[self.df_work['Cluster'] == cluster_id]
                        report_lines.append(f"  Cluster {cluster_id}:")
                        report_lines.append(f"    - Size: {len(cluster_data)} companies")
                        if key_feature in cluster_data.columns:
                            report_lines.append(f"    - Avg {key_feature}: {cluster_data[key_feature].mean():.2f}")
        
        report_lines.append("")
        report_lines.append("="*80)
        report_lines.append("END OF REPORT")
        report_lines.append("="*80)
        
        # Save report
        report_text = "\n".join(report_lines)
        with open('clustering_analysis_report.txt', 'w') as f:
            f.write(report_text)
        
        print("Summary report saved to: clustering_analysis_report.txt")
        print("\n" + report_text)
        
        return report_text
    
    def engineer_business_features(self, df):
        """
        PHASE 1 (Enhanced): Domain-Driven Feature Engineering - STRICT ORDER
        
        Pipeline Order (prevents data leakage):
        1. Handle Zeros → Replace with NaN
        2. Calculate Ratios
        3. Impute with Median
        4. Log Transform skewed variables
        """
        print("\n" + "="*80)
        print("PHASE 1 (ENHANCED): DOMAIN-DRIVEN FEATURE ENGINEERING")
        print("="*80)
        
        df = df.copy()
        
        # === STEP 1: HANDLE ZEROS (Replace with NaN to prevent division artifacts) ===
        print("\nStep 1.1: Handling zeros in base variables...")
        zero_cols = ['Revenue (USD)', 'Employees Total', 'IT Spend', 'Market Value (USD)', 
                     'No. of Servers', 'No. of Storage Devices', 'Company Sites']
        for col in zero_cols:
            if col in df.columns:
                zero_count = (df[col] == 0).sum()
                if zero_count > 0:
                    print(f"  {col}: {zero_count} zeros → NaN")
                    df[col] = df[col].replace(0, np.nan)
        
        # === STEP 2: CALCULATE RATIOS (on clean data) ===
        print("\nStep 1.2: Calculating business ratios...")
        
        # Core efficiency metrics
        if 'Revenue (USD)' in df.columns and 'Employees Total' in df.columns:
            df['Revenue_per_Emp'] = df['Revenue (USD)'] / df['Employees Total']
            print(f"  Revenue_per_Emp: {df['Revenue_per_Emp'].notna().sum()} valid values")
        
        if 'IT Spend' in df.columns and 'Revenue (USD)' in df.columns:
            df['IT_Spend_Ratio'] = df['IT Spend'] / df['Revenue (USD)']
            print(f"  IT_Spend_Ratio: {df['IT_Spend_Ratio'].notna().sum()} valid values")
        
        # Tech intensity (log-transformed for normalization)
        if 'IT Spend' in df.columns and 'Employees Total' in df.columns:
            raw_intensity = df['IT Spend'] / df['Employees Total']
            df['Tech_Intensity_Log'] = np.log1p(raw_intensity.fillna(0))
            print(f"  Tech_Intensity_Log: {df['Tech_Intensity_Log'].notna().sum()} valid values")
        
        # Additional ratios
        if 'Market Value (USD)' in df.columns and 'Employees Total' in df.columns:
            df['market_value_per_employee'] = df['Market Value (USD)'] / df['Employees Total']
        
        if 'Revenue (USD)' in df.columns and 'Company Sites' in df.columns:
            df['revenue_per_site'] = df['Revenue (USD)'] / df['Company Sites']
        
        if 'IT Spend' in df.columns and 'Employees Total' in df.columns:
            df['it_spend_per_employee'] = df['IT Spend'] / df['Employees Total']
        
        if 'No. of Servers' in df.columns and 'No. of Storage Devices' in df.columns and 'Employees Total' in df.columns:
            df['technology_density'] = (df['No. of Servers'].fillna(0) + df['No. of Storage Devices'].fillna(0)) / df['Employees Total']
        
        if 'No. of Servers' in df.columns and 'No. of Storage Devices' in df.columns:
            df['server_storage_ratio'] = df['No. of Servers'] / df['No. of Storage Devices']
        
        if 'Employees Total' in df.columns and 'Company Sites' in df.columns:
            df['employees_per_site'] = df['Employees Total'] / df['Company Sites']
        
        if 'Market Value (USD)' in df.columns and 'Revenue (USD)' in df.columns:
            df['market_to_revenue_ratio'] = df['Market Value (USD)'] / df['Revenue (USD)']
        
        if 'Market Value (USD)' in df.columns and 'Employees Total' in df.columns:
            df['capital_intensity'] = df['Market Value (USD)'] / df['Employees Total']
        
        # === STEP 3: MEDIAN IMPUTATION (for ratio features) ===
        print("\nStep 1.3: Applying median imputation to ratios...")
        ratio_cols = ['Revenue_per_Emp', 'IT_Spend_Ratio', 'Tech_Intensity_Log', 
                      'market_value_per_employee', 'revenue_per_site', 'it_spend_per_employee',
                      'technology_density', 'server_storage_ratio', 'employees_per_site',
                      'market_to_revenue_ratio', 'capital_intensity']
        
        for col in ratio_cols:
            if col in df.columns:
                nan_count = df[col].isna().sum()
                if nan_count > 0:
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col] = df[col].fillna(median_val)
                        print(f"  {col}: {nan_count} NaN → median ({median_val:.4f})")
        
        # === STEP 4: LOG TRANSFORM SKEWED VARIABLES ===
        print("\nStep 1.4: Log-transforming skewed variables...")
        skewed_cols = ['Revenue (USD)', 'Market Value (USD)', 'IT Spend', 'Employees Total']
        for col in skewed_cols:
            if col in df.columns:
                log_col = f'{col.replace(" ", "_").replace("(", "").replace(")", "")}_log'
                df[log_col] = np.log1p(df[col].fillna(0))
                print(f"  {col} → {log_col}")
        
        # === STEP 5: CATEGORICAL & STRUCTURAL FEATURES ===
        print("\nStep 1.5: Creating categorical features...")
        
        if 'Company Sites' in df.columns:
            df['is_multi_site'] = (df['Company Sites'] > 1).astype(int)
        
        if 'Company Age' in df.columns:
            try:
                df['age_bracket'] = pd.cut(df['Company Age'], 
                                          bins=[0, 5, 15, 30, 100, np.inf], 
                                          labels=['Startup', 'Growth', 'Mature', 'Established', 'Legacy'],
                                          include_lowest=True)
            except:
                pass
        
        if 'Employees Total' in df.columns:
            try:
                df['size_bracket'] = pd.cut(df['Employees Total'], 
                                           bins=[0, 10, 50, 250, 1000, 10000, np.inf],
                                           labels=['Micro', 'Small', 'Medium', 'Large', 'Enterprise', 'Mega-Corp'],
                                           include_lowest=True)
            except:
                pass
        
        # === STEP 6: GEOGRAPHIC ENCODING ===
        print("\nStep 1.6: Creating geographic features...")
        if 'Country' in df.columns:
            developed_markets = ['USA', 'UK', 'Germany', 'France', 'Japan', 'Singapore', 
                                'United States', 'United Kingdom', 'Canada', 'Australia']
            df['is_developed_market'] = df['Country'].isin(developed_markets).astype(int)
            try:
                df['region_encoded'] = pd.Categorical(df['Country']).codes
            except:
                pass
        
        # === STEP 7: INDUSTRY Z-SCORES (if available) ===
        industry_col = 'Industry' if 'Industry' in df.columns else ('Sector' if 'Sector' in df.columns else None)
        if industry_col:
            print(f"\nStep 1.7: Creating industry-relative z-scores (grouping by {industry_col})...")
            for metric in ['IT_Spend_Ratio', 'Revenue_per_Emp', 'technology_density']:
                if metric in df.columns:
                    try:
                        df[f'{metric}_industry_zscore'] = df.groupby(industry_col)[metric].transform(
                            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0
                        )
                    except:
                        pass
        
        # Count new features
        new_feature_count = len([c for c in df.columns if any(x in c.lower() for x in 
                                ['_per_', '_ratio', '_zscore', '_log', '_bracket', 'is_', 'intensity'])])
        print(f"\n✅ Created {new_feature_count} engineered features (strict pipeline)")
        return df
    
    def calculate_vif(self, X, feature_names, threshold=10.0):
        """
        Calculate Variance Inflation Factor (VIF) for multicollinearity detection.
        Remove features with VIF > threshold.
        """
        from sklearn.linear_model import LinearRegression
        
        print("\n📊 Calculating Variance Inflation Factor (VIF)...")
        
        X_clean = pd.DataFrame(X, columns=feature_names).fillna(0).replace([np.inf, -np.inf], 0)
        
        vif_data = []
        features_to_keep = list(feature_names)
        
        # Iteratively remove high-VIF features
        iteration = 0
        max_iterations = 10
        
        while iteration < max_iterations:
            iteration += 1
            vif_scores = []
            
            for i, col in enumerate(features_to_keep):
                if len(features_to_keep) < 2:
                    break
                    
                y = X_clean[col].values
                X_other = X_clean[[c for c in features_to_keep if c != col]].values
                
                try:
                    lr = LinearRegression()
                    lr.fit(X_other, y)
                    r_squared = lr.score(X_other, y)
                    vif = 1 / (1 - r_squared) if r_squared < 1 else np.inf
                except:
                    vif = 1.0
                
                vif_scores.append({'feature': col, 'VIF': vif})
            
            vif_df = pd.DataFrame(vif_scores).sort_values('VIF', ascending=False)
            
            # Check if any VIF > threshold
            max_vif = vif_df['VIF'].max()
            if max_vif <= threshold:
                print(f"  ✅ All features have VIF <= {threshold}")
                break
            
            # Remove feature with highest VIF
            worst_feature = vif_df.iloc[0]['feature']
            print(f"  ⚠️ Removing {worst_feature} (VIF={max_vif:.2f} > {threshold})")
            features_to_keep.remove(worst_feature)
            X_clean = X_clean[features_to_keep]
        
        removed = set(feature_names) - set(features_to_keep)
        if removed:
            print(f"  Removed {len(removed)} high-VIF features: {list(removed)[:5]}...")
        
        return features_to_keep, X_clean.values
    
    def hierarchical_segmentation(self, X_scaled, min_cluster_size=100):
        """
        PHASE 2 (Enhanced): Hierarchical Multi-Level Clustering
        Two-stage clustering: Macro segments → Micro sub-segments
        Solves the 82% mega-cluster problem.
        """
        print("\n" + "="*80)
        print("PHASE 2 (ENHANCED): HIERARCHICAL MULTI-LEVEL CLUSTERING")
        print("="*80)
        
        if not GMM_AVAILABLE:
            print("⚠️ GaussianMixture not available, falling back to standard K-Means")
            return None, None
        
        # === STAGE 1: MACRO SEGMENTATION (5-7 broad groups) ===
        print("\nStage 1: Creating macro business segments with Gaussian Mixture Models...")
        
        best_gmm = None
        best_bic = np.inf
        best_n_components = 5
        
        for n_components in range(5, 8):
            try:
                gmm = GaussianMixture(
                    n_components=n_components,
                    covariance_type='full',
                    n_init=10,
                    random_state=42,
                    max_iter=100
                )
                gmm.fit(X_scaled)
                bic = gmm.bic(X_scaled)
                
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    best_n_components = n_components
            except Exception as e:
                print(f"  Error with n_components={n_components}: {e}")
                continue
        
        if best_gmm is None:
            print("⚠️ GMM failed, using K-Means fallback")
            kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
            macro_labels = kmeans.fit_predict(X_scaled)
        else:
            macro_labels = best_gmm.predict(X_scaled)
        
        print(f"Created {len(np.unique(macro_labels))} macro segments")
        macro_sizes = pd.Series(macro_labels).value_counts().sort_index()
        print("Macro segment sizes:", macro_sizes.values.tolist())
        
        # === STAGE 2: SUB-SEGMENTATION within each macro cluster ===
        print("\nStage 2: Creating sub-segments within each macro cluster...")
        
        if not HDBSCAN_AVAILABLE:
            print("⚠️ HDBSCAN not available, using K-Means for sub-segmentation")
            final_labels = np.zeros(len(X_scaled), dtype=int)
            label_counter = 0
            
            for macro_id in np.unique(macro_labels):
                mask = macro_labels == macro_id
                X_subset = X_scaled[mask]
                
                if X_subset.shape[0] < min_cluster_size:
                    final_labels[mask] = label_counter
                    label_counter += 1
                    continue
                
                n_sub = max(2, min(5, X_subset.shape[0] // min_cluster_size))
                kmeans_sub = KMeans(n_clusters=n_sub, random_state=42, n_init=10)
                sub_labels = kmeans_sub.fit_predict(X_subset)
                
                unique_sub = np.unique(sub_labels)
                for sub_id in unique_sub:
                    sub_mask = sub_labels == sub_id
                    global_mask = np.where(mask)[0][sub_mask]
                    final_labels[global_mask] = label_counter
                    label_counter += 1
        else:
            final_labels = np.zeros(len(X_scaled), dtype=int)
            label_counter = 0
            
            for macro_id in np.unique(macro_labels):
                mask = macro_labels == macro_id
                X_subset = X_scaled[mask]
                
                if X_subset.shape[0] < min_cluster_size:
                    final_labels[mask] = label_counter
                    label_counter += 1
                    continue
                
                try:
                    clusterer = hdbscan.HDBSCAN(
                        min_cluster_size=max(50, int(X_subset.shape[0] * 0.05)),
                        min_samples=10,
                        cluster_selection_epsilon=0.5,
                        metric='euclidean'
                    )
                    sub_labels = clusterer.fit_predict(X_subset)
                    
                    unique_sub = np.unique(sub_labels)
                    for sub_id in unique_sub:
                        if sub_id == -1:  # Noise points
                            continue
                        sub_mask = sub_labels == sub_id
                        global_mask = np.where(mask)[0][sub_mask]
                        final_labels[global_mask] = label_counter
                        label_counter += 1
                    
                    # Handle noise points
                    noise_mask = sub_labels == -1
                    if noise_mask.any():
                        global_noise = np.where(mask)[0][noise_mask]
                        final_labels[global_noise] = label_counter
                        label_counter += 1
                except Exception as e:
                    print(f"  Error in HDBSCAN for macro {macro_id}: {e}")
                    final_labels[mask] = label_counter
                    label_counter += 1
        
        print(f"\nFinal segmentation: {len(np.unique(final_labels))} total segments")
        final_sizes = pd.Series(final_labels).value_counts().sort_index()
        print("Segment size distribution:", final_sizes.values.tolist())
        
        # === VALIDATION: Check for balance ===
        largest_pct = final_sizes.max() / len(final_labels) * 100
        print(f"\nLargest cluster: {largest_pct:.1f}% of dataset")
        
        if largest_pct > 40:
            print(f"⚠️ WARNING: Largest cluster is {largest_pct:.1f}% - still imbalanced!")
            print("Consider adjusting min_cluster_size or using constrained clustering")
        else:
            print(f"✅ Balance check passed - largest cluster is {largest_pct:.1f}%")
        
        return final_labels, macro_labels
    
    def ensemble_clustering(self, X, n_clusters=8, n_iterations=20):
        """
        PHASE 3 (Enhanced): Ensemble Clustering for Stability
        Run multiple clustering algorithms and bootstrap iterations.
        Take consensus to create stable clusters.
        """
        print("\n" + "="*80)
        print("PHASE 3 (ENHANCED): ENSEMBLE CLUSTERING FOR STABILITY")
        print("="*80)
        
        all_labels = []
        
        print(f"Running ensemble clustering with {n_iterations} iterations...")
        
        # Method 1: K-Means with different initializations
        print("  Method 1: K-Means variations...")
        for i in range(n_iterations // 2):
            try:
                kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42 + i)
                labels = kmeans.fit_predict(X)
                all_labels.append(labels)
            except:
                continue
        
        # Method 2: Gaussian Mixture Models
        if GMM_AVAILABLE:
            print("  Method 2: Gaussian Mixture Models...")
            for i in range(n_iterations // 4):
                try:
                    gmm = GaussianMixture(n_components=n_clusters, random_state=42 + i, max_iter=50)
                    labels = gmm.fit_predict(X)
                    all_labels.append(labels)
                except:
                    continue
        
        if len(all_labels) == 0:
            print("⚠️ No valid clusterings generated, returning None")
            return None, 0.0
        
        # === CONSENSUS MECHANISM ===
        print("\nComputing consensus via co-association matrix...")
        n_samples = X.shape[0]
        coassoc_matrix = np.zeros((n_samples, n_samples))
        
        for labels in all_labels:
            for i in range(n_samples):
                for j in range(i + 1, n_samples):
                    if labels[i] == labels[j]:
                        coassoc_matrix[i, j] += 1
                        coassoc_matrix[j, i] += 1
        
        coassoc_matrix /= len(all_labels)
        
        # Cluster the co-association matrix
        distance_matrix = 1 - coassoc_matrix
        
        try:
            final_clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, 
                metric='precomputed', 
                linkage='average'
            )
            consensus_labels = final_clusterer.fit_predict(distance_matrix)
        except:
            print("⚠️ Agglomerative clustering failed, using majority vote")
            # Fallback: majority vote
            consensus_labels = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                votes = [labels[i] for labels in all_labels]
                consensus_labels[i] = int(pd.Series(votes).mode()[0] if len(pd.Series(votes).mode()) > 0 else votes[0])
        
        # === STABILITY VALIDATION ===
        ari_scores = [adjusted_rand_score(consensus_labels, labels) for labels in all_labels]
        mean_ari = np.mean(ari_scores)
        
        print(f"\nConsensus clustering completed")
        print(f"Mean ARI with individual iterations: {mean_ari:.4f}")
        
        if mean_ari > 0.8:
            print("✅ Highly stable clustering (ARI > 0.8)")
        elif mean_ari > 0.6:
            print("⚠️ Moderately stable clustering (0.6 < ARI < 0.8)")
        else:
            print("❌ Unstable clustering (ARI < 0.6) - consider different features or algorithms")
        
        return consensus_labels, mean_ari
    
    def create_business_personas(self, df, cluster_labels, feature_cols):
        """
        PHASE 4 (Enhanced): Business Intelligence Layer
        Transform clusters into named business personas with narratives.
        """
        print("\n" + "="*80)
        print("PHASE 4 (ENHANCED): BUSINESS INTELLIGENCE GENERATION")
        print("="*80)
        
        df = df.copy()
        df['Cluster'] = cluster_labels
        personas = {}
        
        for cluster_id in sorted(df['Cluster'].unique()):
            cluster_data = df[df['Cluster'] == cluster_id]
            
            profile = {
                'cluster_id': cluster_id,
                'size': len(cluster_data),
                'pct_of_total': len(cluster_data) / len(df) * 100,
            }
            
            # Statistical profile
            for col in feature_cols:
                if col in cluster_data.columns and pd.api.types.is_numeric_dtype(cluster_data[col]):
                    profile[f'{col}_median'] = cluster_data[col].median()
                    profile[f'{col}_mean'] = cluster_data[col].mean()
                    profile[f'{col}_std'] = cluster_data[col].std()
            
            # === AUTOMATIC PERSONA NAMING ===
            avg_employees = cluster_data['Employees Total'].median() if 'Employees Total' in cluster_data.columns else 0
            avg_it_ratio = cluster_data['it_spend_ratio'].median() if 'it_spend_ratio' in cluster_data.columns else 0
            avg_age = cluster_data['Company Age'].median() if 'Company Age' in cluster_data.columns else 0
            
            # Determine company size category
            if avg_employees < 50:
                size_label = "Micro"
            elif avg_employees < 250:
                size_label = "Small"
            elif avg_employees < 1000:
                size_label = "Medium"
            else:
                size_label = "Enterprise"
            
            # Determine tech maturity
            if avg_it_ratio > 0.05:
                tech_label = "Tech-Forward"
            elif avg_it_ratio > 0.02:
                tech_label = "Digitally Moderate"
            else:
                tech_label = "Tech-Conservative"
            
            # Determine lifecycle stage
            if avg_age < 5:
                stage_label = "Startup"
            elif avg_age < 15:
                stage_label = "Growth Stage"
            else:
                stage_label = "Established"
            
            # Standardized persona name with cluster ID
            persona_name = f"Tier1_Cluster_{cluster_id}: {size_label} {tech_label} {stage_label}"
            profile['persona_name'] = persona_name
            profile['short_name'] = f"{size_label} {tech_label} {stage_label}"
            profile['cluster_id_standard'] = f"Tier1_Cluster_{cluster_id}"
            
            # === COMPARATIVE ANALYSIS ===
            for metric in ['revenue_per_employee', 'it_spend_ratio', 'technology_density']:
                if metric in cluster_data.columns and metric in df.columns:
                    cluster_median = cluster_data[metric].median()
                    overall_median = df[metric].median()
                    
                    if pd.notna(cluster_median) and pd.notna(overall_median) and overall_median != 0:
                        diff_pct = ((cluster_median - overall_median) / overall_median) * 100
                        profile[f'{metric}_vs_market'] = diff_pct
            
            # === RISK & OPPORTUNITY FLAGS ===
            profile['risk_flag'] = None
            profile['opportunity'] = None
            
            if 'it_spend_ratio_industry_zscore' in cluster_data.columns:
                avg_z = cluster_data['it_spend_ratio_industry_zscore'].mean()
                if avg_z < -0.5:
                    profile['risk_flag'] = "Under-investing in IT vs peers"
                    profile['opportunity'] = "Potential for digital transformation services"
                elif avg_z > 0.5:
                    profile['opportunity'] = "Early adopter segment - premium pricing potential"
            
            if 'revenue_per_employee' in profile and 'it_spend_ratio' in profile:
                if profile.get('revenue_per_employee_median', 0) < df['revenue_per_employee'].quantile(0.25) if 'revenue_per_employee' in df.columns else 0:
                    if profile['risk_flag']:
                        profile['risk_flag'] += "; Low productivity per employee"
                    else:
                        profile['risk_flag'] = "Low productivity per employee"
            
            personas[cluster_id] = profile
        
        # === GENERATE NARRATIVE DESCRIPTIONS ===
        for cluster_id, profile in personas.items():
            narrative = f"""
📊 **{profile['persona_name']}** (Cluster {cluster_id})

**Overview:**
- {profile['size']:,} companies ({profile['pct_of_total']:.1f}% of dataset)
- Typical size: {profile.get('Employees Total_median', 0):.0f} employees
- Median revenue: ${profile.get('Revenue (USD)_median', 0):,.0f}

**Technology Profile:**
- IT spend ratio: {profile.get('it_spend_ratio_median', 0):.2%}
- {profile.get('it_spend_ratio_vs_market', 0):+.1f}% vs market average

**Strategic Insights:**
"""
            
            if profile.get('risk_flag'):
                narrative += f"\n⚠️ **Risk:** {profile['risk_flag']}"
            
            if profile.get('opportunity'):
                narrative += f"\n💡 **Opportunity:** {profile['opportunity']}"
            
            profile['narrative'] = narrative.strip()
            print(f"\n{profile['narrative']}")
        
        return personas
    
    def create_interactive_dashboard(self, df, cluster_labels, personas):
        """
        PHASE 5 (Enhanced): Interactive Dashboard
        Create visualizations that let users explore the segmentation.
        """
        print("\n" + "="*80)
        print("PHASE 5 (ENHANCED): INTERACTIVE DASHBOARD CREATION")
        print("="*80)
        
        if not PLOTLY_AVAILABLE:
            print("⚠️ Plotly not available, skipping interactive dashboard")
            return
        
        df = df.copy()
        df['Cluster'] = cluster_labels
        df['Persona'] = df['Cluster'].map({k: v['persona_name'] for k, v in personas.items()})
        
        # === VIZ 1: 3D Cluster Scatter ===
        print("Creating 3D cluster visualization...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [c for c in numeric_cols if c not in ['Cluster']]
        
        if len(feature_cols) > 0:
            try:
                pca = PCA(n_components=3)
                X_viz = pca.fit_transform(df[feature_cols].fillna(0))
                
                df['PC1'] = X_viz[:, 0]
                df['PC2'] = X_viz[:, 1]
                df['PC3'] = X_viz[:, 2]
                
                hover_cols = []
                for col in ['Employees Total', 'Revenue (USD)', 'IT Spend']:
                    if col in df.columns:
                        hover_cols.append(col)
                
                fig1 = px.scatter_3d(
                    df, 
                    x='PC1', y='PC2', z='PC3',
                    color='Persona',
                    hover_data=hover_cols,
                    title='Company Segmentation - 3D View',
                    labels={'Persona': 'Business Segment'}
                )
                fig1.write_html('cluster_3d_interactive.html')
                print("  ✅ Saved: cluster_3d_interactive.html")
            except Exception as e:
                print(f"  ⚠️ Error creating 3D visualization: {e}")
        
        # === VIZ 2: Cluster Comparison Radar Chart ===
        print("Creating radar chart comparison...")
        metrics = ['revenue_per_employee', 'it_spend_ratio', 'technology_density', 
                   'employees_per_site', 'market_value_per_employee']
        
        available_metrics = [m for m in metrics if m in df.columns]
        
        if len(available_metrics) > 0:
            try:
                cluster_medians = df.groupby('Persona')[available_metrics].median()
                cluster_medians_norm = (cluster_medians - cluster_medians.min()) / (cluster_medians.max() - cluster_medians.min() + 1e-10)
                
                fig2 = go.Figure()
                
                for persona in cluster_medians_norm.index:
                    fig2.add_trace(go.Scatterpolar(
                        r=cluster_medians_norm.loc[persona].values,
                        theta=available_metrics,
                        fill='toself',
                        name=persona
                    ))
                
                fig2.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Cluster Performance Profiles - Radar View"
                )
                
                fig2.write_html('cluster_radar_comparison.html')
                print("  ✅ Saved: cluster_radar_comparison.html")
            except Exception as e:
                print(f"  ⚠️ Error creating radar chart: {e}")
        
        # === VIZ 3: Summary Table ===
        print("Creating summary table...")
        try:
            agg_dict = {}
            if 'Employees Total' in df.columns:
                agg_dict['Employees Total'] = ['median', 'mean', 'count']
            if 'Revenue (USD)' in df.columns:
                agg_dict['Revenue (USD)'] = 'median'
            if 'IT Spend' in df.columns:
                agg_dict['IT Spend'] = 'median'
            if 'it_spend_ratio' in df.columns:
                agg_dict['it_spend_ratio'] = 'median'
            
            if len(agg_dict) > 0:
                summary_table = df.groupby('Persona').agg(agg_dict).round(2)
                summary_table.to_html('cluster_summary_table.html')
                print("  ✅ Saved: cluster_summary_table.html")
        except Exception as e:
            print(f"  ⚠️ Error creating summary table: {e}")
        
        print("\n✅ Interactive dashboards created")
    
    def _create_enhanced_png_visualizations(self, df, cluster_labels, personas, feature_cols):
        """
        Generate PNG visualizations for the enhanced pipeline (matplotlib).
        """
        print("\nCreating PNG visualizations...")
        df = df.copy()
        df['Cluster'] = cluster_labels
        df['Persona'] = df['Cluster'].map({k: v['persona_name'] for k, v in personas.items()})
        
        # 1. Cluster size distribution
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            counts = df['Persona'].value_counts()
            colors = sns.color_palette("husl", len(counts))
            ax.bar(range(len(counts)), counts.values, color=colors)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels(counts.index, rotation=45, ha='right')
            ax.set_xlabel('Business Segment (Persona)', fontsize=12)
            ax.set_ylabel('Number of Companies', fontsize=12)
            ax.set_title('Cluster Size Distribution', fontsize=16, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('cluster_sizes.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  -> Saved: cluster_sizes.png")
        except Exception as e:
            print(f"  ⚠️ Error saving cluster_sizes.png: {e}")
        
        # 2. PCA 2D scatter (colored by Persona)
        numeric_cols = [c for c in feature_cols if c in df.columns]
        if len(numeric_cols) > 0:
            try:
                X = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
                pca = PCA(n_components=2, random_state=42)
                X2 = pca.fit_transform(X)
                df_plot = df.copy()
                df_plot['PC1'] = X2[:, 0]
                df_plot['PC2'] = X2[:, 1]
                
                fig, ax = plt.subplots(figsize=(12, 8))
                for i, persona in enumerate(df_plot['Persona'].unique()):
                    mask = df_plot['Persona'] == persona
                    ax.scatter(df_plot.loc[mask, 'PC1'], df_plot.loc[mask, 'PC2'], 
                               label=persona, alpha=0.6, s=30)
                ax.set_xlabel('PC1', fontsize=12)
                ax.set_ylabel('PC2', fontsize=12)
                ax.set_title('Company Segmentation - PCA 2D View', fontsize=16, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('cluster_2d_scatter.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  -> Saved: cluster_2d_scatter.png")
            except Exception as e:
                print(f"  ⚠️ Error saving cluster_2d_scatter.png: {e}")
        
        # 3. Cluster profiles heatmap
        metrics = ['revenue_per_employee', 'it_spend_ratio', 'technology_density', 
                   'employees_per_site', 'market_value_per_employee', 'Employees Total', 
                   'Revenue (USD)', 'IT Spend']
        available = [m for m in metrics if m in df.columns]
        if len(available) > 0:
            try:
                medians = df.groupby('Persona')[available].median()
                medians_norm = medians.apply(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-10), axis=0)
                fig, ax = plt.subplots(figsize=(max(10, len(available) * 1.5), max(6, len(medians_norm) * 0.5)))
                sns.heatmap(medians_norm.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax,
                           cbar_kws={'label': 'Normalized value'})
                ax.set_title('Cluster Profiles Heatmap (Normalized)', fontsize=16, fontweight='bold')
                ax.set_ylabel('Feature', fontsize=12)
                ax.set_xlabel('Persona', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig('cluster_profiles_heatmap.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  -> Saved: cluster_profiles_heatmap.png")
            except Exception as e:
                print(f"  ⚠️ Error saving cluster_profiles_heatmap.png: {e}")
        
        # 4. Radar-style bar comparison (key metrics by persona)
        metrics_radar = ['revenue_per_employee', 'it_spend_ratio', 'technology_density']
        available_radar = [m for m in metrics_radar if m in df.columns]
        if len(available_radar) >= 2:
            try:
                rad = df.groupby('Persona')[available_radar].median()
                rad_norm = (rad - rad.min()) / (rad.max() - rad.min() + 1e-10)
                rad_norm.plot(kind='bar', figsize=(12, 6), colormap='tab10')
                plt.title('Key Metrics by Business Segment (Normalized)', fontsize=16, fontweight='bold')
                plt.ylabel('Normalized value')
                plt.xlabel('Persona')
                plt.xticks(rotation=45, ha='right')
                plt.legend(title='Metric')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig('cluster_metrics_by_persona.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  -> Saved: cluster_metrics_by_persona.png")
            except Exception as e:
                print(f"  ⚠️ Error saving cluster_metrics_by_persona.png: {e}")
        
        print("✅ PNG visualizations saved")
    
    def _create_subcluster_visualizations(self, df_mega_refined, sub_labels):
        """
        Create visualizations for mega-cluster sub-segments.
        """
        print("\nCreating sub-cluster visualizations...")
        
        df = df_mega_refined.copy()
        df['SubCluster'] = sub_labels
        
        # 1. Sub-cluster distribution
        try:
            fig, ax = plt.subplots(figsize=(10, 6))
            counts = df['SubCluster'].value_counts().sort_index()
            colors = sns.color_palette("Set2", len(counts))
            ax.bar(range(len(counts)), counts.values, color=colors)
            ax.set_xticks(range(len(counts)))
            ax.set_xticklabels([f"Sub-{i}" for i in counts.index])
            ax.set_xlabel('Sub-Cluster ID', fontsize=12)
            ax.set_ylabel('Number of Companies', fontsize=12)
            ax.set_title('Mega-Cluster Sub-Segment Distribution', fontsize=16, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)
            
            # Add percentage labels
            total = counts.sum()
            for i, (idx, v) in enumerate(counts.items()):
                ax.text(i, v + total*0.01, f'{v/total*100:.1f}%', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.savefig('subcluster_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("  -> Saved: subcluster_distribution.png")
        except Exception as e:
            print(f"  ⚠️ Error saving subcluster_distribution.png: {e}")
        
        # 2. Sub-cluster profiles comparison
        metrics = ['Revenue (USD)', 'IT Spend', 'Employees Total', 'operating_leverage',
                   'tech_density_refined', 'Geographic Dispersion']
        available = [m for m in metrics if m in df.columns]
        
        if len(available) >= 2:
            try:
                medians = df.groupby('SubCluster')[available].median()
                medians_norm = medians.apply(lambda s: (s - s.min()) / (s.max() - s.min() + 1e-10), axis=0)
                
                fig, ax = plt.subplots(figsize=(12, 6))
                medians_norm.T.plot(kind='bar', ax=ax, colormap='Set2')
                ax.set_title('Sub-Cluster Profiles (Normalized Medians)', fontsize=16, fontweight='bold')
                ax.set_ylabel('Normalized Value')
                ax.set_xlabel('Metric')
                ax.legend(title='Sub-Cluster', bbox_to_anchor=(1.02, 1))
                plt.xticks(rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig('subcluster_profiles.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  -> Saved: subcluster_profiles.png")
            except Exception as e:
                print(f"  ⚠️ Error saving subcluster_profiles.png: {e}")
        
        # 3. Gap analysis visualization
        if 'IT Spend' in df.columns and 'Revenue (USD)' in df.columns:
            try:
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for sub_id in df['SubCluster'].unique():
                    sub_data = df[df['SubCluster'] == sub_id]
                    ax.scatter(
                        sub_data['Revenue (USD)'] / 1e6,  # Millions
                        sub_data['IT Spend'] / 1e3,  # Thousands
                        label=f'Sub-{sub_id} (n={len(sub_data)})',
                        alpha=0.5,
                        s=30
                    )
                
                ax.set_xlabel('Revenue (USD Millions)', fontsize=12)
                ax.set_ylabel('IT Spend (USD Thousands)', fontsize=12)
                ax.set_title('Sub-Cluster: Revenue vs IT Spend\n(Identify Investment Gaps)', 
                            fontsize=14, fontweight='bold')
                ax.legend(bbox_to_anchor=(1.02, 1))
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig('subcluster_gap_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()
                print("  -> Saved: subcluster_gap_analysis.png")
            except Exception as e:
                print(f"  ⚠️ Error saving subcluster_gap_analysis.png: {e}")
        
        print("✅ Sub-cluster visualizations saved")
    
    def identify_mega_cluster(self, cluster_labels, threshold_pct=40.0):
        """
        Identify the mega-cluster (dominant cluster with >threshold_pct of data).
        Returns the cluster ID and mask for the mega-cluster.
        """
        cluster_counts = pd.Series(cluster_labels).value_counts()
        total = len(cluster_labels)
        
        mega_clusters = []
        for cid, count in cluster_counts.items():
            pct = count / total * 100
            if pct >= threshold_pct:
                mega_clusters.append((cid, count, pct))
        
        return mega_clusters
    
    def create_refined_features_for_subset(self, df_subset):
        """
        Create high-variance features specifically for re-clustering a subset.
        These features are designed to find subtle differences within similar-sized companies.
        """
        df = df_subset.copy()
        
        # === TECH DENSITY (IT Spend / No. of Servers) ===
        if 'IT Spend' in df.columns and 'No. of Servers' in df.columns:
            df['tech_density_refined'] = df['IT Spend'] / df['No. of Servers'].replace(0, np.nan)
        
        # === OPERATING LEVERAGE (Revenue / Employees) ===
        if 'Revenue (USD)' in df.columns and 'Employees Total' in df.columns:
            df['operating_leverage'] = df['Revenue (USD)'] / df['Employees Total'].replace(0, np.nan)
        
        # === INFRASTRUCTURE INVESTMENT PER SITE ===
        if 'IT Spend' in df.columns and 'Company Sites' in df.columns:
            df['it_spend_per_site'] = df['IT Spend'] / df['Company Sites'].replace(0, np.nan)
        
        # === TECHNOLOGY COVERAGE (Servers per Site) ===
        if 'No. of Servers' in df.columns and 'Company Sites' in df.columns:
            df['servers_per_site'] = df['No. of Servers'] / df['Company Sites'].replace(0, np.nan)
        
        # === STORAGE INTENSITY ===
        if 'No. of Storage Devices' in df.columns and 'No. of Servers' in df.columns:
            df['storage_to_server_ratio'] = df['No. of Storage Devices'] / df['No. of Servers'].replace(0, np.nan)
        
        # === GEOGRAPHIC DISPERSION INDEX (normalized within subset) ===
        if 'Geographic Dispersion' in df.columns:
            gd = df['Geographic Dispersion']
            df['geo_dispersion_zscore'] = (gd - gd.mean()) / (gd.std() + 1e-10)
        
        # === MARKET VALUE EFFICIENCY ===
        if 'Market Value (USD)' in df.columns and 'IT Spend' in df.columns:
            df['mv_per_it_spend'] = df['Market Value (USD)'] / df['IT Spend'].replace(0, np.nan)
        
        # === AGE-ADJUSTED METRICS ===
        if 'Company Age' in df.columns and 'Revenue (USD)' in df.columns:
            df['revenue_per_year_age'] = df['Revenue (USD)'] / df['Company Age'].replace(0, np.nan)
        
        # === WITHIN-SUBSET Z-SCORES (to find relative differences) ===
        for col in ['Revenue (USD)', 'IT Spend', 'Employees Total', 'Market Value (USD)']:
            if col in df.columns:
                vals = df[col]
                df[f'{col}_subset_zscore'] = (vals - vals.mean()) / (vals.std() + 1e-10)
        
        return df
    
    def recluster_mega_cluster_gmm(self, df_processed, X_scaled, cluster_labels, mega_cluster_id, 
                                    feature_cols, n_subclusters=5):
        """
        Re-cluster the mega-cluster using GMM with refined features.
        Returns new sub-cluster labels for the mega-cluster.
        """
        print(f"\n🔬 RE-CLUSTERING MEGA-CLUSTER {mega_cluster_id} WITH GMM")
        print("="*80)
        
        # Get the mega-cluster data
        mega_mask = cluster_labels == mega_cluster_id
        df_mega = df_processed[mega_mask].copy()
        X_mega = X_scaled[mega_mask]
        
        print(f"Mega-cluster size: {len(df_mega)} companies")
        
        # Create refined features for this subset
        print("\nCreating refined features for subtle differentiation...")
        df_mega_refined = self.create_refined_features_for_subset(df_mega)
        
        # Select refined features for re-clustering
        refined_feature_cols = [
            'tech_density_refined', 'operating_leverage', 'it_spend_per_site',
            'servers_per_site', 'storage_to_server_ratio', 'geo_dispersion_zscore',
            'mv_per_it_spend', 'revenue_per_year_age',
            'Revenue (USD)_subset_zscore', 'IT Spend_subset_zscore', 
            'Employees Total_subset_zscore', 'Market Value (USD)_subset_zscore'
        ]
        available_refined = [c for c in refined_feature_cols if c in df_mega_refined.columns]
        
        # Also include some original features
        original_keep = ['Geographic Dispersion', 'Company Age', 'Company Sites']
        for col in original_keep:
            if col in df_mega_refined.columns and col not in available_refined:
                available_refined.append(col)
        
        print(f"Using {len(available_refined)} refined features: {available_refined[:5]}...")
        
        # Prepare data for GMM
        X_refined = df_mega_refined[available_refined].fillna(0).replace([np.inf, -np.inf], 0)
        
        # Scale the refined features
        scaler = RobustScaler()
        X_refined_scaled = scaler.fit_transform(X_refined)
        
        # Find optimal number of sub-clusters using BIC
        print("\nFinding optimal number of sub-clusters using BIC...")
        best_gmm = None
        best_bic = np.inf
        best_n = n_subclusters
        
        for n in range(3, min(10, len(df_mega) // 100)):
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type='full',
                    n_init=10,
                    random_state=42,
                    max_iter=200
                )
                gmm.fit(X_refined_scaled)
                bic = gmm.bic(X_refined_scaled)
                print(f"  n={n}: BIC={bic:.2f}")
                
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
                    best_n = n
            except Exception as e:
                print(f"  n={n}: Error - {e}")
                continue
        
        print(f"\nOptimal sub-clusters: {best_n} (BIC: {best_bic:.2f})")
        
        # Get sub-cluster labels
        sub_labels = best_gmm.predict(X_refined_scaled)
        sub_probs = best_gmm.predict_proba(X_refined_scaled)
        
        # Calculate confidence (max probability)
        confidence = sub_probs.max(axis=1)
        
        # === TIER 2 SILHOUETTE SCORE ===
        if len(np.unique(sub_labels)) > 1:
            tier2_silhouette = silhouette_score(X_refined_scaled, sub_labels)
            print(f"\n📊 Tier 2 Silhouette Score: {tier2_silhouette:.4f}")
            if tier2_silhouette > 0.5:
                print("  ✅ Good sub-cluster separation")
            elif tier2_silhouette > 0.3:
                print("  ⚠️ Moderate sub-cluster separation")
            else:
                print("  ❌ Weak sub-cluster separation")
        else:
            tier2_silhouette = 0.0
        
        # Store results with STANDARDIZED NAMING
        df_mega_refined['SubCluster'] = sub_labels
        df_mega_refined['SubCluster_Confidence'] = confidence
        df_mega_refined['SubCluster_Name'] = [f"Tier2_Subcluster_{mega_cluster_id}.{s}" for s in sub_labels]
        
        # Analyze sub-clusters
        print("\nSub-cluster distribution (standardized naming):")
        sub_counts = pd.Series(sub_labels).value_counts().sort_index()
        for sub_id, count in sub_counts.items():
            pct = count / len(sub_labels) * 100
            print(f"  Tier2_Subcluster_{mega_cluster_id}.{sub_id}: {count} companies ({pct:.1f}%)")
        
        # Store silhouette in results
        self.results['tier2_silhouette'] = tier2_silhouette
        
        return df_mega_refined, sub_labels, available_refined
    
    def filter_small_clusters_and_identify_anomalies(self, df_processed, cluster_labels, 
                                                       threshold_pct=0.5):
        """
        Filter out small clusters (< threshold_pct) and identify anomalies.
        Creates:
        - Main personas (clusters >= threshold_pct)
        - Anomalous Insight Table (small clusters + outliers)
        """
        print(f"\n🔍 FILTERING SMALL CLUSTERS & IDENTIFYING ANOMALIES")
        print("="*80)
        
        total = len(cluster_labels)
        min_size = int(total * threshold_pct / 100)
        print(f"Minimum cluster size for personas: {min_size} companies ({threshold_pct}%)")
        
        cluster_counts = pd.Series(cluster_labels).value_counts()
        
        main_clusters = []
        anomaly_clusters = []
        
        for cid, count in cluster_counts.items():
            pct = count / total * 100
            if count >= min_size:
                main_clusters.append(cid)
            else:
                anomaly_clusters.append(cid)
        
        print(f"\nMain clusters (>= {threshold_pct}%): {len(main_clusters)}")
        print(f"Anomaly clusters (< {threshold_pct}%): {len(anomaly_clusters)}")
        
        # Create anomaly mask
        anomaly_mask = np.isin(cluster_labels, anomaly_clusters)
        df_anomalies = df_processed[anomaly_mask].copy()
        df_anomalies['Original_Cluster'] = cluster_labels[anomaly_mask]
        
        # Identify anomaly types
        anomaly_insights = self._classify_anomalies(df_anomalies)
        
        return main_clusters, anomaly_clusters, df_anomalies, anomaly_insights
    
    def _classify_anomalies(self, df_anomalies):
        """
        Classify anomalies into strategic categories with STRICT definitions:
        - True Unicorn: Top 1% Rev/Emp AND Bottom 10% Age
        - High Performer: Top 1% Rev/Emp but NOT young
        - Zombie Firms: Old age, very low revenue
        - Tech Giants: High IT spend relative to size
        - Lean Operators: High revenue per employee
        """
        insights = []
        
        # Calculate percentiles for strict classification
        if 'Revenue_per_Emp' in df_anomalies.columns:
            rev_per_emp_col = 'Revenue_per_Emp'
        elif 'revenue_per_employee' in df_anomalies.columns:
            rev_per_emp_col = 'revenue_per_employee'
        else:
            rev_per_emp_col = None
        
        # Get global percentiles (use full dataset if available)
        if hasattr(self, 'df_processed') and self.df_processed is not None:
            ref_df = self.df_processed
        else:
            ref_df = df_anomalies
        
        # Calculate thresholds
        if rev_per_emp_col and rev_per_emp_col in ref_df.columns:
            p99_rev_per_emp = ref_df[rev_per_emp_col].quantile(0.99)
        else:
            p99_rev_per_emp = np.inf
        
        if 'Company Age' in ref_df.columns:
            p10_age = ref_df['Company Age'].quantile(0.10)
        else:
            p10_age = 0
        
        print(f"  Unicorn thresholds: Rev/Emp > ${p99_rev_per_emp:,.0f} (P99) AND Age < {p10_age:.0f} years (P10)")
        
        for idx, row in df_anomalies.iterrows():
            insight = {
                'index': idx,
                'original_cluster': row.get('Original_Cluster', -1),
                'category': 'Unclassified',
                'flags': []
            }
            
            # Get key metrics
            employees = row.get('Employees Total', 0) or 0
            revenue = row.get('Revenue (USD)', 0) or 0
            age = row.get('Company Age', 0) or 0
            it_spend = row.get('IT Spend', 0) or 0
            
            # Calculate ratios
            rev_per_emp = row.get(rev_per_emp_col, 0) if rev_per_emp_col else (revenue / employees if employees > 0 else 0)
            it_ratio = it_spend / revenue if revenue > 0 else 0
            
            # === TRUE UNICORN CHECK (STRICT) ===
            # Top 1% of Revenue per Employee AND Bottom 10% of Company Age
            is_top1_efficiency = rev_per_emp > p99_rev_per_emp if pd.notna(rev_per_emp) else False
            is_young = age < p10_age if pd.notna(age) and p10_age > 0 else False
            
            if is_top1_efficiency and is_young:
                insight['category'] = 'True Unicorn'
                insight['flags'].append(f"🦄 P99 efficiency: ${rev_per_emp:,.0f}/employee")
                insight['flags'].append(f"Young company: {age:.0f} years (< P10)")
            
            # === HIGH PERFORMER (not young enough for Unicorn) ===
            elif is_top1_efficiency:
                insight['category'] = 'High Performer'
                insight['flags'].append(f"P99 efficiency: ${rev_per_emp:,.0f}/employee")
                insight['flags'].append(f"Age: {age:.0f} years (not in P10)")
            
            # === ZOMBIE FIRM CHECK ===
            # Old (>30 years), very low revenue (<$100K) or zero
            elif age > 30 and revenue < 100_000:
                insight['category'] = 'Zombie Firm'
                insight['flags'].append(f"Low revenue after {age:.0f} years")
                if employees > 10:
                    insight['flags'].append(f"Potentially inefficient ({employees:.0f} employees)")
            
            # === TECH GIANT CHECK ===
            # High IT spend ratio (>10%) or absolute IT spend > $1M
            elif it_ratio > 0.10 or it_spend > 1_000_000:
                insight['category'] = 'Tech-Heavy Outlier'
                insight['flags'].append(f"IT spend ratio: {it_ratio:.1%}")
                if it_spend > 1_000_000:
                    insight['flags'].append(f"IT spend: ${it_spend:,.0f}")
            
            # === LEAN OPERATOR CHECK ===
            # Very high revenue per employee (>$500K)
            elif rev_per_emp > 500_000:
                insight['category'] = 'Lean Operator'
                insight['flags'].append(f"Revenue/employee: ${rev_per_emp:,.0f}")
            
            # === DEFAULT: Strategic Outlier ===
            else:
                insight['category'] = 'Strategic Outlier'
                if revenue > 0:
                    insight['flags'].append(f"Revenue: ${revenue:,.0f}")
                if employees > 0:
                    insight['flags'].append(f"Employees: {employees}")
            
            insights.append(insight)
        
        return insights
    
    def generate_gap_analysis(self, df_subclusters, sub_labels, refined_features):
        """
        Generate commercial value gap analysis - DEDUPLICATED.
        Compares each sub-cluster against GLOBAL MEDIAN only (not pairwise).
        """
        print("\n💰 COMMERCIAL VALUE GAP ANALYSIS (vs Global Median)")
        print("="*80)
        
        df = df_subclusters.copy()
        df['SubCluster'] = sub_labels
        
        gap_insights = []
        
        # Key metrics for gap analysis
        metrics = ['Revenue (USD)', 'IT Spend', 'Employees Total', 'Market Value (USD)',
                   'operating_leverage', 'tech_density_refined', 'Revenue_per_Emp', 'IT_Spend_Ratio']
        available_metrics = [m for m in metrics if m in df.columns]
        
        if len(available_metrics) == 0:
            print("  ⚠️ No metrics available for gap analysis")
            return gap_insights
        
        # Calculate GLOBAL medians (baseline)
        global_medians = df[available_metrics].median()
        print(f"\nGlobal Medians (baseline for comparison):")
        for m in available_metrics[:4]:
            val = global_medians[m]
            if 'Ratio' in m or 'ratio' in m:
                print(f"  {m}: {val:.2%}")
            elif val > 1000:
                print(f"  {m}: ${val:,.0f}")
            else:
                print(f"  {m}: {val:.2f}")
        
        # Calculate cluster medians
        cluster_medians = df.groupby('SubCluster')[available_metrics].median()
        cluster_counts = df['SubCluster'].value_counts()
        
        print(f"\nSub-cluster gaps vs Global Median:")
        
        # Compare each sub-cluster against global median (NOT pairwise)
        for sub_id in sorted(df['SubCluster'].unique()):
            sub_median = cluster_medians.loc[sub_id]
            sub_count = cluster_counts[sub_id]
            
            # Check IT Spend gap
            if 'IT Spend' in available_metrics:
                it_sub = sub_median['IT Spend']
                it_global = global_medians['IT Spend']
                
                if pd.notna(it_sub) and pd.notna(it_global) and it_global > 0:
                    it_diff_pct = (it_sub - it_global) / it_global * 100
                    
                    if abs(it_diff_pct) > 25:  # >25% deviation from global
                        direction = "above" if it_diff_pct > 0 else "below"
                        it_gap = abs(it_sub - it_global)
                        
                        gap_insight = {
                            'subcluster': f"Tier2_Subcluster_5.{sub_id}",
                            'type': 'IT Investment Gap',
                            'metric': 'IT Spend',
                            'subcluster_value': it_sub,
                            'global_median': it_global,
                            'deviation_pct': it_diff_pct,
                            'description': f"Tier2_Subcluster_5.{sub_id} is {abs(it_diff_pct):.0f}% {direction} global IT median",
                            'opportunity_value': it_gap * sub_count if it_diff_pct < 0 else None,
                            'opportunity_description': f"${it_gap * sub_count:,.0f} potential IT market ({sub_count} companies)" 
                                                      if it_diff_pct < 0 else f"Tech leaders - premium service segment"
                        }
                        gap_insights.append(gap_insight)
                        print(f"\n  📊 Tier2_Subcluster_5.{sub_id}: IT Spend {it_diff_pct:+.1f}% vs global")
            
            # Check Revenue per Employee gap (efficiency)
            rev_per_emp_col = 'Revenue_per_Emp' if 'Revenue_per_Emp' in available_metrics else 'operating_leverage'
            if rev_per_emp_col in available_metrics:
                eff_sub = sub_median[rev_per_emp_col]
                eff_global = global_medians[rev_per_emp_col]
                
                if pd.notna(eff_sub) and pd.notna(eff_global) and eff_global > 0:
                    eff_diff_pct = (eff_sub - eff_global) / eff_global * 100
                    
                    if abs(eff_diff_pct) > 30:  # >30% deviation
                        direction = "more" if eff_diff_pct > 0 else "less"
                        
                        gap_insight = {
                            'subcluster': f"Tier2_Subcluster_5.{sub_id}",
                            'type': 'Efficiency Gap',
                            'metric': rev_per_emp_col,
                            'subcluster_value': eff_sub,
                            'global_median': eff_global,
                            'deviation_pct': eff_diff_pct,
                            'description': f"Tier2_Subcluster_5.{sub_id} is {abs(eff_diff_pct):.0f}% {direction} efficient than global",
                            'opportunity_value': None,
                            'opportunity_description': f"Benchmark target for efficiency improvement" 
                                                      if eff_diff_pct > 0 else f"Efficiency consulting opportunity"
                        }
                        gap_insights.append(gap_insight)
                        print(f"  📊 Tier2_Subcluster_5.{sub_id}: Efficiency {eff_diff_pct:+.1f}% vs global")
        
        # Summary
        print(f"\n✅ Found {len(gap_insights)} significant gaps (>25-30% deviation from global median)")
        
        return gap_insights
    
    def create_two_tier_clustering(self, df_processed, X_scaled, feature_cols):
        """
        Two-tier clustering approach:
        Tier 1: Main population clustering
        Tier 2: Re-cluster the mega-cluster for finer segmentation
        """
        print("\n" + "="*80)
        print("🎯 TWO-TIER CLUSTERING APPROACH")
        print("="*80)
        
        # TIER 1: Initial clustering
        print("\n📌 TIER 1: Main Population Clustering")
        
        # Use GMM for initial clustering
        best_gmm = None
        best_bic = np.inf
        
        for n in range(5, 10):
            try:
                gmm = GaussianMixture(
                    n_components=n,
                    covariance_type='full',
                    n_init=10,
                    random_state=42,
                    max_iter=100
                )
                gmm.fit(X_scaled)
                bic = gmm.bic(X_scaled)
                
                if bic < best_bic:
                    best_bic = bic
                    best_gmm = gmm
            except:
                continue
        
        tier1_labels = best_gmm.predict(X_scaled)
        n_tier1_clusters = len(np.unique(tier1_labels))
        print(f"Tier 1 clusters: {n_tier1_clusters}")
        
        # === TIER 1 SILHOUETTE SCORE ===
        if n_tier1_clusters > 1:
            tier1_silhouette = silhouette_score(X_scaled, tier1_labels)
            print(f"\n📊 Tier 1 Silhouette Score: {tier1_silhouette:.4f}")
            if tier1_silhouette > 0.5:
                print("  ✅ Good cluster separation")
            elif tier1_silhouette > 0.3:
                print("  ⚠️ Moderate cluster separation")
            else:
                print("  ❌ Weak cluster separation")
            self.results['tier1_silhouette'] = tier1_silhouette
        else:
            self.results['tier1_silhouette'] = 0.0
        
        # Identify mega-cluster
        mega_clusters = self.identify_mega_cluster(tier1_labels, threshold_pct=40.0)
        
        if len(mega_clusters) == 0:
            print("✅ No mega-cluster detected - Tier 1 is sufficient")
            return tier1_labels, None, None, None, None
        
        # TIER 2: Re-cluster the mega-cluster
        print(f"\n📌 TIER 2: Re-clustering mega-cluster(s)")
        
        mega_cid, mega_count, mega_pct = mega_clusters[0]
        print(f"Mega-cluster {mega_cid}: {mega_count} companies ({mega_pct:.1f}%)")
        
        df_mega_refined, sub_labels, refined_features = self.recluster_mega_cluster_gmm(
            df_processed, X_scaled, tier1_labels, mega_cid, feature_cols, n_subclusters=5
        )
        
        # Filter small clusters and identify anomalies
        print("\n📌 Filtering small clusters and identifying anomalies...")
        main_clusters, anomaly_clusters, df_anomalies, anomaly_insights = \
            self.filter_small_clusters_and_identify_anomalies(df_processed, tier1_labels, threshold_pct=0.5)
        
        # Generate gap analysis for sub-clusters
        gap_insights = self.generate_gap_analysis(df_mega_refined, sub_labels, refined_features)
        
        return tier1_labels, df_mega_refined, sub_labels, anomaly_insights, gap_insights
    
    def generate_tiered_report(self, df_processed, tier1_labels, df_mega_refined, sub_labels,
                                anomaly_insights, gap_insights, feature_cols):
        """
        Generate comprehensive tiered clustering report.
        """
        print("\n" + "="*80)
        print("📝 GENERATING TIERED CLUSTERING REPORT")
        print("="*80)
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("TWO-TIER CLUSTERING ANALYSIS REPORT")
        report_lines.append("(Standardized Naming: Tier1_Cluster_[X], Tier2_Subcluster_[X].[Y])")
        report_lines.append("="*80)
        report_lines.append("")
        
        # === QUALITY METRICS ===
        report_lines.append("CLUSTER QUALITY METRICS")
        report_lines.append("-"*80)
        tier1_sil = self.results.get('tier1_silhouette', 0)
        tier2_sil = self.results.get('tier2_silhouette', 0)
        report_lines.append(f"Tier 1 Silhouette Score: {tier1_sil:.4f} {'✅' if tier1_sil > 0.5 else '⚠️'}")
        report_lines.append(f"Tier 2 Silhouette Score: {tier2_sil:.4f} {'✅' if tier2_sil > 0.5 else '⚠️'}")
        report_lines.append("")
        
        # === TIER 1 SUMMARY ===
        report_lines.append("TIER 1: MAIN POPULATION SEGMENTATION")
        report_lines.append("-"*80)
        
        cluster_counts = pd.Series(tier1_labels).value_counts().sort_index()
        total = len(tier1_labels)
        
        for cid, count in cluster_counts.items():
            pct = count / total * 100
            status = "⚠️ MEGA-CLUSTER" if pct > 40 else ("📍 MINOR (<0.5%)" if pct < 0.5 else "✅")
            report_lines.append(f"Tier1_Cluster_{cid}: {count:,} companies ({pct:.1f}%) {status}")
        
        report_lines.append("")
        
        # === TIER 2 SUMMARY (Mega-cluster breakdown) ===
        if df_mega_refined is not None and sub_labels is not None:
            # Find mega-cluster ID
            mega_cid = cluster_counts.idxmax()
            
            report_lines.append(f"TIER 2: SUB-SEGMENTATION OF Tier1_Cluster_{mega_cid} (GMM)")
            report_lines.append("-"*80)
            
            sub_counts = pd.Series(sub_labels).value_counts().sort_index()
            mega_total = len(sub_labels)
            
            for sub_id, count in sub_counts.items():
                pct = count / mega_total * 100
                
                # Get sub-cluster profile
                sub_data = df_mega_refined[df_mega_refined['SubCluster'] == sub_id]
                
                median_revenue = sub_data['Revenue (USD)'].median() if 'Revenue (USD)' in sub_data.columns else 0
                median_it = sub_data['IT Spend'].median() if 'IT Spend' in sub_data.columns else 0
                median_emp = sub_data['Employees Total'].median() if 'Employees Total' in sub_data.columns else 0
                
                report_lines.append(f"\nTier2_Subcluster_{mega_cid}.{sub_id}: {count:,} companies ({pct:.1f}% of mega-cluster)")
                report_lines.append(f"  Median Revenue: ${median_revenue:,.0f}")
                report_lines.append(f"  Median IT Spend: ${median_it:,.0f}")
                report_lines.append(f"  Median Employees: {median_emp:.0f}")
            
            report_lines.append("")
        
        # === GAP ANALYSIS ===
        if gap_insights and len(gap_insights) > 0:
            report_lines.append("COMMERCIAL VALUE GAP ANALYSIS")
            report_lines.append("-"*80)
            
            for insight in gap_insights:
                report_lines.append(f"\n📊 {insight['type']}")
                report_lines.append(f"   {insight['description']}")
                report_lines.append(f"   💡 {insight['opportunity_description']}")
            
            report_lines.append("")
        
        # === ANOMALOUS INSIGHT TABLE ===
        if anomaly_insights and len(anomaly_insights) > 0:
            report_lines.append("STRATEGIC OUTLIERS (Anomalous Insight Table)")
            report_lines.append("-"*80)
            report_lines.append("Companies in clusters < 0.5% of population - for manual inspection")
            report_lines.append("")
            
            # Group by category
            categories = {}
            for insight in anomaly_insights:
                cat = insight['category']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(insight)
            
            for category, items in categories.items():
                emoji = {'Unicorn': '🦄', 'Zombie Firm': '🧟', 'Tech-Heavy Outlier': '💻', 
                        'Lean Operator': '🏃', 'Strategic Outlier': '🔍'}.get(category, '📌')
                report_lines.append(f"\n{emoji} {category}: {len(items)} companies")
                for item in items[:5]:  # Show top 5 per category
                    flags = "; ".join(item['flags'][:2]) if item['flags'] else "No flags"
                    report_lines.append(f"   - Cluster {item['original_cluster']}: {flags}")
                if len(items) > 5:
                    report_lines.append(f"   ... and {len(items) - 5} more")
            
            report_lines.append("")
        
        report_lines.append("="*80)
        report_lines.append("END OF TIERED REPORT")
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        
        with open('tiered_clustering_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("✅ Saved: tiered_clustering_report.txt")
        print("\n" + report_text)
        
        return report_text
    
    def run_full_analysis(self):
        """Run the complete analysis - enhanced or standard pipeline"""
        if self.use_enhanced_pipeline:
            return self.run_enhanced_pipeline()
        else:
            return self.run_standard_pipeline()
    
    def run_standard_pipeline(self):
        """Run the original 4-phase analysis"""
        try:
            # Phase 1
            self.phase1_context_analysis()
            
            # Phase 2
            self.phase2_filtering_encoding()
            
            # Phase 3
            self.phase3_clustering_loop()
            
            # Phase 4
            self.phase4_interpretability()
            
            # Generate report
            self.generate_summary_report()
            
            print("\n" + "="*80)
            print("ANALYSIS COMPLETE!")
            print("="*80)
            print("\nGenerated files:")
            print("  - clustering_analysis_report.txt (summary report)")
            print("  - umap_clusters.png (UMAP visualization)")
            print("  - cluster_sizes.png (cluster size distribution)")
            print("  - feature_importance.png (feature importance plot)")
            print("  - cluster_profiles_heatmap.png (cluster profiles)")
            
            return self.results
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def run_enhanced_pipeline(self):
        """
        Run the enhanced competition-grade pipeline with business intelligence
        """
        try:
            print("="*80)
            print("🏆 COMPETITION-GRADE CLUSTERING PIPELINE")
            print("="*80)
            
            # Load data
            print("\n📊 Loading data...")
            self.df = pd.read_excel(self.data_file)
            print(f"Dataset shape: {self.df.shape}")
            
            # PHASE 1: Domain-Driven Feature Engineering
            print("\n📐 PHASE 1: Domain-Driven Feature Engineering")
            self.df_processed = self.engineer_business_features(self.df)
            
            # Select features for clustering
            exclude_cols = ['Company Name', 'Country', 'Region', 'Industry', 'Sector', 
                          'Cluster', 'Persona', 'age_bracket', 'size_bracket', 'SubCluster',
                          'SubCluster_Name', 'SubCluster_Confidence', 'Original_Cluster']
            feature_cols = [col for col in self.df_processed.columns 
                          if col not in exclude_cols
                          and pd.api.types.is_numeric_dtype(self.df_processed[col])]
            
            X = self.df_processed[feature_cols].fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
            
            print(f"Initial features: {len(feature_cols)}")
            
            # === VIF CHECK (Remove multicollinear features) ===
            print("\n📊 PHASE 1b: Multicollinearity Check (VIF)")
            feature_cols_clean, X_clean = self.calculate_vif(X.values, feature_cols, threshold=10.0)
            
            removed_features = set(feature_cols) - set(feature_cols_clean)
            if removed_features:
                print(f"Removed {len(removed_features)} high-VIF features")
                self.results['removed_vif_features'] = list(removed_features)
            
            feature_cols = feature_cols_clean
            X = pd.DataFrame(X_clean, columns=feature_cols, index=self.df_processed.index)
            
            print(f"✅ Final features for clustering: {len(feature_cols)}")
            
            # Scale features
            scaler = RobustScaler()
            X_scaled = scaler.fit_transform(X)
            
            # PHASE 2: Hierarchical Clustering
            print("\n🎯 PHASE 2: Hierarchical Multi-Level Clustering")
            final_labels, macro_labels = self.hierarchical_segmentation(
                X_scaled, 
                min_cluster_size=100
            )
            
            if final_labels is None:
                print("⚠️ Hierarchical clustering failed, using standard K-Means")
                kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
                final_labels = kmeans.fit_predict(X_scaled)
                macro_labels = None
            
            self.cluster_labels = final_labels
            
            # PHASE 3: TWO-TIER CLUSTERING (Re-cluster mega-cluster + filter anomalies)
            print("\n🔬 PHASE 3: TWO-TIER CLUSTERING ANALYSIS")
            tier1_labels, df_mega_refined, sub_labels, anomaly_insights, gap_insights = \
                self.create_two_tier_clustering(self.df_processed, X_scaled, feature_cols)
            
            self.cluster_labels = tier1_labels
            self.results['tier1_labels'] = tier1_labels
            self.results['df_mega_refined'] = df_mega_refined
            self.results['sub_labels'] = sub_labels
            self.results['anomaly_insights'] = anomaly_insights
            self.results['gap_insights'] = gap_insights
            
            # PHASE 4: Business Intelligence (for main clusters only)
            print("\n💼 PHASE 4: Business Intelligence Generation")
            
            # Filter small clusters for persona generation
            main_clusters, anomaly_clusters, df_anomalies, _ = \
                self.filter_small_clusters_and_identify_anomalies(
                    self.df_processed, self.cluster_labels, threshold_pct=0.5
                )
            
            # Create personas only for main clusters
            df_main = self.df_processed.copy()
            main_mask = np.isin(self.cluster_labels, main_clusters)
            
            self.personas = self.create_business_personas(
                df_main[main_mask], 
                self.cluster_labels[main_mask], 
                feature_cols
            )
            
            # PHASE 5: Interactive Outputs
            print("\n📊 PHASE 5: Creating Interactive Dashboards")
            self.create_interactive_dashboard(
                self.df_processed, 
                self.cluster_labels, 
                self.personas
            )
            
            # Add cluster labels to processed dataframe
            self.df_processed['Cluster'] = self.cluster_labels
            self.df_processed['Persona'] = self.df_processed['Cluster'].map(
                {k: v['persona_name'] for k, v in self.personas.items()}
            )
            self.df_processed['Persona'] = self.df_processed['Persona'].fillna('Strategic Outlier')
            
            # Generate PNG visualizations
            print("\n🖼️ PHASE 5b: Creating PNG visualizations")
            self._create_enhanced_png_visualizations(
                self.df_processed,
                self.cluster_labels,
                self.personas,
                feature_cols,
            )
            
            # Create sub-cluster visualizations if mega-cluster was re-clustered
            if df_mega_refined is not None and sub_labels is not None:
                self._create_subcluster_visualizations(df_mega_refined, sub_labels)
            
            # Generate validation report
            print("\n📋 Generating validation report...")
            self._generate_enhanced_validation_report(X_scaled, feature_cols)
            
            # Generate tiered report
            print("\n📋 Generating tiered clustering report...")
            self.generate_tiered_report(
                self.df_processed, tier1_labels, df_mega_refined, sub_labels,
                anomaly_insights, gap_insights, feature_cols
            )
            
            print("\n" + "="*80)
            print("✅ ENHANCED PIPELINE COMPLETE - Ready for Presentation")
            print("="*80)
            print("\nGenerated files:")
            print("  - cluster_3d_interactive.html (3D interactive visualization)")
            print("  - cluster_radar_comparison.html (radar chart comparison)")
            print("  - cluster_summary_table.html (summary table)")
            print("  - cluster_sizes.png (cluster size distribution)")
            print("  - cluster_2d_scatter.png (PCA 2D scatter)")
            print("  - cluster_profiles_heatmap.png (cluster profiles heatmap)")
            print("  - cluster_metrics_by_persona.png (metrics by persona)")
            print("  - subcluster_distribution.png (mega-cluster sub-segments)")
            print("  - subcluster_profiles.png (sub-cluster comparison)")
            print("  - enhanced_clustering_report.txt (validation report)")
            print("  - tiered_clustering_report.txt (two-tier analysis)")
            
            return {
                'cluster_labels': self.cluster_labels,
                'personas': self.personas,
                'df_processed': self.df_processed,
                'df_mega_refined': df_mega_refined,
                'sub_labels': sub_labels,
                'anomaly_insights': anomaly_insights,
                'gap_insights': gap_insights,
                'results': self.results
            }
            
        except Exception as e:
            print(f"\nError during enhanced analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _generate_enhanced_validation_report(self, X_scaled, feature_cols):
        """Generate metrics proving cluster quality"""
        from sklearn.metrics import davies_bouldin_score
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ENHANCED CLUSTERING VALIDATION REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # Cluster metrics
        sil = silhouette_score(X_scaled, self.cluster_labels)
        ch = calinski_harabasz_score(X_scaled, self.cluster_labels)
        db = davies_bouldin_score(X_scaled, self.cluster_labels)
        
        report_lines.append("CLUSTER QUALITY METRICS")
        report_lines.append("-"*80)
        report_lines.append(f"Silhouette Score: {sil:.4f} {'✅' if sil > 0.5 else '⚠️'}")
        report_lines.append(f"Calinski-Harabasz Score: {ch:.4f}")
        report_lines.append(f"Davies-Bouldin Score: {db:.4f} {'✅' if db < 1.0 else '⚠️'} (lower is better)")
        report_lines.append("")
        
        # Cluster balance
        cluster_sizes = pd.Series(self.cluster_labels).value_counts().sort_index()
        largest_pct = cluster_sizes.max() / len(self.cluster_labels) * 100
        
        report_lines.append("CLUSTER BALANCE")
        report_lines.append("-"*80)
        report_lines.append(f"Number of clusters: {len(cluster_sizes)}")
        report_lines.append(f"Largest cluster: {largest_pct:.1f}% {'✅' if largest_pct < 40 else '⚠️'}")
        report_lines.append(f"Cluster size range: {cluster_sizes.min()} - {cluster_sizes.max()}")
        report_lines.append("")
        
        # Personas summary
        report_lines.append("BUSINESS PERSONAS")
        report_lines.append("-"*80)
        for cluster_id, persona in sorted(self.personas.items()):
            report_lines.append(f"\n{persona['persona_name']} (Cluster {cluster_id})")
            report_lines.append(f"  Size: {persona['size']:,} companies ({persona['pct_of_total']:.1f}%)")
            if persona.get('risk_flag'):
                report_lines.append(f"  ⚠️ Risk: {persona['risk_flag']}")
            if persona.get('opportunity'):
                report_lines.append(f"  💡 Opportunity: {persona['opportunity']}")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        report_text = "\n".join(report_lines)
        with open('enhanced_clustering_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print("✅ Validation report saved to: enhanced_clustering_report.txt")
        print("\n" + report_text)


if __name__ == "__main__":
    # Run the enhanced analysis (competition-grade pipeline)
    # Set use_enhanced_pipeline=False to use the original 4-phase method
    # Use strict_validation_gate=True to raise and halt if Silhouette < 0.5.
    analyzer = LatentSparseClustering(
        data_file='champions_group_processed.xlsx',
        strict_validation_gate=False,
        use_enhanced_pipeline=True,  # Use enhanced pipeline by default
    )
    results = analyzer.run_full_analysis()
