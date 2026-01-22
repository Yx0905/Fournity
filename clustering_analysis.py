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
from sklearn.metrics import silhouette_score, calinski_harabasz_score
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
    
    def __init__(self, data_file='champions_group_processed.xlsx'):
        self.data_file = data_file
        self.df = None
        self.df_processed = None
        self.feature_info = {}
        self.results = {}
        self.cluster_labels = None
        self.best_model = None
        self.best_params = {}
        
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
                # Update scaled data to match
                self.df_scaled = self.df_scaled.loc[self.df_work.index]
                # Update sample count
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
            print("-> Selected: Gower Distance (FAMD preprocessing)")
        elif num_ratio > 0.8:
            self.feature_info['distance_metric'] = 'Euclidean/Cosine'
            self.feature_info['use_famd'] = False
            print("-> Selected: Euclidean/Cosine Distance")
        else:
            self.feature_info['distance_metric'] = 'Mixed'
            self.feature_info['use_famd'] = True
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
                    print(f"  -> Dropping {col}")
                    numerical_cols.remove(col)
                    self.df_work = self.df_work.drop(columns=[col])
            else:
                print("No highly correlated numerical pairs found")
        
        # Theil's U for categoricals (simplified - using Cramér's V as proxy)
        if len(categorical_cols) > 1 and THEIL_AVAILABLE:
            print("\nChecking categorical redundancy...")
            # Simplified approach: drop low cardinality categoricals if they're redundant
            for col in categorical_cols:
                if self.df_work[col].nunique() == 1:
                    print(f"  -> Dropping {col} (single value)")
                    categorical_cols.remove(col)
                    self.df_work = self.df_work.drop(columns=[col])
        
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
            self.df_scaled[numerical_cols] = scaler.fit_transform(self.df_work[numerical_cols])
        
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
            # Prepare data for FAMD
            famd_data = self.df_work[numerical_cols + categorical_cols].copy()
            
            # Ensure no NaN values remain
            # Fill numerical NaN with median
            for col in numerical_cols:
                if col in famd_data.columns and famd_data[col].isnull().any():
                    famd_data[col] = famd_data[col].fillna(famd_data[col].median())
            
            # Fill categorical NaN with mode or 'Unknown'
            for col in categorical_cols:
                if col in famd_data.columns and famd_data[col].isnull().any():
                    mode_val = famd_data[col].mode()[0] if len(famd_data[col].mode()) > 0 else 'Unknown'
                    famd_data[col] = famd_data[col].fillna(mode_val)
            
            # Double check for any remaining NaN
            if famd_data.isnull().any().any():
                print("Warning: Still have NaN values. Dropping rows with NaN...")
                rows_before = len(famd_data)
                famd_data = famd_data.dropna()
                rows_after = len(famd_data)
                print(f"  -> Dropped {rows_before - rows_after} rows with NaN")
                # Update df_work and df_scaled to match
                self.df_work = self.df_work.loc[famd_data.index].copy()
                self.df_scaled = self.df_scaled.loc[famd_data.index].copy()
            
            # Verify no NaN before fitting
            if famd_data.isnull().any().any():
                print("ERROR: NaN values still present. Columns with NaN:")
                print(famd_data.isnull().sum()[famd_data.isnull().sum() > 0])
                raise ValueError("Cannot proceed with NaN values in FAMD input")
            
            # Update sample count if rows were dropped
            if len(famd_data) != self.feature_info['n_samples']:
                self.feature_info['n_samples'] = len(famd_data)
                print(f"Updated sample count: {len(famd_data)}")
            
            famd = FAMD(n_components=min(20, len(numerical_cols) + len(categorical_cols) - 1))
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
            
        else:
            print("Using PCA...")
            # Use only numerical features for PCA
            # Ensure no NaN in scaled data
            pca_data = self.df_scaled[numerical_cols].copy()
            if pca_data.isnull().any().any():
                print("Warning: NaN in scaled data. Filling with 0...")
                pca_data = pca_data.fillna(0)
            
            pca = PCA(n_components=min(20, len(numerical_cols)))
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
        
        return self.df_reduced
    
    def phase3_clustering_loop(self):
        """
        Phase 3: Iterative Clustering Loop
        """
        print("\n" + "="*80)
        print("PHASE 3: ITERATIVE CLUSTERING LOOP")
        print("="*80)
        
        n_samples = len(self.df_reduced)
        
        # Decision logic for clustering algorithm
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
        
        # Determine k range
        max_k = min(15, n_samples // 100)  # Reasonable upper bound
        k_range = range(2, max(3, max_k))
        
        param_grid = {
            'n_clusters': list(k_range),
            'random_state': [42],
            'n_init': [10]
        }
        
        if cluster_class == MiniBatchKMeans:
            param_grid['batch_size'] = [256, 512]
        
        best_score = -1
        best_params = None
        results_grid = []
        
        print(f"Testing {len(list(ParameterGrid(param_grid)))} parameter combinations...")
        
        for params in ParameterGrid(param_grid):
            try:
                model = cluster_class(**params)
                labels = model.fit_predict(self.df_reduced)
                
                # Skip if only one cluster
                if len(np.unique(labels)) < 2:
                    continue
                
                # Step 3.2: Internal Validation
                sil_score = silhouette_score(self.df_reduced, labels)
                ch_score = calinski_harabasz_score(self.df_reduced, labels)
                
                results_grid.append({
                    'params': params,
                    'silhouette': sil_score,
                    'calinski_harabasz': ch_score,
                    'n_clusters': params['n_clusters']
                })
                
                # Combined score (weighted)
                combined_score = 0.6 * sil_score + 0.4 * (ch_score / 1000)  # Normalize CH score
                
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
        print(f"Best silhouette score: {results_grid[results_grid.index(next(r for r in results_grid if r['params'] == best_params))]['silhouette']:.3f}")
        
        self.best_params = best_params
        
        # Step 3.3: Stability Testing
        print("\nStep 3.3: Stability Testing...")
        n_folds = 5
        stability_scores = []
        
        # Store original centers
        centers_original = self.best_model.cluster_centers_
        
        for fold in range(n_folds):
            # Shuffle data with different random seeds
            np.random.seed(42 + fold)
            indices = np.random.permutation(len(self.df_reduced))
            df_shuffled = self.df_reduced.iloc[indices].reset_index(drop=True)
            
            # Fit model with same parameters
            fold_params = best_params.copy()
            model_fold = cluster_class(**fold_params)
            labels_fold = model_fold.fit_predict(df_shuffled)
            
            # Calculate cluster centers
            centers_fold = model_fold.cluster_centers_
            
            # Calculate center shift (normalized)
            if len(centers_fold) == len(centers_original):
                # Align clusters by finding best matching
                distances = cdist(centers_fold, centers_original)
                # Find minimum distance matching
                shift = np.mean([np.min(distances[i, :]) for i in range(len(centers_fold))])
                stability_scores.append(shift)
        
        avg_shift = np.mean(stability_scores)
        stability_threshold = 0.1  # 10% shift threshold
        
        print(f"Average cluster center shift: {avg_shift:.4f}")
        if avg_shift > stability_threshold:
            print(f"WARNING: Model may be unstable (shift > {stability_threshold})")
            print("Consider increasing sparsity penalty or using different parameters")
        else:
            print("Model stability: PASSED")
        
        self.results['stability'] = {
            'avg_shift': avg_shift,
            'is_stable': avg_shift <= stability_threshold
        }
        self.results['grid_results'] = results_grid
        
        # Validation Gate
        best_result = next(r for r in results_grid if r['params'] == best_params)
        sil_score = best_result['silhouette']
        
        print(f"\nValidation Gate: Silhouette Score = {sil_score:.3f}")
        if sil_score > 0.5:
            print("✓ Validation PASSED - Proceeding to Profiling")
            self.results['validation_passed'] = True
        else:
            print("✗ Validation FAILED - Consider re-running with Sparse Autoencoder")
            self.results['validation_passed'] = False
        
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
            report_lines.append(f"Calinski-Harabasz Score: {best_result['calinski_harabasz']:.4f}")
        
        report_lines.append(f"Stability Check: {'PASSED' if self.results.get('stability', {}).get('is_stable', False) else 'FAILED'}")
        if 'stability' in self.results:
            report_lines.append(f"  Average cluster center shift: {self.results['stability']['avg_shift']:.4f}")
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
    
    def run_full_analysis(self):
        """Run the complete 4-phase analysis"""
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


if __name__ == "__main__":
    # Run the analysis
    analyzer = LatentSparseClustering(data_file='champions_group_processed.xlsx')
    results = analyzer.run_full_analysis()
