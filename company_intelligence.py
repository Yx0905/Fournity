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
        """Perform initial data exploration"""
        print("\n" + "="*50)
        print("DATA EXPLORATION")
        print("="*50)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"\nMissing Values:")
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            'Missing Count': missing,
            'Missing %': missing_pct
        })
        print(missing_df[missing_df['Missing Count'] > 0])
        
        print(f"\nData Types:")
        print(self.df.dtypes)
        
        print(f"\nBasic Statistics:")
        print(self.df.describe())
        
        print(f"\nFirst few rows:")
        print(self.df.head())
        
        return self.df
    
    def preprocess_data(self, exclude_cols: List[str] = None):
        """
        Preprocess data for clustering
        
        Args:
            exclude_cols: List of column names to exclude from analysis
        """
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        if exclude_cols is None:
            exclude_cols = []
        
        # Start with a copy
        df_processed = self.df.copy()
        
        # Identify columns to use
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove excluded columns
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        # Handle missing values in numeric columns
        for col in numeric_cols:
            if df_processed[col].isnull().sum() > 0:
                # Use median if available, otherwise use 0
                fill_value = df_processed[col].median() if not pd.isna(df_processed[col].median()) else 0
                df_processed[col].fillna(fill_value, inplace=True)
        
        # Encode categorical variables
        encoded_features = []
        for col in categorical_cols:
            if df_processed[col].nunique() < 50:  # Only encode if reasonable number of categories
                le = LabelEncoder()
                df_processed[f'{col}_encoded'] = le.fit_transform(
                    df_processed[col].astype(str).fillna('Unknown')
                )
                self.label_encoders[col] = le
                encoded_features.append(f'{col}_encoded')
        
        # Combine numeric and encoded features
        feature_cols = numeric_cols + encoded_features
        
        # Remove columns with zero variance
        feature_cols = [col for col in feature_cols if df_processed[col].std() > 0]
        
        if len(feature_cols) == 0:
            raise ValueError("No valid features found after preprocessing. Check your data.")
        
        self.feature_names = feature_cols
        self.df_processed = df_processed[feature_cols].copy()
        
        # Scale features
        self.df_processed_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df_processed),
            columns=self.df_processed.columns,
            index=self.df_processed.index
        )
        
        print(f"\nProcessed {len(feature_cols)} features for analysis")
        print(f"Features: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"Features: {feature_cols}")
        
        return self.df_processed_scaled
    
    def determine_optimal_clusters(self, max_k: int = 10):
        """
        Determine optimal number of clusters using elbow method and silhouette score
        
        Args:
            max_k: Maximum number of clusters to test
        """
        print("\n" + "="*50)
        print("DETERMINING OPTIMAL CLUSTERS")
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
        
        # Find optimal k (elbow method + silhouette)
        optimal_k = k_range[np.argmax(silhouette_scores)]
        print(f"\nOptimal number of clusters: {optimal_k} (based on silhouette score)")
        
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
        """Analyze characteristics of each cluster"""
        print("\n" + "="*50)
        print("CLUSTER ANALYSIS")
        print("="*50)
        
        cluster_analysis = {}
        
        # Analyze numeric features
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'Cluster']
        
        for cluster_id in sorted(self.df['Cluster'].unique()):
            cluster_data = self.df[self.df['Cluster'] == cluster_id]
            cluster_analysis[cluster_id] = {
                'size': len(cluster_data),
                'percentage': (len(cluster_data) / len(self.df)) * 100,
                'numeric_stats': cluster_data[numeric_cols].describe().to_dict() if numeric_cols else {},
                'categorical_profiles': {}
            }
            
            # Analyze categorical features
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            for col in categorical_cols:
                if col != 'Cluster':
                    value_counts = cluster_data[col].value_counts(normalize=True).head(5)
                    cluster_analysis[cluster_id]['categorical_profiles'][col] = value_counts.to_dict()
            
            print(f"\nCluster {cluster_id}:")
            print(f"  Size: {cluster_analysis[cluster_id]['size']} companies ({cluster_analysis[cluster_id]['percentage']:.1f}%)")
        
        return cluster_analysis
    
    def compare_clusters(self, feature: str = None):
        """
        Compare clusters across different features
        
        Args:
            feature: Specific feature to compare (if None, compares key features)
        """
        print("\n" + "="*50)
        print("CLUSTER COMPARISON")
        print("="*50)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'Cluster']
        
        if not numeric_cols:
            print("Warning: No numeric columns available for comparison.")
            return pd.DataFrame()
        
        comparison = None
        if feature and feature in numeric_cols:
            # Compare specific feature
            comparison = self.df.groupby('Cluster')[feature].agg(['mean', 'std', 'min', 'max', 'count'])
            print(f"\nComparison of {feature} across clusters:")
            print(comparison)
        else:
            # Compare key numeric features
            print("\nKey feature comparison across clusters:")
            key_features = numeric_cols[:10]  # Top 10 numeric features
            if key_features:
                comparison = self.df.groupby('Cluster')[key_features].mean()
                print(comparison.round(2))
            else:
                comparison = pd.DataFrame()
        
        return comparison if comparison is not None else pd.DataFrame()
    
    def identify_patterns(self):
        """Identify notable patterns, strengths, risks, and anomalies"""
        print("\n" + "="*50)
        print("PATTERN IDENTIFICATION")
        print("="*50)
        
        patterns = {
            'outliers': [],
            'correlations': {},
            'cluster_differences': {},
            'anomalies': []
        }
        
        # Identify outliers using IQR method
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'Cluster']
        
        if numeric_cols:
            for col in numeric_cols[:10]:  # Check top 10 numeric features
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
        
        # Calculate correlations
        if len(numeric_cols) > 1:
            corr_matrix = self.df[numeric_cols[:10]].corr()
            patterns['correlations'] = corr_matrix.to_dict()
        
        # Identify cluster differences
        if numeric_cols:
            for cluster_id in sorted(self.df['Cluster'].unique()):
                cluster_data = self.df[self.df['Cluster'] == cluster_id]
                other_data = self.df[self.df['Cluster'] != cluster_id]
                
                if len(cluster_data) == 0 or len(other_data) == 0:
                    continue
                
                differences = {}
                for col in numeric_cols[:5]:  # Top 5 features
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
        Apply TF-IDF vectorization to text columns
        
        Args:
            text_columns: List of column names containing text data
            max_features: Maximum number of TF-IDF features to create
        """
        print("\n" + "="*50)
        print("TF-IDF FEATURE EXTRACTION")
        print("="*50)
        
        if text_columns is None:
            # Automatically detect text columns (object type with high cardinality)
            text_columns = []
            categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
            for col in categorical_cols:
                try:
                    # Check if column might contain text descriptions
                    sample_values = self.df[col].dropna().head(10)
                    if len(sample_values) == 0:
                        continue
                    avg_length = sample_values.astype(str).str.len().mean()
                    if not pd.isna(avg_length) and avg_length > 20:  # Likely text if average length > 20 chars
                        text_columns.append(col)
                except Exception:
                    continue
        
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
            
            print(f"Created {self.tfidf_features.shape[1]} TF-IDF features")
            
            # Show top terms
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            print(f"\nTop 10 TF-IDF terms: {list(feature_names[:10])}")
            
            return self.tfidf_features
        except Exception as e:
            print(f"Error in TF-IDF processing: {e}")
            return None
    
    def perform_chi_square_test(self, categorical_cols: List[str] = None):
        """
        Perform Chi-square test of independence between categorical variables and clusters
        
        Args:
            categorical_cols: List of categorical column names to test
        """
        print("\n" + "="*50)
        print("CHI-SQUARE TEST ANALYSIS")
        print("="*50)
        
        if self.clusters is None:
            print("Error: Must perform clustering first!")
            return None
        
        if categorical_cols is None:
            categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
            categorical_cols = [col for col in categorical_cols if col != 'Cluster']
        
        chi_square_results = {}
        
        for col in categorical_cols:
            try:
                # Create contingency table
                contingency = pd.crosstab(self.df[col], self.df['Cluster'])
                
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
        significant_vars = [col for col, result in chi_square_results.items() 
                          if result['is_significant']]
        print(f"\n{'='*50}")
        print(f"Summary: {len(significant_vars)}/{len(chi_square_results)} categorical variables "
              f"show significant association with clusters")
        
        return chi_square_results
    
    def apply_dimensionality_reduction(self, method: str = 'pca', n_components: int = 2):
        """
        Apply dimensionality reduction techniques
        
        Args:
            method: Method to use ('pca', 'tsne', 'umap', 'svd')
            n_components: Number of components for reduction
        """
        print("\n" + "="*50)
        print(f"DIMENSIONALITY REDUCTION: {method.upper()}")
        print("="*50)
        
        if self.df_processed_scaled is None:
            print("Error: Must preprocess data first!")
            return None
        
        reduction_results = {}
        
        if method.lower() == 'pca':
            reducer = PCA(n_components=n_components)
            reduced_data = reducer.fit_transform(self.df_processed_scaled)
            reduction_results['explained_variance'] = reducer.explained_variance_ratio_
            print(f"PCA: Explained variance ratio: {reduction_results['explained_variance']}")
            print(f"Total variance explained: {sum(reduction_results['explained_variance']):.2%}")
            
        elif method.lower() == 'svd' or method.lower() == 'truncated_svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
            reduced_data = reducer.fit_transform(self.df_processed_scaled)
            reduction_results['explained_variance'] = reducer.explained_variance_ratio_
            print(f"TruncatedSVD: Explained variance ratio: {reduction_results['explained_variance']}")
            print(f"Total variance explained: {sum(reduction_results['explained_variance']):.2%}")
            
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
        Train logistic regression model to predict cluster membership
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        print("\n" + "="*50)
        print("LOGISTIC REGRESSION - CLUSTER PREDICTION")
        print("="*50)
        
        if self.clusters is None:
            print("Error: Must perform clustering first!")
            return None
        
        if self.df_processed_scaled is None:
            print("Error: Must preprocess data first!")
            return None
        
        # Prepare features and target
        X = self.df_processed_scaled.values
        y = self.clusters
        
        # Combine with TF-IDF features if available
        if self.tfidf_features is not None:
            print("Combining features with TF-IDF features...")
            X = np.hstack([X, self.tfidf_features.values])
        
        # Split data
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
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
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
        
        # Feature importance (coefficients)
        if self.logistic_model.n_features_in_ < 20:
            print("\nFeature Importance (Top coefficients per cluster):")
            feature_names = list(self.feature_names)
            if self.tfidf_features is not None:
                feature_names += list(self.tfidf_features.columns)
            
            for i, cluster_id in enumerate(sorted(set(y))):
                print(f"\nCluster {cluster_id}:")
                coefs = self.logistic_model.coef_[i]
                top_indices = np.argsort(np.abs(coefs))[-5:][::-1]
                for idx in top_indices:
                    if idx < len(feature_names):
                        print(f"  {feature_names[idx]}: {coefs[idx]:.4f}")
        
        results = {
            'model': self.logistic_model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'classification_report': classification_report(y_test, y_test_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_test_pred)
        }
        
        return results
    
    def train_linear_regression(self, target_feature: str = None, test_size: float = 0.2, random_state: int = 42):
        """
        Train linear regression model to predict a target feature
        
        Args:
            target_feature: Name of the target feature to predict (if None, uses first numeric column)
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
        """
        print("\n" + "="*50)
        print("LINEAR REGRESSION - FEATURE PREDICTION")
        print("="*50)
        
        if self.df_processed_scaled is None:
            print("Error: Must preprocess data first!")
            return None
        
        # Select target feature
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'Cluster']
        
        if not numeric_cols:
            print("Error: No numeric features available for linear regression!")
            return None
        
        if target_feature is None:
            target_feature = numeric_cols[0]
            print(f"No target feature specified. Using first numeric column: {target_feature}")
        elif target_feature not in numeric_cols:
            print(f"Warning: {target_feature} not found. Using first numeric column: {numeric_cols[0]}")
            target_feature = numeric_cols[0]
        
        # Prepare features and target
        X = self.df_processed_scaled.values
        y = self.df[target_feature].values
        
        # Combine with TF-IDF features if available
        if self.tfidf_features is not None:
            print("Combining features with TF-IDF features...")
            X = np.hstack([X, self.tfidf_features.values])
        
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
        
        # 2. PCA visualization (2D)
        if len(self.feature_names) > 2:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(self.df_processed_scaled)
            
            fig, ax = plt.subplots(figsize=(12, 8))
            scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], 
                               c=self.clusters, cmap='viridis', alpha=0.6)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            ax.set_title('Company Segments (PCA Visualization)')
            plt.colorbar(scatter, label='Cluster')
            plt.tight_layout()
            plt.savefig('pca_clusters.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Saved pca_clusters.png")
        
        # 3. Feature comparison across clusters
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'Cluster']
        
        if numeric_cols:
            # Select top 6 features for comparison
            top_features = numeric_cols[:6]
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            axes = axes.flatten()
            
            for idx, feature in enumerate(top_features):
                self.df.boxplot(column=feature, by='Cluster', ax=axes[idx])
                axes[idx].set_title(f'{feature} by Cluster')
                axes[idx].set_xlabel('Cluster')
            
            plt.suptitle('Feature Comparison Across Clusters', y=1.02)
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
        """Generate a comprehensive report"""
        print("\n" + "="*50)
        print("GENERATING REPORT")
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
        
        # Add detailed cluster analysis
        report.append("="*80)
        report.append("DETAILED CLUSTER ANALYSIS")
        report.append("="*80)
        
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
                        report.append(f"    Mean: {stats['mean']:.2f}")
                        report.append(f"    Std: {stats['std']:.2f}")
                        report.append(f"    Min: {stats['min']:.2f}")
                        report.append(f"    Max: {stats['max']:.2f}")
                report.append("")
            
            if info['categorical_profiles']:
                report.append("Categorical Profiles:")
                for feature, profile in list(info['categorical_profiles'].items())[:5]:
                    report.append(f"  {feature}:")
                    for value, pct in list(profile.items())[:5]:
                        report.append(f"    {value}: {pct:.1%}")
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
        print("STARTING FULL COMPANY INTELLIGENCE ANALYSIS")
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
        
        # 4. Analyze clusters
        cluster_analysis = self.analyze_clusters()
        
        # 5. Compare clusters
        comparison = self.compare_clusters()
        
        # 6. Identify patterns
        patterns = self.identify_patterns()
        
        # 7. Apply TF-IDF (if text columns exist)
        tfidf_features = self.apply_tfidf()
        
        # 8. Perform Chi-square tests
        chi_square_results = self.perform_chi_square_test()
        
        # 9. Apply additional dimensionality reduction
        dim_reduction_results = {}
        for method in ['pca', 'svd']:
            result = self.apply_dimensionality_reduction(method=method, n_components=3)
            if result:
                dim_reduction_results[method] = result
        
        # 10. Train logistic regression
        lr_results = self.train_logistic_regression()
        
        # 11. Train linear regression
        linear_reg_results = self.train_linear_regression()
        
        # 12. Generate insights
        insights = self.generate_llm_insights(cluster_analysis, patterns)
        
        # 13. Visualize
        self.visualize_results()
        self.visualize_dimensionality_reduction(['pca', 'svd'])
        
        # 14. Add new results to patterns
        if chi_square_results:
            patterns['chi_square'] = chi_square_results
        if lr_results:
            patterns['logistic_regression'] = {
                'train_accuracy': lr_results['train_accuracy'],
                'test_accuracy': lr_results['test_accuracy']
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
        print("  - dimensionality_reduction.png (multiple reduction methods)")
        print("  - linear_regression_results.png (linear regression visualization)")
        
        # Build return dictionary
        return_dict = {
            'cluster_analysis': cluster_analysis,
            'patterns': patterns,
            'insights': insights,
            'report': report
        }
        
        # Add new analysis results
        if chi_square_results:
            return_dict['chi_square_results'] = chi_square_results
        if lr_results:
            return_dict['logistic_regression'] = lr_results
        if linear_reg_results:
            return_dict['linear_regression'] = linear_reg_results
        if dim_reduction_results:
            return_dict['dimensionality_reduction'] = dim_reduction_results
        if tfidf_features is not None:
            return_dict['tfidf_features'] = tfidf_features
        
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
