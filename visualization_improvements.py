"""
Enhanced Visualization Improvements for Company Intelligence System
This module provides improved graphics for PCA clusters and feature comparisons
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set modern style (with fallback for different matplotlib versions)
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('ggplot')
sns.set_palette("husl")

def create_enhanced_pca_visualization(df_processed_scaled, clusters, feature_names, 
                                     save_path='pca_clusters_enhanced.png'):
    """
    Create an enhanced PCA visualization with better graphics
    
    Improvements:
    - Better color scheme with distinct cluster colors
    - Cluster labels and centroids
    - Enhanced feature importance visualization
    - Better legends and annotations
    - Statistical information
    """
    # Perform PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(df_processed_scaled)
    
    # Create figure with better layout
    fig = plt.figure(figsize=(20, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Get unique clusters and assign colors
    unique_clusters = sorted(set(clusters))
    n_clusters = len(unique_clusters)
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    
    # ===== MAIN PCA SCATTER PLOT (Top Left, spans 2 columns) =====
    ax_main = fig.add_subplot(gs[0, :2])
    
    # Plot each cluster separately for better control
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        cluster_size = mask.sum()
        ax_main.scatter(
            pca_result[mask, 0], pca_result[mask, 1],
            c=[cluster_colors[cluster_id]],
            label=f'Cluster {cluster_id} (n={cluster_size})',
            alpha=0.6,
            s=50,
            edgecolors='white',
            linewidths=0.5
        )
        
        # Add cluster centroid
        centroid_x = pca_result[mask, 0].mean()
        centroid_y = pca_result[mask, 1].mean()
        ax_main.scatter(
            centroid_x, centroid_y,
            c='black',
            marker='X',
            s=200,
            edgecolors='white',
            linewidths=2,
            zorder=10,
            label=f'Centroid {cluster_id}' if cluster_id == unique_clusters[0] else ""
        )
        # Add cluster label near centroid
        ax_main.annotate(
            f'C{cluster_id}',
            (centroid_x, centroid_y),
            fontsize=12,
            fontweight='bold',
            ha='center',
            va='center',
            color='white',
            zorder=11
        )
    
    ax_main.set_xlabel(
        f'First Principal Component (PC1)\nExplained Variance: {pca.explained_variance_ratio_[0]:.2%}',
        fontsize=12,
        fontweight='bold'
    )
    ax_main.set_ylabel(
        f'Second Principal Component (PC2)\nExplained Variance: {pca.explained_variance_ratio_[1]:.2%}',
        fontsize=12,
        fontweight='bold'
    )
    ax_main.set_title(
        'Enhanced PCA Visualization: Company Segments\n(Colored by Cluster, X = Centroids)',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax_main.grid(True, alpha=0.3, linestyle='--')
    ax_main.legend(loc='upper right', framealpha=0.9, fontsize=9)
    
    # ===== FEATURE IMPORTANCE - PC1 (Top Right) =====
    ax_pc1 = fig.add_subplot(gs[0, 2])
    
    all_feature_names = df_processed_scaled.columns.tolist()
    components_df = pd.DataFrame(
        pca.components_.T,
        columns=['PC1', 'PC2'],
        index=all_feature_names
    )
    
    # Top 10 features for PC1
    top_pc1 = components_df.nlargest(10, 'PC1')['PC1']
    colors_pc1 = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_pc1.values]
    
    y_pos = np.arange(len(top_pc1))
    ax_pc1.barh(y_pos, top_pc1.values, color=colors_pc1, alpha=0.7, edgecolor='black')
    ax_pc1.set_yticks(y_pos)
    ax_pc1.set_yticklabels([f[:30] + '...' if len(f) > 30 else f for f in top_pc1.index], 
                           fontsize=8)
    ax_pc1.set_xlabel('PC1 Loading', fontweight='bold')
    ax_pc1.set_title('Top 10 Features\nContributing to PC1', fontweight='bold', fontsize=11)
    ax_pc1.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax_pc1.grid(True, alpha=0.3, axis='x')
    
    # ===== FEATURE IMPORTANCE - PC2 (Bottom Left) =====
    ax_pc2 = fig.add_subplot(gs[1, 0])
    
    top_pc2 = components_df.nlargest(10, 'PC2')['PC2']
    colors_pc2 = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_pc2.values]
    
    y_pos = np.arange(len(top_pc2))
    ax_pc2.barh(y_pos, top_pc2.values, color=colors_pc2, alpha=0.7, edgecolor='black')
    ax_pc2.set_yticks(y_pos)
    ax_pc2.set_yticklabels([f[:30] + '...' if len(f) > 30 else f for f in top_pc2.index], 
                           fontsize=8)
    ax_pc2.set_xlabel('PC2 Loading', fontweight='bold')
    ax_pc2.set_title('Top 10 Features\nContributing to PC2', fontweight='bold', fontsize=11)
    ax_pc2.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax_pc2.grid(True, alpha=0.3, axis='x')
    
    # ===== CLUSTER STATISTICS (Bottom Middle) =====
    ax_stats = fig.add_subplot(gs[1, 1])
    ax_stats.axis('off')
    
    # Calculate cluster statistics
    stats_text = "CLUSTER STATISTICS\n" + "="*30 + "\n\n"
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        cluster_size = mask.sum()
        percentage = (cluster_size / len(clusters)) * 100
        
        # Calculate mean distance from centroid
        cluster_data = pca_result[mask]
        centroid = cluster_data.mean(axis=0)
        distances = np.sqrt(((cluster_data - centroid) ** 2).sum(axis=1))
        mean_dist = distances.mean()
        
        stats_text += f"Cluster {cluster_id}:\n"
        stats_text += f"  Size: {cluster_size} ({percentage:.1f}%)\n"
        stats_text += f"  Mean Dist: {mean_dist:.2f}\n"
        stats_text += f"  PC1 Range: [{pca_result[mask, 0].min():.2f}, {pca_result[mask, 0].max():.2f}]\n"
        stats_text += f"  PC2 Range: [{pca_result[mask, 1].min():.2f}, {pca_result[mask, 1].max():.2f}]\n\n"
    
    ax_stats.text(0.1, 0.9, stats_text, transform=ax_stats.transAxes,
                  fontsize=9, verticalalignment='top', family='monospace',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # ===== EXPLAINED VARIANCE (Bottom Right) =====
    ax_var = fig.add_subplot(gs[1, 2])
    
    # Calculate cumulative variance for more components
    pca_full = PCA()
    pca_full.fit(df_processed_scaled)
    n_components_show = min(10, len(pca_full.explained_variance_ratio_))
    
    explained_var = pca_full.explained_variance_ratio_[:n_components_show]
    cumulative_var = np.cumsum(explained_var)
    
    x_pos = np.arange(1, n_components_show + 1)
    ax_var.bar(x_pos, explained_var, alpha=0.7, color='steelblue', 
               label='Individual', edgecolor='black')
    ax_var.plot(x_pos, cumulative_var, 'ro-', linewidth=2, markersize=8,
                label='Cumulative', color='darkred')
    ax_var.set_xlabel('Principal Component', fontweight='bold')
    ax_var.set_ylabel('Explained Variance Ratio', fontweight='bold')
    ax_var.set_title('PCA Explained Variance\n(First 10 Components)', 
                     fontweight='bold', fontsize=11)
    ax_var.legend()
    ax_var.grid(True, alpha=0.3, axis='y')
    ax_var.set_xticks(x_pos)
    
    plt.suptitle('Enhanced PCA Analysis: Company Segmentation', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved enhanced PCA visualization: {save_path}")


def create_enhanced_feature_comparison(df, clusters, numeric_cols, 
                                      save_path='feature_comparison_enhanced.png',
                                      max_features=6):
    """
    Create an enhanced feature comparison visualization
    
    Improvements:
    - Violin plots + boxplots for better distribution visualization
    - Color-coded by cluster
    - Statistical annotations (mean, median, significance)
    - Better layout and styling
    - Log scale handling with clear indication
    """
    # Select top features
    top_features = numeric_cols[:max_features] if len(numeric_cols) > max_features else numeric_cols
    
    if not top_features:
        print("No numeric features available for comparison")
        return
    
    n_features = len(top_features)
    n_cols = min(3, n_features)
    n_rows = (n_features + n_cols - 1) // n_cols
    
    # Create figure with better spacing
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    if n_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
    
    # Get unique clusters and colors
    unique_clusters = sorted(set(clusters))
    n_clusters = len(unique_clusters)
    colors = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    cluster_colors = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    
    for idx, feature in enumerate(top_features):
        ax = axes[idx]
        
        # Prepare data
        feature_data = df[feature].copy()
        
        # Check if log transformation needed
        valid_data = feature_data[feature_data > 0]
        if len(valid_data) > 0:
            data_range = valid_data.max() / (valid_data.min() + 1e-10)
            use_log = data_range > 100
        else:
            use_log = False
        
        # Create data for plotting
        plot_data = []
        plot_labels = []
        plot_colors_list = []
        
        for cluster_id in unique_clusters:
            mask = clusters == cluster_id
            cluster_data = feature_data[mask].dropna()
            
            if len(cluster_data) > 0:
                if use_log:
                    cluster_data = np.log10(cluster_data + 1)
                    y_label = f'Log10({feature} + 1)'
                else:
                    y_label = feature
                
                plot_data.append(cluster_data.values)
                plot_labels.append(f'C{cluster_id}\n(n={len(cluster_data)})')
                plot_colors_list.append(cluster_colors[cluster_id])
        
        # Create violin plot with boxplot overlay
        if plot_data:
            parts = ax.violinplot(plot_data, positions=range(len(plot_data)), 
                                 showmeans=True, showmedians=True, widths=0.7)
            
            # Customize violin plot colors
            for pc, color in zip(parts['bodies'], plot_colors_list):
                pc.set_facecolor(color)
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # Customize other elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmeans', 'cmedians'):
                if partname in parts:
                    parts[partname].set_color('black')
                    parts[partname].set_linewidth(1.5)
            
            # Add boxplot overlay for quartiles
            bp = ax.boxplot(plot_data, positions=range(len(plot_data)),
                           widths=0.3, patch_artist=True, 
                           showfliers=False, zorder=10)
            
            for patch, color in zip(bp['boxes'], plot_colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
                patch.set_edgecolor('black')
                patch.set_linewidth(1.5)
            
            # Add mean markers
            for i, data in enumerate(plot_data):
                mean_val = np.mean(data)
                median_val = np.median(data)
                ax.scatter(i, mean_val, color='red', marker='D', s=100, 
                          zorder=15, label='Mean' if i == 0 else '', edgecolors='white', linewidths=1)
                ax.scatter(i, median_val, color='blue', marker='s', s=80, 
                          zorder=15, label='Median' if i == 0 else '', edgecolors='white', linewidths=1)
            
            # Statistical annotations
            if len(plot_data) >= 2:
                # Perform ANOVA to check if differences are significant
                try:
                    f_stat, p_value = stats.f_oneway(*plot_data)
                    if p_value < 0.001:
                        sig_text = "***"
                    elif p_value < 0.01:
                        sig_text = "**"
                    elif p_value < 0.05:
                        sig_text = "*"
                    else:
                        sig_text = "ns"
                    
                    ax.text(0.02, 0.98, f'ANOVA: p={p_value:.4f} {sig_text}',
                           transform=ax.transAxes, fontsize=9,
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
                except:
                    pass
            
            ax.set_xticks(range(len(plot_labels)))
            ax.set_xticklabels(plot_labels, fontsize=10, fontweight='bold')
            ax.set_ylabel(y_label if use_log else feature, fontsize=11, fontweight='bold')
            ax.set_title(
                f'{feature}\n{"(Log Scale)" if use_log else "(Linear Scale)"}',
                fontsize=12, fontweight='bold', pad=10
            )
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')
            ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
    
    # Hide extra subplots
    for idx in range(n_features, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle(
        'Enhanced Feature Comparison Across Clusters\n(Violin + Boxplot, Mean=◆, Median=■)',
        fontsize=14, fontweight='bold', y=0.995
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved enhanced feature comparison: {save_path}")


def create_cluster_heatmap_comparison(df, clusters, numeric_cols, 
                                     save_path='cluster_heatmap_enhanced.png'):
    """
    Create a heatmap comparing cluster characteristics
    """
    unique_clusters = sorted(set(clusters))
    
    # Calculate mean values for each cluster and feature
    cluster_means = []
    for cluster_id in unique_clusters:
        mask = clusters == cluster_id
        cluster_data = df.loc[mask, numeric_cols].mean()
        cluster_means.append(cluster_data.values)
    
    cluster_means_df = pd.DataFrame(
        cluster_means,
        index=[f'Cluster {c}' for c in unique_clusters],
        columns=numeric_cols[:10]  # Limit to top 10 features
    )
    
    # Normalize for better visualization
    cluster_means_normalized = cluster_means_df.div(cluster_means_df.max(axis=0), axis=1)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, max(6, len(unique_clusters) * 0.8)))
    
    sns.heatmap(
        cluster_means_normalized,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn_r',
        center=0.5,
        cbar_kws={'label': 'Normalized Value (0-1)'},
        linewidths=0.5,
        linecolor='black',
        ax=ax
    )
    
    ax.set_title('Cluster Characteristics Heatmap\n(Normalized Mean Values)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Clusters', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved cluster heatmap: {save_path}")
