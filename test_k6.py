"""Test with K=6 clusters"""
from company_intelligence import CompanyIntelligence

analyzer = CompanyIntelligence('champions_group_data.xlsx')
analyzer.preprocess_data()
analyzer.perform_clustering(n_clusters=6)
cluster_analysis = analyzer.analyze_clusters()

print("\n" + "="*80)
print("K=6 CLUSTER ANALYSIS")
print("="*80)

for cluster_id in sorted(cluster_analysis.keys()):
    info = cluster_analysis[cluster_id]
    print(f"\nCluster {cluster_id}:")
    print(f"  Size: {info['size']} companies ({info['percentage']:.1f}%)")
    if info['numeric_stats']:
        print("  Key Metrics:")
        for feature, stats in info['numeric_stats'].items():
            if 'mean' in stats:
                print(f"    {feature}: {stats['mean']:,.2f}")
    if info['categorical_profiles']:
        for feature, profile in info['categorical_profiles'].items():
            top = sorted(profile.items(), key=lambda x: x[1], reverse=True)[0]
            print(f"  Top {feature}: {top[0]} ({top[1]:.1%})")
