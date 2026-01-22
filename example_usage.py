"""
Example usage of the Company Intelligence System
Demonstrates various ways to use the analysis tool
"""

from company_intelligence import CompanyIntelligence
import os

def example_basic_usage():
    """Basic usage example"""
    print("="*60)
    print("EXAMPLE 1: Basic Usage")
    print("="*60)
    
    # Initialize with data file
    analyzer = CompanyIntelligence('champions_group_data.xlsx')
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    print("\nAnalysis complete! Check the generated files.")
    return results


def example_custom_clusters():
    """Example with custom cluster count"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Cluster Count")
    print("="*60)
    
    analyzer = CompanyIntelligence('champions_group_data.xlsx')
    analyzer.preprocess_data()
    
    # Use 5 clusters instead of optimal
    analyzer.perform_clustering(n_clusters=5)
    
    # Analyze results
    cluster_analysis = analyzer.analyze_clusters()
    patterns = analyzer.identify_patterns()
    
    # Generate insights
    insights = analyzer.generate_llm_insights(cluster_analysis, patterns)
    print(insights)
    
    return analyzer


def example_exclude_columns():
    """Example excluding specific columns"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Excluding Columns")
    print("="*60)
    
    analyzer = CompanyIntelligence('champions_group_data.xlsx')
    
    # Exclude ID columns or other non-analytical columns
    analyzer.preprocess_data(exclude_cols=['Company_ID', 'Internal_Code'])
    
    # Continue with analysis
    analyzer.perform_clustering()
    cluster_analysis = analyzer.analyze_clusters()
    
    return analyzer, cluster_analysis


def example_step_by_step():
    """Example of step-by-step analysis"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Step-by-Step Analysis")
    print("="*60)
    
    analyzer = CompanyIntelligence('champions_group_data.xlsx')
    
    # Step 1: Explore
    print("\n1. Exploring data...")
    analyzer.explore_data()
    
    # Step 2: Preprocess
    print("\n2. Preprocessing...")
    analyzer.preprocess_data()
    
    # Step 3: Find optimal clusters
    print("\n3. Finding optimal clusters...")
    optimal_k = analyzer.determine_optimal_clusters(max_k=8)
    
    # Step 4: Cluster
    print(f"\n4. Clustering with {optimal_k} clusters...")
    analyzer.perform_clustering(n_clusters=optimal_k)
    
    # Step 5: Analyze
    print("\n5. Analyzing clusters...")
    cluster_analysis = analyzer.analyze_clusters()
    
    # Step 6: Compare
    print("\n6. Comparing clusters...")
    comparison = analyzer.compare_clusters()
    
    # Step 7: Patterns
    print("\n7. Identifying patterns...")
    patterns = analyzer.identify_patterns()
    
    # Step 8: Visualize
    print("\n8. Creating visualizations...")
    analyzer.visualize_results()
    
    # Step 9: Generate report
    print("\n9. Generating report...")
    insights = analyzer.generate_llm_insights(cluster_analysis, patterns)
    report = analyzer.generate_report(cluster_analysis, patterns, insights)
    
    return analyzer


def example_specific_analysis():
    """Example focusing on specific features"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Feature-Specific Analysis")
    print("="*60)
    
    analyzer = CompanyIntelligence('champions_group_data.xlsx')
    analyzer.preprocess_data()
    analyzer.perform_clustering()
    
    # Compare specific feature across clusters
    # Replace 'Revenue' with an actual column name from your data
    numeric_cols = analyzer.df.select_dtypes(include=['number']).columns.tolist()
    if numeric_cols:
        feature = numeric_cols[0]  # Use first numeric column
        print(f"\nComparing '{feature}' across clusters:")
        comparison = analyzer.compare_clusters(feature=feature)
        print(comparison)
    
    return analyzer


if __name__ == "__main__":
    # Run examples
    print("\n" + "="*80)
    print("COMPANY INTELLIGENCE SYSTEM - USAGE EXAMPLES")
    print("="*80)
    
    # Uncomment the example you want to run:
    
    # Example 1: Basic usage (recommended to start)
    example_basic_usage()
    
    # Example 2: Custom clusters
    # example_custom_clusters()
    
    # Example 3: Exclude columns
    # example_exclude_columns()
    
    # Example 4: Step by step
    # example_step_by_step()
    
    # Example 5: Specific analysis
    # example_specific_analysis()
    
    print("\n" + "="*80)
    print("Examples complete!")
    print("="*80)
