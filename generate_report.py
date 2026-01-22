#!/usr/bin/env python3
"""
Quick script to generate the company intelligence report
Run this file to automatically generate the complete analysis report
"""

from company_intelligence import CompanyIntelligence
import os
from dotenv import load_dotenv

def main():
    """Generate the company intelligence report"""
    
    print("="*60)
    print("Company Intelligence Analysis - Report Generator")
    print("="*60)
    print()
    
    # Load environment variables (for API key - optional)
    load_dotenv()
    
    # Initialize the analyzer
    data_path = 'champions_group_data.xlsx'
    api_key = os.getenv('OPENAI_API_KEY')  # Optional: for LLM insights
    
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found: {data_path}")
        print("Please make sure the file exists in the current directory.")
        return
    
    print(f"Loading data from: {data_path}")
    print()
    
    try:
        # Initialize analyzer (automatically loads data and filters inactive companies)
        analyzer = CompanyIntelligence(data_path, api_key=api_key)
        
        # Run complete analysis pipeline (generates report automatically)
        print("Starting full analysis pipeline...")
        print()
        results = analyzer.run_full_analysis()
        
        print()
        print("="*60)
        print("✅ Report Generated Successfully!")
        print("="*60)
        print()
        print("Generated Files:")
        print("  📄 company_intelligence_report.txt - Comprehensive analysis report")
        print("  📊 companies_with_segments.csv - Dataset with cluster labels")
        print("  📈 PNG files - Visualization charts")
        print()
        print(f"Clusters identified: {len(set(analyzer.clusters)) if analyzer.clusters is not None else 'N/A'}")
        print()
        print("You can now open 'company_intelligence_report.txt' to view the full analysis.")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
