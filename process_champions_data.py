#!/usr/bin/env python3
"""
CHAMPIONS GROUP Dataset Processing Script
Processes company data to generate insights and derived metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime

def process_champions_data():
    """
    Main function to process the CHAMPIONS GROUP dataset
    """
    # Read the Excel file
    print("Reading Excel file...")
    df = pd.read_excel('champions_group_data.xlsx')
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Original number of companies: {len(df)}")
    
    # Store original count
    original_count = len(df)
    
    # Remove duplicate rows
    print("\nRemoving duplicate rows...")
    df = df.drop_duplicates()
    duplicates_removed = original_count - len(df)
    print(f"Removed {duplicates_removed} duplicate rows")
    
    # Identify column names (case-insensitive matching)
    # We need to find columns that match our requirements
    columns_to_keep = []
    column_mapping = {}
    
    # Common variations of column names
    revenue_cols = [col for col in df.columns if 'revenue' in str(col).lower() and 'usd' in str(col).lower()]
    market_value_cols = [col for col in df.columns if 'market' in str(col).lower() and 'value' in str(col).lower() and 'usd' in str(col).lower()]
    it_spend_cols = [col for col in df.columns if 'it' in str(col).lower() and 'spend' in str(col).lower()]
    employees_cols = [col for col in df.columns if 'employees' in str(col).lower() and 'total' in str(col).lower()]
    sites_cols = [col for col in df.columns if 'sites' in str(col).lower() and 'company' in str(col).lower()]
    servers_cols = [col for col in df.columns if 'servers' in str(col).lower() and 'no' in str(col).lower()]
    storage_cols = [col for col in df.columns if 'storage' in str(col).lower() and 'devices' in str(col).lower()]
    year_found_cols = [col for col in df.columns if 'year' in str(col).lower() and 'found' in str(col).lower()]
    
    # Geographic columns (for geographic dispersion)
    geo_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['country', 'region', 'geographic', 'location'])]
    
    print("\nColumn identification:")
    print(f"Revenue columns found: {revenue_cols}")
    print(f"Market Value columns found: {market_value_cols}")
    print(f"IT Spend columns found: {it_spend_cols}")
    print(f"Employees columns found: {employees_cols}")
    print(f"Company Sites columns found: {sites_cols}")
    print(f"Servers columns found: {servers_cols}")
    print(f"Storage columns found: {storage_cols}")
    print(f"Year Found columns found: {year_found_cols}")
    print(f"Geographic columns found: {geo_cols}")
    
    # Select the first matching column for each category
    revenue_col = revenue_cols[0] if revenue_cols else None
    market_value_col = market_value_cols[0] if market_value_cols else None
    it_spend_col = it_spend_cols[0] if it_spend_cols else None
    employees_col = employees_cols[0] if employees_cols else None
    sites_col = sites_cols[0] if sites_cols else None
    servers_col = servers_cols[0] if servers_cols else None
    storage_col = storage_cols[0] if storage_cols else None
    year_found_col = year_found_cols[0] if year_found_cols else None
    
    # If exact matches not found, try broader search
    if not revenue_col:
        revenue_col = [col for col in df.columns if 'revenue' in str(col).lower()][0] if [col for col in df.columns if 'revenue' in str(col).lower()] else None
    if not market_value_col:
        market_value_col = [col for col in df.columns if 'market' in str(col).lower() and 'value' in str(col).lower()][0] if [col for col in df.columns if 'market' in str(col).lower() and 'value' in str(col).lower()] else None
    if not employees_col:
        employees_col = [col for col in df.columns if 'employees' in str(col).lower()][0] if [col for col in df.columns if 'employees' in str(col).lower()] else None
    
    # Create a subset with required columns
    required_cols = []
    if revenue_col:
        required_cols.append(revenue_col)
        column_mapping['Revenue (USD)'] = revenue_col
    if market_value_col:
        required_cols.append(market_value_col)
        column_mapping['Market Value (USD)'] = market_value_col
    if it_spend_col:
        required_cols.append(it_spend_col)
        column_mapping['IT Spend'] = it_spend_col
    if employees_col:
        required_cols.append(employees_col)
        column_mapping['Employees Total'] = employees_col
    if sites_col:
        required_cols.append(sites_col)
        column_mapping['Company Sites'] = sites_col
    if servers_col:
        required_cols.append(servers_col)
        column_mapping['No. of Servers'] = servers_col
    if storage_col:
        required_cols.append(storage_col)
        column_mapping['No. of Storage Devices'] = storage_col
    if year_found_col:
        required_cols.append(year_found_col)
        column_mapping['Year Found'] = year_found_col
    
    # Add geographic columns for dispersion calculation
    for geo_col in geo_cols[:3]:  # Take up to 3 geographic columns
        if geo_col not in required_cols:
            required_cols.append(geo_col)
    
    # Also keep company identifier if available
    id_cols = [col for col in df.columns if any(term in str(col).lower() for term in ['company name', 'company', 'name'])]
    if not id_cols:
        id_cols = [col for col in df.columns if 'id' in str(col).lower()]
    if id_cols:
        company_id_col = id_cols[0]
        if company_id_col not in required_cols:
            required_cols.insert(0, company_id_col)
            column_mapping['Company Name'] = company_id_col
    
    print(f"\nSelected columns: {required_cols}")
    
    # Create working dataframe with selected columns
    df_work = df[required_cols].copy()
    
    # Rename columns to standard names
    rename_dict = {}
    for std_name, orig_name in column_mapping.items():
        if orig_name in df_work.columns:
            rename_dict[orig_name] = std_name
    
    df_work = df_work.rename(columns=rename_dict)
    
    # Also handle case variations (e.g., "IT spend" -> "IT Spend")
    if 'IT spend' in df_work.columns and 'IT Spend' not in df_work.columns:
        df_work = df_work.rename(columns={'IT spend': 'IT Spend'})
    
    # Rename company identifier if it was added
    if 'Company Name' in column_mapping and column_mapping['Company Name'] in df_work.columns:
        if column_mapping['Company Name'] != 'Company Name':
            df_work = df_work.rename(columns={column_mapping['Company Name']: 'Company Name'})
    
    # Convert numeric columns
    numeric_cols = ['Revenue (USD)', 'Market Value (USD)', 'IT Spend', 'Employees Total', 
                   'Company Sites', 'No. of Servers', 'No. of Storage Devices', 'Year Found']
    
    for col in numeric_cols:
        if col in df_work.columns:
            df_work[col] = pd.to_numeric(df_work[col], errors='coerce')
    
    # Filter out companies with employees = 0, market value = 0, or Revenue = 0
    print("\nFiltering companies...")
    before_filter = len(df_work)
    
    # Create filter conditions
    filter_conditions = []
    
    if 'Employees Total' in df_work.columns:
        filter_conditions.append(df_work['Employees Total'] != 0)
        filter_conditions.append(df_work['Employees Total'].notna())
    
    if 'Market Value (USD)' in df_work.columns:
        filter_conditions.append(df_work['Market Value (USD)'] != 0)
        filter_conditions.append(df_work['Market Value (USD)'].notna())
    
    if 'Revenue (USD)' in df_work.columns:
        filter_conditions.append(df_work['Revenue (USD)'] != 0)
        filter_conditions.append(df_work['Revenue (USD)'].notna())
    
    if filter_conditions:
        combined_filter = pd.Series([True] * len(df_work))
        for condition in filter_conditions:
            combined_filter = combined_filter & condition
        df_work = df_work[combined_filter].copy()
    
    after_filter = len(df_work)
    filtered_out = before_filter - after_filter
    
    print(f"Companies before filtering: {before_filter}")
    print(f"Companies after filtering: {after_filter}")
    print(f"Companies filtered out: {filtered_out}")
    
    # Calculate derived metrics
    print("\nCalculating derived metrics...")
    
    # Revenue per Employee
    if 'Revenue (USD)' in df_work.columns and 'Employees Total' in df_work.columns:
        df_work['Revenue per Employee'] = df_work['Revenue (USD)'] / df_work['Employees Total']
    
    # Market Value per Employee
    if 'Market Value (USD)' in df_work.columns and 'Employees Total' in df_work.columns:
        df_work['Market Value per Employee'] = df_work['Market Value (USD)'] / df_work['Employees Total']
    
    # IT Spend Ratio (IT Spend / Revenue)
    if 'IT Spend' in df_work.columns and 'Revenue (USD)' in df_work.columns:
        df_work['IT Spend Ratio'] = df_work['IT Spend'] / df_work['Revenue (USD)']
    
    # Employees per Site
    if 'Employees Total' in df_work.columns and 'Company Sites' in df_work.columns:
        # Replace 0 and NaN with NaN, then divide
        sites_clean = df_work['Company Sites'].replace(0, np.nan)
        df_work['Employees per Site'] = df_work['Employees Total'] / sites_clean
    
    # Technology Density Index (Servers + Storage Devices per Employee)
    if 'No. of Servers' in df_work.columns and 'No. of Storage Devices' in df_work.columns and 'Employees Total' in df_work.columns:
        servers = df_work['No. of Servers'].fillna(0).replace(0, np.nan)
        storage = df_work['No. of Storage Devices'].fillna(0).replace(0, np.nan)
        tech_total = servers.fillna(0) + storage.fillna(0)
        # Only calculate where we have at least one technology metric
        df_work['Technology Density Index'] = np.where(
            (servers.notna()) | (storage.notna()),
            tech_total / df_work['Employees Total'],
            np.nan
        )
    
    # Company Age (Current Year - Year Found)
    if 'Year Found' in df_work.columns:
        current_year = datetime.now().year
        df_work['Company Age'] = current_year - df_work['Year Found']
        # Handle invalid years
        df_work['Company Age'] = df_work['Company Age'].apply(lambda x: x if 0 <= x <= 500 else np.nan)
    
    # Geographic Dispersion (count of unique geographic locations)
    # This is a simplified version - counts non-null geographic values
    geo_dispersion_cols = [col for col in df_work.columns if any(term in str(col).lower() for term in ['country', 'region', 'geographic', 'location'])]
    if geo_dispersion_cols:
        df_work['Geographic Dispersion'] = df_work[geo_dispersion_cols].notna().sum(axis=1)
    else:
        # If no geographic columns, set to 1 (single location assumed)
        df_work['Geographic Dispersion'] = 1
    
    # Save processed data
    output_file = 'champions_group_processed.xlsx'
    df_work.to_excel(output_file, index=False)
    print(f"\nProcessed data saved to: {output_file}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"Total companies in processed dataset: {len(df_work)}")
    print(f"Total companies filtered out: {filtered_out}")
    print(f"Duplicate rows removed: {duplicates_removed}")
    
    print("\nDerived Metrics Summary:")
    metric_cols = ['Revenue per Employee', 'Market Value per Employee', 'IT Spend Ratio', 
                   'Employees per Site', 'Technology Density Index', 'Company Age', 'Geographic Dispersion']
    
    for col in metric_cols:
        if col in df_work.columns:
            valid_data = df_work[col].dropna()
            if len(valid_data) > 0:
                print(f"\n{col}:")
                print(f"  Valid records: {len(valid_data)}")
                print(f"  Mean: {valid_data.mean():.2f}")
                print(f"  Median: {valid_data.median():.2f}")
                print(f"  Min: {valid_data.min():.2f}")
                print(f"  Max: {valid_data.max():.2f}")
            else:
                print(f"\n{col}: No valid data available")
    
    return df_work

if __name__ == "__main__":
    try:
        df_processed = process_champions_data()
        print("\nProcessing completed successfully!")
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
