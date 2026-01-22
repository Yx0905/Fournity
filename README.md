# CHAMPIONS GROUP Dataset Processing

This project processes the CHAMPIONS GROUP dataset to generate actionable insights about companies through data-driven analysis and segmentation.

## Overview

The processing script filters and transforms raw company data to create a clean dataset with derived metrics that help understand company characteristics, operational efficiency, and technology adoption.

## Features

- **Data Filtering**: Removes companies with zero employees, zero market value, or zero revenue
- **Duplicate Removal**: Eliminates duplicate rows from the dataset
- **Derived Metrics Calculation**: Computes key performance indicators including:
  - Revenue per Employee
  - Market Value per Employee
  - IT Spend Ratio
  - Employees per Site
  - Technology Density Index
  - Company Age
  - Geographic Dispersion

## Requirements

- Python 3.7+
- pandas >= 2.0.0
- openpyxl >= 3.0.0
- numpy >= 1.24.0

## Installation

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your `champions_group_data.xlsx` file in the same directory as the script
2. Run the processing script:
```bash
python3 process_champions_data.py
```

3. The processed data will be saved to `champions_group_processed.xlsx`

## Input Columns Used

The script processes the following columns from the input dataset:
- Revenue (USD)
- Market Value (USD)
- IT Spend
- Employees Total
- Company Sites
- No. of Servers
- No. of Storage Devices
- Year Found
- Geographic columns (Country, Region, etc.)

## Output

The processed dataset includes:
- All original selected columns
- All calculated derived metrics
- Summary statistics printed to console

## Filtering Results

The script reports:
- Number of companies in the original dataset
- Number of duplicate rows removed
- Number of companies filtered out (those with zero employees, market value, or revenue)
- Final number of companies in the processed dataset

## Derived Metrics Explained

1. **Revenue per Employee**: Total revenue divided by number of employees (efficiency metric)
2. **Market Value per Employee**: Market capitalization divided by number of employees
3. **IT Spend Ratio**: IT spending as a percentage of total revenue
4. **Employees per Site**: Average number of employees per company location
5. **Technology Density Index**: Combined servers and storage devices per employee
6. **Company Age**: Years since company founding (current year - year found)
7. **Geographic Dispersion**: Number of unique geographic locations (based on available geographic columns)

## Notes

- Companies with missing or zero values in key fields (employees, revenue, market value) are excluded
- Some derived metrics may show "No valid data available" if the underlying data columns are mostly empty or zero
- The script handles various column name formats and case variations automatically
