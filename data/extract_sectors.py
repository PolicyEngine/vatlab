#!/usr/bin/env python3
"""Extract sector names from ONS business statistics Excel file."""

import pandas as pd
import json

# Read the Excel file
file_path = 'ons-business-statistics/ukbusinessworkbook2024.xlsx'

# Try to read the first sheet to understand structure
xl_file = pd.ExcelFile(file_path)
print(f"Sheet names: {xl_file.sheet_names}")

# Read Table 1 which should have industry breakdown
df = pd.read_excel(file_path, sheet_name='Table 1', header=None)

# Find where the actual data starts (skip metadata rows)
for i, row in df.iterrows():
    if any(str(cell).startswith('A :') for cell in row):
        header_row = i - 1
        break

# Read with proper header
df = pd.read_excel(file_path, sheet_name='Table 1', header=header_row)

# Extract main industry sections
sectors = []
for col in df.columns:
    if isinstance(col, str) and ' : ' in col and col[0].isalpha():
        # This is a main sector like "A : Agriculture, forestry and fishing"
        letter = col.split(' : ')[0]
        name = col.split(' : ')[1]
        sectors.append({
            'code': letter,
            'name': name,
            'full': col
        })

# Print sectors for use in the app
print("\nMain industry sectors:")
for sector in sectors:
    print(f"  {sector['code']}: {sector['name']}")

# Save to JSON for easy import
with open('sectors.json', 'w') as f:
    json.dump(sectors, f, indent=2)

print(f"\nTotal sectors found: {len(sectors)}")
print("Saved to sectors.json")