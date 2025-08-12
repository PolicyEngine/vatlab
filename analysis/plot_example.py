#!/usr/bin/env python3
"""
Plot turnover distribution for a specified SIC code with comparison to ONS and HMRC data.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

# =============================================================================
# ADJUST SIC CODE HERE
# =============================================================================
TARGET_SIC_CODE = 43  # Change this number to plot different SIC codes
# =============================================================================

def load_all_data():
    """Load synthetic, ONS, and HMRC data."""
    # Load synthetic data
    data_path = Path(__file__).parent / 'synthetic_firms_turnover.h5'
    
    try:
        synthetic_df = pd.read_hdf(data_path, key='firms')
        print(f"Loaded {len(synthetic_df):,} synthetic firms from HDF5 file")
    except:
        csv_path = Path(__file__).parent / 'synthetic_firms_turnover.csv'
        if csv_path.exists():
            synthetic_df = pd.read_csv(csv_path)
            print(f"Loaded {len(synthetic_df):,} synthetic firms from CSV file")
        else:
            print("Error: No synthetic data file found!")
            return None, None, None
    
    # Load ONS data
    project_root = Path(__file__).parent.parent
    ons_path = project_root / 'data' / 'ONS_UK_business_data' / 'firm_turnover.csv'
    ons_df = pd.read_csv(ons_path)
    
    # Load HMRC data
    hmrc_sector_path = project_root / 'data' / 'HMRC_VAT_annual_statistics' / 'vat_population_by_sector.csv'
    hmrc_sector_df = pd.read_csv(hmrc_sector_path)
    
    return synthetic_df, ons_df, hmrc_sector_df

def get_ons_data_for_sic(ons_df, sic_code):
    """Get ONS data for specific SIC code."""
    ons_row = ons_df[ons_df['SIC Code'] == sic_code]
    if len(ons_row) == 0:
        return None
    
    row = ons_row.iloc[0]
    return {
        'description': row['Description'],
        'bands': {
            '0-49k': row['0-49'],
            '50-99k': row['50-99'],
            '100-249k': row['100-249'],
            '250-499k': row['250-499'],
            '500-999k': row['500-999'],
            '1000-4999k': row['1000-4999'],
            '5000k+': row['5000+'],
            'total': row['Total']
        }
    }

def get_hmrc_data_for_sic(hmrc_df, sic_code):
    """Get HMRC data for specific SIC code."""
    sic_formatted = str(sic_code).zfill(5)
    hmrc_row = hmrc_df[hmrc_df['Trade_Sector'] == sic_formatted]
    if len(hmrc_row) == 0:
        return None
    
    row = hmrc_row.iloc[0]
    return {
        'description': row['Trade_Sub_Sector'],
        'total': row['2023-24']
    }

def plot_sic_distribution():
    """Plot the turnover distribution for the specified SIC code."""
    print(f"Analyzing SIC code {TARGET_SIC_CODE}...")
    
    synthetic_df, ons_df, hmrc_df = load_all_data()
    if synthetic_df is None:
        return
    
    # Get data for target SIC code - try multiple formats
    sic_formatted = str(TARGET_SIC_CODE).zfill(5)
    sic_data = synthetic_df[synthetic_df['sic_code'] == sic_formatted].copy()
    
    # If not found, try without zero padding
    if len(sic_data) == 0:
        sic_data = synthetic_df[synthetic_df['sic_code'] == str(TARGET_SIC_CODE)].copy()
    
    # If still not found, try different formats efficiently
    if len(sic_data) == 0:
        # Get unique SIC codes and find matches that start with target number
        unique_sics = synthetic_df['sic_code'].unique()
        sic_pattern = str(TARGET_SIC_CODE)
        matching_sics = [s for s in unique_sics if str(s).startswith(sic_pattern)]
        
        if matching_sics:
            sic_data = synthetic_df[synthetic_df['sic_code'].isin(matching_sics)].copy()
            print(f"Found {len(matching_sics)} SIC codes starting with {sic_pattern}: {matching_sics[:10]}")
        else:
            print(f"No SIC codes found starting with {sic_pattern}")
    
    # Debug: show available SIC codes if no match found
    if len(sic_data) == 0:
        unique_sics = sorted(synthetic_df['sic_code'].unique())
        print(f"Available SIC codes (first 20): {unique_sics[:20]}")
        target_related = [s for s in unique_sics if str(TARGET_SIC_CODE) in str(s)]
        if target_related:
            print(f"SIC codes containing '{TARGET_SIC_CODE}': {target_related}")
        return
    
    
    # Get ONS and HMRC comparison data
    ons_info = get_ons_data_for_sic(ons_df, TARGET_SIC_CODE)
    hmrc_info = get_hmrc_data_for_sic(hmrc_df, TARGET_SIC_CODE)
    
    print(f"Found {len(sic_data):,} synthetic firms for SIC {TARGET_SIC_CODE}")
    
    # Convert turnover to thousands for plotting
    sic_data['turnover_k'] = sic_data['annual_turnover_k']
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create single plot figure
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    
    # Get sector description
    description = "Unknown"
    if ons_info:
        description = ons_info['description']
    elif hmrc_info:
        description = hmrc_info['description']
    
    fig.suptitle(f'SIC {TARGET_SIC_CODE}: {description}\nTurnover Distribution', 
                 fontsize=16, fontweight='bold')
    
    # Single histogram plot
    ax.hist(sic_data['turnover_k'], bins=50, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Annual Turnover (£000s)', fontsize=12)
    ax.set_ylabel('Number of Firms', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add 90k threshold line (dashed)
    threshold_90k = 90
    ax.axvline(x=threshold_90k, color='red', linestyle='--', linewidth=2, 
                label=f'£{threshold_90k}k Threshold')
    ax.legend()
    
    # Format x-axis with thousands separators
    ax.ticklabel_format(style='plain', axis='x')
    
    # Adjust layout 
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.4)  # Make room for text below
    
    # Save the plot
    output_path = Path(__file__).parent / f'sic_{TARGET_SIC_CODE}_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Show the plot
    plt.show()
    
    # Print detailed statistics comparing all three datasets
    print(f"\n" + "="*80)
    print(f"SUMMARY STATISTICS FOR SIC {TARGET_SIC_CODE}: {description}")
    print("="*80)
    
    # Synthetic data statistics
    print(f"\n📊 SYNTHETIC DATA STATISTICS:")
    print(f"   Total firms: {len(sic_data):,}")
    print(f"   Mean turnover: £{sic_data['turnover_k'].mean():.1f}k")
    print(f"   Median turnover: £{sic_data['turnover_k'].median():.1f}k")
    print(f"   Standard deviation: £{sic_data['turnover_k'].std():.1f}k")
    print(f"   Min turnover: £{sic_data['turnover_k'].min():.1f}k")
    print(f"   Max turnover: £{sic_data['turnover_k'].max():.1f}k")
    
    # 90k threshold analysis
    below_90k = sic_data[sic_data['turnover_k'] <= threshold_90k]
    above_90k = sic_data[sic_data['turnover_k'] > threshold_90k]
    
    print(f"\n🎯 THRESHOLD ANALYSIS (£{threshold_90k}k):")
    print(f"   Below threshold: {len(below_90k):,} firms ({len(below_90k)/len(sic_data)*100:.1f}%)")
    print(f"   Above threshold: {len(above_90k):,} firms ({len(above_90k)/len(sic_data)*100:.1f}%)")
    
    # ONS comparison
    if ons_info:
        print(f"\n📋 ONS DATA COMPARISON:")
        print(f"   ONS Total: {ons_info['bands']['total']:,} firms")
        print(f"   Synthetic Total: {len(sic_data):,} firms")
        print(f"   Difference: {len(sic_data) - ons_info['bands']['total']:+,} firms")
        
        print(f"\n   ONS Turnover Band Distribution:")
        for band_name, count in ons_info['bands'].items():
            if band_name != 'total':
                print(f"      {band_name:>12}: {count:>8,} firms")
    else:
        print(f"\n📋 ONS DATA: Not available for SIC {TARGET_SIC_CODE}")
    
    # HMRC comparison
    if hmrc_info:
        print(f"\n🏛️  HMRC DATA COMPARISON:")
        print(f"   HMRC Total: {hmrc_info['total']:,} firms")
        print(f"   Synthetic Total: {len(sic_data):,} firms")
        print(f"   Difference: {len(sic_data) - hmrc_info['total']:+,} firms")
        print(f"   Accuracy: {(1 - abs(len(sic_data) - hmrc_info['total']) / hmrc_info['total']) * 100:.1f}%")
    else:
        print(f"\n🏛️  HMRC DATA: Not available for SIC {TARGET_SIC_CODE}")
    
    # Comparison summary
    print(f"\n📈 DATASET COMPARISON SUMMARY:")
    if ons_info and hmrc_info:
        print(f"   ONS Total:       {ons_info['bands']['total']:>8,} firms")
        print(f"   HMRC Total:      {hmrc_info['total']:>8,} firms")
        print(f"   Synthetic Total: {len(sic_data):>8,} firms")
        
        ons_diff = abs(len(sic_data) - ons_info['bands']['total']) / ons_info['bands']['total'] * 100
        hmrc_diff = abs(len(sic_data) - hmrc_info['total']) / hmrc_info['total'] * 100
        
        print(f"   ONS Accuracy:    {100-ons_diff:>7.1f}%")
        print(f"   HMRC Accuracy:   {100-hmrc_diff:>7.1f}%")
    
    elif ons_info:
        print(f"   ONS Total:       {ons_info['bands']['total']:>8,} firms")  
        print(f"   Synthetic Total: {len(sic_data):>8,} firms")
    elif hmrc_info:
        print(f"   HMRC Total:      {hmrc_info['total']:>8,} firms")
        print(f"   Synthetic Total: {len(sic_data):>8,} firms")

if __name__ == "__main__":
    plot_sic_distribution()