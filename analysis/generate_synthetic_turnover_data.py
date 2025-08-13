#!/usr/bin/env python3
"""
Generate synthetic firm-level turnover data from ONS firm_turnover.csv,
validated against HMRC datasets for accuracy.

This script generates individual firms matching ONS totals while minimizing
validation error against HMRC statistics.

The data uses turnover values in thousands of pounds (£k), so a value of 1000
represents £1,000,000 in annual turnover.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load ONS and HMRC data files from standardized CSV sources.
    
    Returns:
        tuple: Contains ONS turnover data, ONS employment data, HMRC turnover bands,
               HMRC sector data, and ONS total firm count for calibration.
    """
    print("LOADING DATA FILES")
    print("─" * 40)
    print("STEP 1a: Locating data files...")
    print("WHY: We need multiple official data sources for comprehensive synthetic data generation")
    
    # Define paths to all required data files relative to project root
    project_root = Path(__file__).parent.parent
    ons_path = project_root / 'data' / 'ONS_UK_business_data' / 'firm_turnover.csv'
    ons_employment_path = project_root / 'data' / 'ONS_UK_business_data' / 'firm_employment.csv'
    hmrc_turnover_path = project_root / 'data' / 'HMRC_VAT_annual_statistics' / 'vat_population_by_turnover_band.csv'
    hmrc_sector_path = project_root / 'data' / 'HMRC_VAT_annual_statistics' / 'vat_population_by_sector.csv'
    
    print(f"✓ ONS turnover data: {ons_path.name}")
    print(f"✓ ONS employment data: {ons_employment_path.name}")
    print(f"✓ HMRC turnover bands: {hmrc_turnover_path.name}")
    print(f"✓ HMRC sector data: {hmrc_sector_path.name}")
    
    print("\nSTEP 1b: Loading CSV files...")
    print("EXPLANATION: Reading official statistics into memory for processing")
    
    # Load all CSV files containing the raw data
    print("  → Loading ONS business turnover by sector and size (values in £k)...")
    ons_df = pd.read_csv(ons_path)
    print(f"    ✓ Loaded {len(ons_df):,} rows x {len(ons_df.columns)} columns")
    
    print("  → Loading ONS employment by sector and size...")
    ons_employment_df = pd.read_csv(ons_employment_path)
    print(f"    ✓ Loaded {len(ons_employment_df):,} rows x {len(ons_employment_df.columns)} columns")
    
    print("  → Loading HMRC VAT registered businesses by turnover band...")
    hmrc_turnover_df = pd.read_csv(hmrc_turnover_path)
    print(f"    ✓ Loaded {len(hmrc_turnover_df):,} rows x {len(hmrc_turnover_df.columns)} columns")
    
    print("  → Loading HMRC VAT registered businesses by sector...")
    hmrc_sector_df = pd.read_csv(hmrc_sector_path)
    print(f"    ✓ Loaded {len(hmrc_sector_df):,} rows x {len(hmrc_sector_df.columns)} columns")
    
    print("\nSTEP 1c: Extracting total firm counts for calibration...")
    print("WHY: We need baseline totals to ensure our synthetic data matches official statistics")
    
    # Extract total number of firms from ONS data for calibration
    # First try to find explicit total row (identified by empty SIC Code)
    print("  → Searching for ONS total firm count...")
    ons_total_row = ons_df[ons_df['SIC Code'].isna() | (ons_df['SIC Code'] == '')]
    if len(ons_total_row) > 0:
        ons_total = ons_total_row.iloc[0]['Total']
        print("    ✓ Found explicit total row in ONS data")
    else:
        # Fallback: sum all sector totals (excluding any summary rows with 'Total' in description)
        print("    → No explicit total found, summing sector totals...")
        sector_rows = ons_df[~ons_df['Description'].str.contains('Total', na=False)]
        ons_total = sector_rows['Total'].sum()
        print("    ✓ Calculated total from sector sums")
    
    print("  → Extracting HMRC total for validation...")
    hmrc_total = hmrc_turnover_df.iloc[-1]['Total']
    print("    ✓ Found HMRC total in latest data row")
    
    print(f"\nDATA LOADING SUMMARY:")
    print(f"  ONS total firms: {ons_total:,} (structure baseline)")
    print(f"  HMRC total firms: {hmrc_total:,} (validation target)")
    print(f"  Scaling factor: {hmrc_total/ons_total:.2f}x (HMRC/ONS ratio)")
    print(f"✓ All data files loaded successfully")
    
    return ons_df, ons_employment_df, hmrc_turnover_df, hmrc_sector_df, ons_total

def generate_values_in_band(band_name, count, min_val, max_val, midpoint, value_type="turnover"):
    """Generate realistic values within a specified band using appropriate distributions.
    
    Uses different statistical distributions based on firm characteristics:
    - Beta distribution for small firms: Creates realistic clustering toward lower values
      within each band, reflecting that most small firms operate closer to the minimum
      of their size category rather than being uniformly distributed.
    - Log-normal distribution for large firms: Captures the heavy-tailed nature of
      large firm distributions where a few very large firms dominate, which is
      characteristic of real-world business size distributions.
    
    Args:
        band_name (str): Name of the band (e.g., '0-49', '5000+', '0-4', '250+')
        count (int): Number of values to generate
        min_val (float): Minimum value for the band
        max_val (float): Maximum value for the band  
        midpoint (float): Midpoint of the band for distribution centering
        value_type (str): 'turnover' or 'employment' to determine distribution choice
        
    Returns:
        np.array: Array of generated values within the specified range
    """
    # Return empty array if no firms needed
    if count == 0:
        return np.array([])
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # For employment, use specific logic for small bands
    if value_type == "employment" and band_name == '0-4':
        # Micro businesses: use uniform distribution as requested
        values = np.random.uniform(1, 4, count)
        return np.round(values).astype(int)
    
    # For large bands (high turnover or employment), use log-normal distribution
    # This captures the "power law" nature of business sizes where few large entities dominate
    if (value_type == "turnover" and band_name in ['5000+', '1000-4999', '500-999']) or \
       (value_type == "employment" and band_name == '250+'):
        
        log_mean = np.log(midpoint)
        # Higher variance for very large firms to reflect greater diversity
        log_std = 0.8 if band_name in ['5000+', '250+'] else 0.6
        values = np.random.lognormal(log_mean, log_std, count)
        values = np.clip(values, min_val, max_val)
        
        if value_type == "employment":
            values = np.round(values).astype(int)
        return values
    
    # For small to medium bands, use beta distribution (right-skewed toward lower values)
    # This reflects that most firms cluster at the smaller end of their size category
    else:
        # Beta parameters create right skew - more firms at lower end of band
        alpha, beta = 2, 4 if band_name in ['0-49', '50-99'] else 2
        uniform_values = np.random.beta(alpha, beta, count)
        values = min_val + uniform_values * (max_val - min_val)
        
        if value_type == "employment":
            values = np.round(values).astype(int)
        return values

def assign_employment_to_firms(firms_data, ons_employment_df):
    """Assign employment values to firms using ONS employment band distributions.
    
    Ensures the synthetic firm employment distribution exactly matches ONS statistics
    by calculating precise target counts for each employment band and generating
    appropriate employment values within each band.
    
    Args:
        firms_data (list): List of firm dictionaries to assign employment to
        ons_employment_df (DataFrame): ONS employment data by sector and size band
    """
    # Set random seed for reproducible employment assignments
    np.random.seed(42)
    
    # Define employment size bands matching ONS categories
    emp_bands = ['0-4', '5-9', '10-19', '20-49', '50-99', '100-249', '250+']
    
    # Calculate total firm counts for each employment band from ONS data
    # Sum across all sectors to get economy-wide totals
    total_ons_counts = {}
    for band in emp_bands:
        if band in ons_employment_df.columns:
            # Exclude summary/total rows to avoid double counting
            sector_rows = ons_employment_df[~ons_employment_df['Description'].str.contains('Total', na=False)]
            total_ons_counts[band] = int(sector_rows[band].fillna(0).sum())
        else:
            total_ons_counts[band] = 0
    
    # Calculate totals for proportional allocation
    total_ons_firms = sum(total_ons_counts.values())
    target_firms = len(firms_data)
    
    # Fallback if no employment data available
    if total_ons_firms == 0:
        for firm in firms_data:
            firm['employment'] = 1  # Default to single-person businesses
        return
    
    # Calculate target firm counts for each employment band
    # Proportionally allocate based on ONS employment distribution
    band_targets = {}
    allocated_total = 0
    
    for band in emp_bands:
        # Calculate proportional target for this band
        exact_target = int(round(target_firms * total_ons_counts[band] / total_ons_firms))
        band_targets[band] = exact_target
        allocated_total += exact_target
    
    # Adjust for rounding differences to ensure exact total matches
    difference = target_firms - allocated_total
    if difference != 0:
        # Add/subtract difference from the largest band to minimize distortion
        largest_band = max(band_targets.keys(), key=lambda x: band_targets[x])
        band_targets[largest_band] += difference
    
    # Generate employment values for each band according to calculated targets
    employment_assignments = []
    
    # Define band parameters for value generation
    band_params = {
        '0-4': (1, 4, 2.5),
        '5-9': (5, 9, 7),
        '10-19': (10, 19, 14.5),
        '20-49': (20, 49, 34.5),
        '50-99': (50, 99, 74.5),
        '100-249': (100, 249, 174.5),
        '250+': (250, 2000, 400)  # Cap large firms at reasonable maximum
    }
    
    for band in emp_bands:
        target_count = band_targets[band]
        if target_count > 0:
            min_val, max_val, midpoint = band_params[band]
            emp_values = generate_values_in_band(
                band, target_count, min_val, max_val, midpoint, value_type="employment"
            )
            employment_assignments.extend(emp_values)
    
    # Randomize assignment order to avoid systematic bias
    np.random.shuffle(employment_assignments)
    
    # Assign employment values to firms
    for i, firm in enumerate(firms_data):
        if i < len(employment_assignments):
            firm['employment'] = employment_assignments[i]
        else:
            firm['employment'] = 1  # Default for any remaining firms

def map_to_hmrc_band(turnover_k):
    """Map turnover (in thousands of pounds) to HMRC turnover band categories."""
    if turnover_k <= 0:
        return 'Negative_or_Zero'
    elif turnover_k <= 90:  # VAT registration threshold (approximately)
        return '£1_to_Threshold'
    elif turnover_k <= 150:
        return '£Threshold_to_£150k'
    elif turnover_k <= 300:
        return '£150k_to_£300k'
    elif turnover_k <= 500:
        return '£300k_to_£500k'
    elif turnover_k <= 1000:
        return '£500k_to_£1m'
    elif turnover_k <= 10000:
        return '£1m_to_£10m'
    else:
        return 'Greater_than_£10m'

def map_to_employment_band(employment):
    """Map employment count to ONS employment band categories."""
    if employment <= 4:
        return '0-4'
    elif employment <= 9:
        return '5-9'
    elif employment <= 19:
        return '10-19'
    elif employment <= 49:
        return '20-49'
    elif employment <= 99:
        return '50-99'
    elif employment <= 249:
        return '100-249'
    else:
        return '250+'

def calculate_accuracy(synthetic_counts, target_counts):
    """Calculate accuracy between synthetic and target distributions."""
    accuracies = []
    for key, target in target_counts.items():
        synthetic = synthetic_counts.get(key, 0)
        if target > 0:
            accuracy = 1 - abs(synthetic - target) / target
        else:
            accuracy = 1.0 if synthetic == 0 else 0.0
        accuracies.append(accuracy)
    return np.mean(accuracies)

def validate_distribution(synthetic_df, target_data, validation_type="bands"):
    """Validate synthetic data against target distribution and print results."""
    
    if validation_type == "bands":
        print("  → Mapping synthetic firms to HMRC turnover bands...")
        print("    WHY: HMRC uses specific turnover thresholds (VAT registration, etc.)")
        
        # HMRC turnover band validation
        hmrc_latest = target_data.iloc[-1]
        hmrc_bands = {
            'Negative_or_Zero': hmrc_latest['Negative_or_Zero'],
            '£1_to_Threshold': hmrc_latest['£1_to_Threshold'],
            '£Threshold_to_£150k': hmrc_latest['£Threshold_to_£150k'],
            '£150k_to_£300k': hmrc_latest['£150k_to_£300k'],
            '£300k_to_£500k': hmrc_latest['£300k_to_£500k'],
            '£500k_to_£1m': hmrc_latest['£500k_to_£1m'],
            '£1m_to_£10m': hmrc_latest['£1m_to_£10m'],
            'Greater_than_£10m': hmrc_latest['Greater_than_£10m']
        }
        
        synthetic_df['hmrc_band'] = synthetic_df['annual_turnover_k'].apply(map_to_hmrc_band)
        synthetic_bands = synthetic_df['hmrc_band'].value_counts()
        
        print("  → Calculating band-by-band accuracy...")
        print("\nTURNOVER BAND VALIDATION RESULTS:")
        print("-" * 70)
        print(f"{'Band':>25} {'Synthetic':>10} {'HMRC Target':>12} {'Accuracy':>10}")
        print("-" * 70)
        
        accuracies = []
        for band, target in hmrc_bands.items():
            synthetic = synthetic_bands.get(band, 0)
            accuracy = 1 - abs(synthetic - target) / target if target > 0 else (1.0 if synthetic == 0 else 0.0)
            accuracies.append(accuracy)
            status = "✓" if accuracy > 0.90 else "⚠" if accuracy > 0.80 else "✗"
            print(f"  {status} {band:>22}: {synthetic:>8,} vs {target:>8,} ({accuracy:>6.1%})")
        
        overall_accuracy = calculate_accuracy(synthetic_bands, hmrc_bands)
        print("-" * 70)
        print(f"TURNOVER BAND OVERALL ACCURACY: {overall_accuracy:.1%}")
        return overall_accuracy
    
    elif validation_type == "sectors":
        print("  → Extracting HMRC sector targets...")
        print("    WHY: Different industries have different VAT registration patterns")
        
        # HMRC sector validation
        synthetic_sectors = synthetic_df['sic_code'].value_counts()
        
        sector_targets = {}
        for _, row in target_data.iterrows():
            sic_code = str(row['Trade_Sector'])
            hmrc_count = row['2023-24']
            if sic_code != 'Total' and not pd.isna(sic_code) and sic_code != '' and hmrc_count > 0:
                sector_targets[sic_code] = hmrc_count
        
        print(f"  → Found {len(sector_targets)} sectors in HMRC data")
        print("  → Calculating sector-by-sector accuracy...")
        
        print(f"\nSECTOR VALIDATION RESULTS (Top 10 largest sectors):")
        print("-" * 65)
        print(f"{'SIC Code':>8} {'Synthetic':>10} {'HMRC Target':>12} {'Accuracy':>10}")
        print("-" * 65)
        
        # Sort by HMRC count (largest first) and show top 10
        sorted_sectors = sorted(sector_targets.items(), key=lambda x: x[1], reverse=True)[:10]
        
        for sic_code, target in sorted_sectors:
            synthetic = synthetic_sectors.get(sic_code, 0)
            accuracy = 1 - abs(synthetic - target) / target if target > 0 else (1.0 if synthetic == 0 else 0.0)
            status = "✓" if accuracy > 0.90 else "⚠" if accuracy > 0.80 else "✗"
            print(f"  {status} {sic_code:>6}: {synthetic:>8,} vs {target:>8,} ({accuracy:>6.1%})")
        
        overall_accuracy = calculate_accuracy(synthetic_sectors, sector_targets)
        print("-" * 65)
        print(f"SECTOR OVERALL ACCURACY: {overall_accuracy:.1%} (across {len(sector_targets)} sectors)")
        return overall_accuracy
    
    elif validation_type == "employment":
        print("  → Reloading ONS employment data for validation...")
        print("    WHY: Employment bands show firm size distribution (micro, small, medium, large)")
        
        # ONS employment validation - need to reload ONS employment data
        project_root = Path(__file__).parent.parent
        ons_employment_path = project_root / 'data' / 'ONS_UK_business_data' / 'firm_employment.csv'
        ons_employment_df = pd.read_csv(ons_employment_path)
        
        # Get ONS employment totals by band
        emp_bands = ['0-4', '5-9', '10-19', '20-49', '50-99', '100-249', '250+']
        ons_emp_totals = {}
        
        print("  → Calculating ONS employment band totals...")
        for band in emp_bands:
            if band in ons_employment_df.columns:
                sector_rows = ons_employment_df[~ons_employment_df['Description'].str.contains('Total', na=False)]
                ons_emp_totals[band] = sector_rows[band].fillna(0).sum()
            else:
                ons_emp_totals[band] = 0
        
        # Get synthetic employment distribution
        print("  → Mapping synthetic firms to employment bands...")
        synthetic_df['employment_band'] = synthetic_df['employment'].apply(map_to_employment_band)
        synthetic_emp = synthetic_df['employment_band'].value_counts()
        
        print("  → Calculating employment band accuracy...")
        print(f"\nEMPLOYMENT BAND VALIDATION RESULTS:")
        print("-" * 65)
        print(f"{'Band':>8} {'Synthetic':>10} {'ONS Target':>11} {'Accuracy':>10}")
        print("-" * 65)
        
        for band in emp_bands:
            target = ons_emp_totals.get(band, 0)
            synthetic = synthetic_emp.get(band, 0)
            accuracy = 1 - abs(synthetic - target) / target if target > 0 else (1.0 if synthetic == 0 else 0.0)
            status = "✓" if accuracy > 0.90 else "⚠" if accuracy > 0.80 else "✗"
            print(f"  {status} {band:>6}: {synthetic:>8,} vs {target:>8,} ({accuracy:>6.1%})")
        
        overall_accuracy = calculate_accuracy(synthetic_emp, ons_emp_totals)
        print("-" * 65)
        print(f"EMPLOYMENT OVERALL ACCURACY: {overall_accuracy:.1%}")
        return overall_accuracy

def generate_synthetic_firms_calibrated(ons_df, ons_employment_df, ons_total, hmrc_turnover_df, hmrc_sector_df):
    """Generate synthetic firm dataset using ONS business structure calibrated to HMRC statistics.
    
    This function creates individual firm records using a sophisticated calibration approach:
    1. Generate base firms from ONS structure (sector x turnover band combinations)
    2. Calculate calibration factors for both turnover bands and sectors vs HMRC targets
    3. Add negative/zero turnover firms to match HMRC data
    4. Apply sector-stratified sampling with calibration weights
    5. Perform targeted turnover adjustments to improve band accuracy
    
    Args:
        ons_df (DataFrame): ONS business turnover data by sector and size bands (values in £k)
        ons_employment_df (DataFrame): ONS employment data by sector and size
        ons_total (int): Total number of firms from ONS data
        hmrc_turnover_df (DataFrame): HMRC VAT business counts by turnover band
        hmrc_sector_df (DataFrame): HMRC VAT business counts by sector
        
    Returns:
        DataFrame: Synthetic firms with sic_code, annual_turnover_k, and employment columns
    """
    print("\nGENERATING SYNTHETIC FIRMS")
    print("─" * 50)
    print("OVERVIEW: Creating individual firm records from aggregate statistics")
    print("METHOD: Multi-stage calibration process for statistical accuracy")
    
    print("\nPRELIMINARY: Extracting HMRC targets for calibration...")
    print("WHY: HMRC data provides ground truth for total counts by turnover band")
    
    # Extract HMRC target statistics for calibration
    hmrc_latest = hmrc_turnover_df.iloc[-1]
    hmrc_total = hmrc_latest['Total']
    hmrc_bands = {
        'Negative_or_Zero': hmrc_latest['Negative_or_Zero'],
        '£1_to_Threshold': hmrc_latest['£1_to_Threshold'],
        '£Threshold_to_£150k': hmrc_latest['£Threshold_to_£150k'],
        '£150k_to_£300k': hmrc_latest['£150k_to_£300k'],
        '£300k_to_£500k': hmrc_latest['£300k_to_£500k'],
        '£500k_to_£1m': hmrc_latest['£500k_to_£1m'],
        '£1m_to_£10m': hmrc_latest['£1m_to_£10m'],
        'Greater_than_£10m': hmrc_latest['Greater_than_£10m']
    }
    
    print(f"✓ ONS total (structure baseline): {ons_total:,}")
    print(f"✓ HMRC total (calibration target): {hmrc_total:,}")
    print(f"✓ Extracted {len(hmrc_bands)} HMRC turnover bands for validation")
    
    # STEP 1: Generate base synthetic firms using ONS turnover band structure
    print(f"\nSTEP 2a: Generate base firms using ONS structure...")
    print("EXPLANATION: Creating initial firm population from ONS sector x size data")
    print("WHY: ONS provides the most detailed breakdown of business structure by industry and size")
    
    # Define parameters for ONS turnover bands (min, max, midpoint in thousands of pounds)
    band_params = {
        '0-49': (0, 49, 24.5),          # Very small businesses  
        '50-99': (50, 99, 74.5),        # Small businesses  
        '100-249': (100, 249, 174.5),   # Small-medium businesses
        '250-499': (250, 499, 374.5),   # Medium businesses
        '500-999': (500, 999, 749.5),   # Medium-large businesses
        '1000-4999': (1000, 4999, 2999.5),  # Large businesses
        '5000+': (5000, 50000, 15000)   # Very large businesses (£5M+ turnover)
    }
    
    # Generate synthetic firms from ONS data structure
    all_base_firms = []
    
    # Process each sector (row) in the ONS data
    for _, row in ons_df.iterrows():
        sic_code = row['SIC Code']
        
        # Skip summary/total rows to avoid double counting
        if pd.isna(sic_code) or sic_code == '' or sic_code == 'Total':
            continue
            
        # Standardize SIC code format (5 digits with leading zeros)
        sic_formatted = str(int(sic_code)).zfill(5)
        
        # Generate firms for each ONS turnover band within this sector
        for band, (min_val, max_val, midpoint) in band_params.items():
            # Check if this sector has firms in this turnover band
            if band in row and pd.notna(row[band]) and row[band] > 0:
                count = int(row[band])  # Number of firms in this sector/band combination
                
                if count > 0:
                    # Generate realistic turnover values for firms in this band
                    turnovers = generate_values_in_band(
                        band, count, min_val, max_val, midpoint, value_type="turnover"
                    )
                    
                    # Create individual firm records
                    for turnover in turnovers:
                        all_base_firms.append({
                            'sic_code': sic_formatted,
                            'annual_turnover_k': turnover
                        })
    
    print(f"  → Generated {len(all_base_firms):,} base firm records from ONS structure")
    print(f"  → Adding employment data to all firms...")
    print("    WHY: Employment size is crucial for realistic business characteristics")
    
    # Add employment data to all firms
    assign_employment_to_firms(all_base_firms, ons_employment_df)
    base_df = pd.DataFrame(all_base_firms)
    print(f"    ✓ Employment assigned to all {len(base_df):,} base firms")
    print(f"✓ STEP 2a complete: {len(base_df):,} base firms with turnover and employment")
    
    # STEP 2: Calculate calibration factors
    print(f"\nSTEP 2b: Calculate calibration factors...")
    print("EXPLANATION: Computing adjustment weights to align with HMRC statistics")
    print("WHY: ONS and HMRC data use different methodologies, so we need calibration")
    
    # Map base firms to HMRC bands for calibration factor calculation
    base_df['hmrc_band'] = base_df['annual_turnover_k'].apply(map_to_hmrc_band)
    current_band_counts = base_df['hmrc_band'].value_counts()
    
    # Calculate turnover band calibration factors
    band_factors = {}
    for band, hmrc_target in hmrc_bands.items():
        current_count = current_band_counts.get(band, 0)
        if current_count > 0:
            factor = hmrc_target / current_count
        else:
            factor = 1.0
        band_factors[band] = factor
    
    # Calculate sector calibration factors
    current_sector_counts = base_df['sic_code'].value_counts()
    sector_factors = {}
    
    for _, row in hmrc_sector_df.iterrows():
        sic_code = str(row['Trade_Sector'])
        hmrc_count = row['2023-24']
        
        if sic_code != 'Total' and not pd.isna(sic_code) and sic_code != '' and hmrc_count > 0:
            current_count = current_sector_counts.get(sic_code, 0)
            if current_count > 0:
                sector_factors[sic_code] = hmrc_count / current_count
            else:
                sector_factors[sic_code] = 1.0
    
    # Calculate combined calibration weights for each firm
    calibration_weights = []
    for _, firm in base_df.iterrows():
        sic_code = firm['sic_code']
        hmrc_band = firm['hmrc_band']
        
        # Get calibration factors
        band_factor = band_factors.get(hmrc_band, 1.0)
        sector_factor = sector_factors.get(sic_code, 1.0)
        
        # Balanced combination (equal weight to both factors)
        combined_factor = (band_factor + sector_factor) / 2
        calibration_weights.append(combined_factor)
    
    base_df['calibration_weight'] = calibration_weights
    print(f"✓ STEP 2b complete: Calibration weights calculated for all firms")
    print(f"  Average band factor: {np.mean(list(band_factors.values())):.2f}")
    print(f"  Average sector factor: {np.mean(list(sector_factors.values())):.2f}")
    
    # STEP 3: Add negative/zero turnover firms
    print(f"\nSTEP 2c: Add zero turnover firms...")
    print("EXPLANATION: Creating firms with zero turnover to match HMRC data")
    print("WHY: HMRC data includes struggling businesses with zero/minimal turnover")
    
    negative_zero_target = hmrc_bands['Negative_or_Zero']
    all_calibrated_firms = []
    
    if negative_zero_target > 0:
        # Get sector distribution from HMRC for negative firms
        hmrc_sector_weights = {}
        total_hmrc_sector = 0
        
        for _, row in hmrc_sector_df.iterrows():
            sic_code = str(row['Trade_Sector'])
            hmrc_count = row['2023-24']
            
            if sic_code != 'Total' and not pd.isna(sic_code) and sic_code != '' and hmrc_count > 0:
                hmrc_sector_weights[sic_code] = hmrc_count
                total_hmrc_sector += hmrc_count
        
        # Normalize weights
        for sic in hmrc_sector_weights:
            hmrc_sector_weights[sic] = hmrc_sector_weights[sic] / total_hmrc_sector
        
        # Generate negative/zero firms
        print(f"Generating {negative_zero_target:,} zero turnover firms...")
        for sic_code, weight in hmrc_sector_weights.items():
            firms_for_sector = int(negative_zero_target * weight)
            if firms_for_sector > 0:
                for _ in range(firms_for_sector):
                    turnover = 0  # Zero turnover for struggling businesses
                    all_calibrated_firms.append({
                        'sic_code': sic_code,
                        'annual_turnover_k': turnover,
                        'employment': 1  # Default employment for zero-turnover firms
                    })
        
        print(f"    ✓ Added {len(all_calibrated_firms):,} zero turnover firms")
    else:
        print("    → No zero turnover firms needed (HMRC target = 0)")
    
    print(f"✓ STEP 2c complete: Zero-turnover firms added as per HMRC distribution")
    
    # STEP 4: Sector-stratified sampling for positive firms
    print(f"\nSTEP 2d: Sector-stratified sampling for positive firms...")
    print("EXPLANATION: Sampling firms within each sector using calibration weights")
    print("WHY: Ensures we get the right number of firms per sector as per HMRC targets")
    
    remaining_target = hmrc_total - len(all_calibrated_firms)
    print(f"Target positive firms: {remaining_target:,}")
    
    # Get sector targets (excluding negative/zero already allocated)
    sector_targets = {}
    for _, row in hmrc_sector_df.iterrows():
        sic_code = str(row['Trade_Sector'])
        hmrc_count = row['2023-24']
        
        if sic_code != 'Total' and not pd.isna(sic_code) and sic_code != '' and hmrc_count > 0:
            # Subtract negative firms already allocated
            negative_allocated = int(negative_zero_target * hmrc_sector_weights.get(sic_code, 0))
            positive_target = max(0, hmrc_count - negative_allocated)
            if positive_target > 0:
                sector_targets[sic_code] = positive_target
    
    # Sample within each sector using calibration weights
    stage1_firms = []
    for sic_code, target_count in sector_targets.items():
        # Get available firms for this sector
        sector_firms = base_df[base_df['sic_code'] == sic_code]
        
        if len(sector_firms) > 0:
            # Sample using calibration weights as probabilities
            sector_weights = sector_firms['calibration_weight'].values
            sector_probs = sector_weights / sector_weights.sum()
            
            # Sample with replacement if needed
            selected_indices = np.random.choice(
                len(sector_firms), 
                size=min(target_count, remaining_target), 
                replace=True, 
                p=sector_probs
            )
            selected_firms = sector_firms.iloc[selected_indices]
            
            for _, firm in selected_firms.iterrows():
                stage1_firms.append({
                    'sic_code': firm['sic_code'],
                    'annual_turnover_k': firm['annual_turnover_k'],
                    'employment': firm['employment']
                })
            
            remaining_target -= len(selected_firms)
            if remaining_target <= 0:
                break
    
    # Add positive firms to the final dataset
    all_calibrated_firms.extend(stage1_firms)
    print(f"    ✓ Added {len(stage1_firms):,} positive firms through stratified sampling")
    print(f"✓ STEP 2d complete: {len(all_calibrated_firms):,} total firms (negative + positive)")
    
    # STEP 5: Fine-tune turnover bands through targeted adjustments
    print(f"\nSTEP 2e: Fine-tune turnover band distribution...")
    print("EXPLANATION: Making targeted adjustments to improve band accuracy")
    print("WHY: Initial sampling may not perfectly match HMRC band distributions")
    
    final_df = pd.DataFrame(all_calibrated_firms)
    final_df['hmrc_band'] = final_df['annual_turnover_k'].apply(map_to_hmrc_band)
    current_bands = final_df['hmrc_band'].value_counts()
    
    # Apply targeted adjustments for worst-performing bands
    band_adjustments_needed = {}
    for band, hmrc_target in hmrc_bands.items():
        current_count = current_bands.get(band, 0)
        difference = hmrc_target - current_count
        band_adjustments_needed[band] = difference
    
    # Find bands that need the most adjustment and fix top 3
    sorted_adjustments = sorted(band_adjustments_needed.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
    
    adjustment_count = 0
    for band, needed_change in sorted_adjustments[:3]:
        if abs(needed_change) < 1000:  # Skip small adjustments
            continue
            
        if needed_change > 0:
            # Need more firms in this band - find over-represented bands to adjust from
            over_bands = [b for b, change in band_adjustments_needed.items() if change < -500]
            
            for source_band in over_bands:
                if needed_change <= 0:
                    break
                
                # Find firms in source band that we can move
                source_firms = final_df[final_df['hmrc_band'] == source_band]
                move_count = min(abs(needed_change), len(source_firms) // 10)  # Move up to 10%
                
                if move_count > 0:
                    # Randomly select firms to adjust their turnover
                    firms_to_adjust = source_firms.sample(move_count, random_state=42)
                    
                    # Generate new turnover values for target band
                    if band == '£1_to_Threshold':
                        new_turnovers = np.random.uniform(1, 84, move_count)
                    elif band == '£Threshold_to_£150k':
                        new_turnovers = np.random.uniform(90, 150, move_count)
                    elif band == '£150k_to_£300k':
                        new_turnovers = np.random.uniform(150, 300, move_count)
                    elif band == '£300k_to_£500k':
                        new_turnovers = np.random.uniform(300, 500, move_count)
                    elif band == '£500k_to_£1m':
                        new_turnovers = np.random.uniform(500, 1000, move_count)
                    elif band == '£1m_to_£10m':
                        new_turnovers = np.random.uniform(1000, 10000, move_count)
                    else:
                        new_turnovers = np.random.uniform(10000, 50000, move_count)
                    
                    # Update turnover values
                    final_df.loc[firms_to_adjust.index, 'annual_turnover_k'] = new_turnovers
                    final_df.loc[firms_to_adjust.index, 'hmrc_band'] = band
                    
                    needed_change -= move_count
                    adjustment_count += move_count
    
    if adjustment_count > 0:
        print(f"    ✓ Applied {adjustment_count:,} targeted turnover adjustments")
        print(f"    → Improved accuracy for worst-performing turnover bands")
    else:
        print(f"    → No significant adjustments needed (bands already well-matched)")
    
    print(f"✓ STEP 2e complete: Final dataset contains {len(final_df):,} synthetic firms")
    print(f"✓ SYNTHETIC FIRM GENERATION COMPLETE")
    
    return final_df[['sic_code', 'annual_turnover_k', 'employment']]

def main():
    """Main execution function for synthetic firm data generation.
    
    Orchestrates the complete synthetic data generation process:
    1. Load all required datasets (ONS turnover, ONS employment, HMRC validation data)
    2. Generate synthetic firms using ONS structure scaled to HMRC targets
    3. Validate the synthetic data against HMRC reference statistics
    4. Save the final dataset in CSV format
    
    The process creates individual firm records with realistic distributions that
    match official UK business statistics across multiple dimensions.
    """
    print("SYNTHETIC FIRM DATA GENERATION")
    print("Using ONS firm_turnover.csv as base structure")
    print("Scaling to match HMRC total counts")
    print("\n" + "WHY WE'RE DOING THIS:")
    print("- ONS data provides detailed sector x size band structure but aggregated totals")
    print("- HMRC data provides accurate total firm counts by turnover bands and sectors")
    print("- We combine both to generate individual firm records that match real UK business statistics")
    print("- This creates a synthetic dataset for policy analysis while preserving statistical accuracy")
    
    # STEP 1: Load all required datasets
    print(f"\n{'='*80}")
    print("PHASE 1: DATA LOADING")
    print(f"{'='*80}")
    print("EXPLANATION: Loading official UK business statistics from multiple sources")
    print("- ONS (Office for National Statistics): Business structure by sector and size")
    print("- HMRC (Her Majesty's Revenue and Customs): VAT registration data for validation")
    print("- This multi-source approach ensures our synthetic data matches reality")
    
    ons_df, ons_employment_df, hmrc_turnover_df, hmrc_sector_df, ons_total = load_data()
    print(f"✓ Data loading completed successfully")
    
    # STEP 2: Generate synthetic firms using calibrated approach
    print(f"\n{'='*80}")
    print("PHASE 2: SYNTHETIC FIRM GENERATION")
    print(f"{'='*80}")
    print("EXPLANATION: Creating individual firm records from aggregate statistics")
    print("- Start with ONS business structure (how many firms in each sector/size combination)")
    print("- Apply calibration to match HMRC total counts (ground truth validation)")
    print("- Generate realistic turnover and employment values within each category")
    print("- This ensures each synthetic firm represents a realistic UK business")
    
    synthetic_df = generate_synthetic_firms_calibrated(
        ons_df, ons_employment_df, ons_total, hmrc_turnover_df, hmrc_sector_df
    )
    print(f"✓ Synthetic firm generation completed")
    
    # STEP 3: Comprehensive validation against reference statistics
    print(f"\n{'='*80}")
    print("PHASE 3: VALIDATION & QUALITY ASSURANCE")
    print(f"{'='*80}")
    print("EXPLANATION: Verifying our synthetic data matches official statistics")
    print("- Compare turnover band distribution against HMRC VAT data")
    print("- Compare sector distribution against HMRC sector statistics") 
    print("- Compare employment distribution against ONS employment data")
    print("- This validation ensures our synthetic firms are statistically representative")
    
    print(f"\n{'-'*60}")
    print("VALIDATION 1: TURNOVER BAND DISTRIBUTION")
    print(f"{'-'*60}")
    print("Comparing synthetic firm turnover bands against HMRC VAT registration data...")
    print("WHY: HMRC data shows the true distribution of UK businesses by revenue size")
    band_accuracy = validate_distribution(synthetic_df, hmrc_turnover_df, "bands")
    
    print(f"\n{'-'*60}")
    print("VALIDATION 2: SECTOR DISTRIBUTION") 
    print(f"{'-'*60}")
    print("Comparing synthetic firm sectors against HMRC sector statistics...")
    print("WHY: Ensures we have the right mix of industries (manufacturing, services, etc.)")
    sector_accuracy = validate_distribution(synthetic_df, hmrc_sector_df, "sectors")
    
    print(f"\n{'-'*60}")
    print("VALIDATION 3: EMPLOYMENT DISTRIBUTION")
    print(f"{'-'*60}")
    print("Comparing synthetic firm employment against ONS employment band data...")
    print("WHY: Ensures realistic firm sizes (micro, small, medium, large enterprises)")
    emp_accuracy = validate_distribution(synthetic_df, None, "employment")
    
    # Calculate overall accuracy
    overall_accuracy = (band_accuracy + sector_accuracy + emp_accuracy) / 3
    
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY & INTERPRETATION")
    print(f"{'='*80}")
    print(f"Turnover band accuracy: {band_accuracy:.1%} - How well we match HMRC revenue distribution")
    print(f"Sector accuracy:        {sector_accuracy:.1%} - How well we match HMRC industry distribution")
    print(f"Employment accuracy:    {emp_accuracy:.1%} - How well we match ONS firm size distribution")
    print(f"Overall accuracy:       {overall_accuracy:.1%} - Combined statistical accuracy")
    print(f"\nINTERPRETation:")
    if overall_accuracy >= 0.95:
        print("✓ EXCELLENT: Synthetic data very closely matches official statistics")
    elif overall_accuracy >= 0.90:
        print("✓ GOOD: Synthetic data closely matches official statistics")
    elif overall_accuracy >= 0.80:
        print("⚠ FAIR: Synthetic data reasonably matches official statistics")
    else:
        print("⚠ POOR: Synthetic data needs improvement to better match official statistics")
    
    # STEP 4: Save the final synthetic dataset in CSV format
    print(f"\n{'='*80}")
    print("PHASE 4: DATASET EXPORT")
    print(f"{'='*80}")
    print("EXPLANATION: Saving validated synthetic dataset for policy analysis")
    print("- Each row represents one synthetic UK business")
    print("- Columns include SIC code (industry), annual turnover (£k), and employment")
    print("- Data maintains statistical accuracy while protecting business confidentiality")
    
    csv_path = Path(__file__).parent / 'synthetic_firms_turnover.csv'
    synthetic_df.to_csv(csv_path, index=False)
    
    file_size_mb = csv_path.stat().st_size / 1024 / 1024
    print(f"✓ Saved {len(synthetic_df):,} firms to {csv_path}")
    print(f"✓ File size: {file_size_mb:.1f} MB")
    print(f"✓ Columns: {list(synthetic_df.columns)}")
    print(f"✓ Ready for policy analysis and VAT threshold modeling")
    
    # Show sample of generated data
    print(f"\nSAMPLE DATA (First 10 synthetic firms):")
    print("─" * 60)
    print(synthetic_df.head(10))
    
    print(f"\n{'='*80}")
    print("PROCESS COMPLETE")
    print(f"{'='*80}")
    print("✓ Successfully generated synthetic UK business dataset")
    print("✓ Statistical accuracy validated against official sources") 
    print("✓ Dataset ready for VAT threshold and policy analysis")
    print("✓ All data anonymized and suitable for research use")

if __name__ == "__main__":
    main()