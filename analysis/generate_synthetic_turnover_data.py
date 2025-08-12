#!/usr/bin/env python3
"""
Generate synthetic firm-level turnover data from ONS firm_turnover.csv,
validated against HMRC datasets for accuracy.

This script generates individual firms matching ONS totals while minimizing
validation error against HMRC statistics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load ONS and HMRC data files."""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    project_root = Path(__file__).parent.parent
    ons_path = project_root / 'data' / 'ONS_UK_business_data' / 'firm_turnover.csv'
    ons_employment_path = project_root / 'data' / 'ONS_UK_business_data' / 'firm_employment.csv'
    hmrc_turnover_path = project_root / 'data' / 'HMRC_VAT_annual_statistics' / 'vat_population_by_turnover_band.csv'
    hmrc_sector_path = project_root / 'data' / 'HMRC_VAT_annual_statistics' / 'vat_population_by_sector.csv'
    
    ons_df = pd.read_csv(ons_path)
    ons_employment_df = pd.read_csv(ons_employment_path)
    hmrc_turnover_df = pd.read_csv(hmrc_turnover_path)
    hmrc_sector_df = pd.read_csv(hmrc_sector_path)
    
    # Get ONS total from explicit Total row (empty SIC Code)
    ons_total_row = ons_df[ons_df['SIC Code'].isna() | (ons_df['SIC Code'] == '')]
    if len(ons_total_row) > 0:
        ons_total = ons_total_row.iloc[0]['Total']
    else:
        # Fallback: sum all sectors (excluding any rows with 'Total' in description)
        sector_rows = ons_df[~ons_df['Description'].str.contains('Total', na=False)]
        ons_total = sector_rows['Total'].sum()
    
    print(f"ONS total firms: {ons_total:,}")
    print(f"HMRC total (for validation): {hmrc_turnover_df.iloc[-1]['Total']:,}")
    
    return ons_df, ons_employment_df, hmrc_turnover_df, hmrc_sector_df, ons_total

def generate_turnover_in_band(band_name, count, min_val, max_val, midpoint):
    """Generate realistic turnover values within a band."""
    if count == 0:
        return np.array([])
    
    np.random.seed(42)
    
    if band_name == '5s+':
        # Large firms: Log-normal distribution
        log_mean = np.log(midpoint)
        log_std = 0.8
        values = np.random.lognormal(log_mean, log_std, count)
        values = np.clip(values, min_val, max_val)
    elif band_name in ['1000-4999', '500-999']:
        # Medium firms: Log-normal
        log_mean = np.log(midpoint)
        log_std = 0.6
        values = np.random.lognormal(log_mean, log_std, count)
        values = np.clip(values, min_val, max_val)
    else:
        # Small firms: Beta distribution (right-skewed)
        alpha, beta = 2, 4
        uniform_values = np.random.beta(alpha, beta, count)
        values = min_val + uniform_values * (max_val - min_val)
    
    return values

def generate_employment_in_band(band_name, count):
    """Generate realistic employment values within an employment band.
    
    Uses different distributions based on employment band size:
    - Small bands (0-4): Most firms have 1-3 employees
    - Medium bands: Uniform distribution within band
    - Large bands (250+): Log-normal distribution for realistic spread
    """
    if count == 0:
        return np.array([])
    
    np.random.seed(42)
    
    # Map band names to ranges
    if band_name == '0-4':
        # Most small firms have 1-3 employees
        values = np.random.choice([1, 2, 3, 4], size=count, p=[0.4, 0.3, 0.2, 0.1])
    elif band_name == '5-9':
        values = np.random.uniform(5, 9, count)
    elif band_name == '10-19':
        values = np.random.uniform(10, 19, count)
    elif band_name == '20-49':
        values = np.random.uniform(20, 49, count)
    elif band_name == '50-99':
        values = np.random.uniform(50, 99, count)
    elif band_name == '100-249':
        values = np.random.uniform(100, 249, count)
    elif band_name == '250+':
        # Large firms: log-normal distribution
        log_mean = np.log(400)  # Mean around 400 employees
        log_std = 0.8
        values = np.random.lognormal(log_mean, log_std, count)
        values = np.clip(values, 250, 2000)  # Cap at reasonable max
    else:
        values = np.array([1] * count)  # Default to 1 employee
    
    return np.round(values).astype(int)

def assign_employment_to_firms(firms_data, ons_employment_df):
    """Assign employment to firms using precise ONS employment band matching.
    
    Focus on exact ONS band calibration with improved within-band distributions.
    """
    np.random.seed(42)
    
    emp_bands = ['0-4', '5-9', '10-19', '20-49', '50-99', '100-249', '250+']
    
    # Get exact ONS employment band counts
    total_ons_counts = {}
    for band in emp_bands:
        if band in ons_employment_df.columns:
            sector_rows = ons_employment_df[~ons_employment_df['Description'].str.contains('Total', na=False)]
            total_ons_counts[band] = int(sector_rows[band].fillna(0).sum())
        else:
            total_ons_counts[band] = 0
    
    total_ons_firms = sum(total_ons_counts.values())
    target_firms = len(firms_data)
    
    if total_ons_firms == 0:
        for firm in firms_data:
            firm['employment'] = 1
        return
    
    # Calculate exact target counts for each band
    band_targets = {}
    allocated_total = 0
    
    for band in emp_bands:
        exact_target = int(round(target_firms * total_ons_counts[band] / total_ons_firms))
        band_targets[band] = exact_target
        allocated_total += exact_target
    
    # Adjust for rounding differences
    difference = target_firms - allocated_total
    if difference != 0:
        # Add/remove from largest band
        largest_band = max(band_targets.keys(), key=lambda x: band_targets[x])
        band_targets[largest_band] += difference
    
    # Generate exact employment assignments for each band
    employment_assignments = []
    
    for band in emp_bands:
        target_count = band_targets[band]
        if target_count > 0:
            emp_values = generate_employment_in_band_improved(band, target_count)
            employment_assignments.extend(emp_values)
    
    # Shuffle to randomize assignment across firms
    np.random.shuffle(employment_assignments)
    
    # Assign to firms
    for i, firm in enumerate(firms_data):
        if i < len(employment_assignments):
            firm['employment'] = employment_assignments[i]
        else:
            firm['employment'] = 1

def generate_employment_in_band_improved(band_name, count):
    """Generate improved employment values within bands using better distributions."""
    if count == 0:
        return np.array([])
    
    np.random.seed(42)
    
    if band_name == '0-4':
        # Realistic small firm distribution: heavily weighted to 1-2 employees
        values = np.random.choice([1, 2, 3, 4], size=count, p=[0.45, 0.35, 0.15, 0.05])
    elif band_name == '5-9':
        # Beta distribution for realistic spread
        beta_values = np.random.beta(2, 3, count)
        values = 5 + (beta_values * 4)
        values = np.round(values).astype(int)
    elif band_name == '10-19':
        # Slightly right-skewed within band
        beta_values = np.random.beta(2, 2.5, count)
        values = 10 + (beta_values * 9)
        values = np.round(values).astype(int)
    elif band_name == '20-49':
        # More uniform but slight right skew
        beta_values = np.random.beta(2, 2, count)
        values = 20 + (beta_values * 29)
        values = np.round(values).astype(int)
    elif band_name == '50-99':
        # Uniform distribution
        values = np.random.uniform(50, 99, count)
        values = np.round(values).astype(int)
    elif band_name == '100-249':
        # Slight left skew for medium-large firms
        beta_values = np.random.beta(2.5, 2, count)
        values = 100 + (beta_values * 149)
        values = np.round(values).astype(int)
    elif band_name == '250+':
        # Log-normal for large firms, more conservative
        log_mean = np.log(380)
        log_std = 0.7
        values = np.random.lognormal(log_mean, log_std, count)
        values = np.clip(values, 250, 3000)
        values = np.round(values).astype(int)
    else:
        values = np.array([1] * count)
    
    return values

def generate_synthetic_firms_calibrated(ons_df, ons_employment_df, ons_total, hmrc_turnover_df, hmrc_sector_df):
    """Generate synthetic firms using ONS structure calibrated to HMRC targets."""
    print("\n" + "=" * 60)
    print("GENERATING ONS+HMRC CALIBRATED SYNTHETIC FIRMS")
    print("=" * 60)
    
    # Get HMRC targets for calibration
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
    
    print(f"ONS total: {ons_total:,}")
    print(f"HMRC total: {hmrc_total:,}")
    
    # Step 1: Generate base firms using ONS structure
    print(f"\nStep 1: Generate base firms using ONS structure...")
    
    # ONS band parameters for realistic distributions
    band_params = {
        '0-49': (0, 49, 24.5),
        '50-99': (50, 99, 74.5),
        '100-249': (100, 249, 174.5),
        '250-499': (250, 499, 374.5),
        '500-999': (500, 999, 749.5),
        '1000-4999': (1000, 4999, 2999.5),
        '5000+': (5000, 50000, 15000)
    }
    
    # Generate base firms from ONS data with employment data
    all_base_firms = []
    
    # Get employment data for each SIC code
    employment_by_sic = {}
    for _, emp_row in ons_employment_df.iterrows():
        emp_sic_code = emp_row['SIC Code']
        if pd.isna(emp_sic_code) or emp_sic_code == '' or emp_sic_code == 'Total':
            continue
        emp_sic_formatted = str(int(emp_sic_code)).zfill(5)
        employment_by_sic[emp_sic_formatted] = emp_row
    
    for _, row in ons_df.iterrows():
        sic_code = row['SIC Code']
        
        # Skip total/empty rows
        if pd.isna(sic_code) or sic_code == '' or sic_code == 'Total':
            continue
            
        sic_formatted = str(int(sic_code)).zfill(5)
        
        # Get employment data for this SIC code
        emp_data = employment_by_sic.get(sic_formatted)
        
        # Generate firms for each ONS turnover band
        for band, (min_val, max_val, midpoint) in band_params.items():
            if band in row and pd.notna(row[band]) and row[band] > 0:
                count = int(row[band])
                
                if count > 0:
                    # Generate turnover values
                    turnovers = generate_turnover_in_band(
                        band, count, min_val, max_val, midpoint
                    )
                    
                    for turnover in turnovers:
                        all_base_firms.append({
                            'sic_code': sic_formatted,
                            'annual_turnover_k': turnover
                        })
    
    # Assign employment to all firms using ONS employment distribution
    print(f"Assigning employment to {len(all_base_firms):,} firms...")
    assign_employment_to_firms(all_base_firms, ons_employment_df)
    
    base_df = pd.DataFrame(all_base_firms)
    print(f"Generated {len(base_df):,} base firms from ONS structure with employment")
    
    # Step 2: Calculate calibration factors for both HMRC datasets
    print(f"\nStep 2: Calculate calibration factors...")
    
    # Map base firms to HMRC bands
    def map_to_hmrc_band(turnover):
        if turnover <= 0:
            return 'Negative_or_Zero'
        elif turnover <= 85:
            return '£1_to_Threshold'
        elif turnover <= 150:
            return '£Threshold_to_£150k'
        elif turnover <= 300:
            return '£150k_to_£300k'
        elif turnover <= 500:
            return '£300k_to_£500k'
        elif turnover <= 1000:
            return '£500k_to_£1m'
        elif turnover <= 10000:
            return '£1m_to_£10m'
        else:
            return 'Greater_than_£10m'
    
    base_df['hmrc_band'] = base_df['annual_turnover_k'].apply(map_to_hmrc_band)
    
    # Current band distribution from ONS-based generation
    current_band_counts = base_df['hmrc_band'].value_counts()
    
    # Band calibration factors
    band_factors = {}
    print("HMRC band calibration factors:")
    for band, hmrc_target in hmrc_bands.items():
        current_count = current_band_counts.get(band, 0)
        if current_count > 0:
            factor = hmrc_target / current_count
        else:
            factor = 1.0
        band_factors[band] = factor
        print(f"  {band:>20}: {factor:.3f} (target: {hmrc_target:,}, current: {current_count:,})")
    
    # Sector calibration factors
    hmrc_sector_clean = hmrc_sector_df.copy()
    hmrc_sector_clean['Trade_Sector'] = hmrc_sector_clean['Trade_Sector'].astype(str)
    
    current_sector_counts = base_df['sic_code'].value_counts()
    
    sector_factors = {}
    print(f"\nHMRC sector calibration factors (top 10):")
    count = 0
    for _, row in hmrc_sector_clean.iterrows():
        if count >= 10:
            break
        sic_code = row['Trade_Sector']
        hmrc_count = row['2023-24']
        
        if sic_code == 'Total' or pd.isna(sic_code) or sic_code == '' or hmrc_count <= 0:
            continue
            
        current_count = current_sector_counts.get(sic_code, 0)
        if current_count > 0:
            factor = hmrc_count / current_count
        else:
            factor = 1.0
        sector_factors[sic_code] = factor
        print(f"  SIC {sic_code}: {factor:.3f} (target: {hmrc_count:,}, current: {current_count:,})")
        count += 1
    
    # Fill in factors for all sectors
    for _, row in hmrc_sector_clean.iterrows():
        sic_code = row['Trade_Sector']
        hmrc_count = row['2023-24']
        
        if sic_code == 'Total' or pd.isna(sic_code) or sic_code == '' or hmrc_count <= 0:
            continue
            
        if sic_code not in sector_factors:
            current_count = current_sector_counts.get(sic_code, 0)
            if current_count > 0:
                sector_factors[sic_code] = hmrc_count / current_count
            else:
                sector_factors[sic_code] = 1.0
    
    # Step 3: Apply calibration with balanced weighting
    print(f"\nStep 3: Apply balanced calibration...")
    
    # Calculate combined calibration weights for each firm
    calibration_weights = []
    target_total = hmrc_total  # Target HMRC total
    
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
    
    # Step 4: Add missing negative/zero turnover firms first
    print(f"\nStep 4: Add negative/zero turnover firms...")
    
    negative_zero_target = hmrc_bands['Negative_or_Zero']
    all_calibrated_firms = []
    
    if negative_zero_target > 0:
        # Get sector weights from HMRC for distribution
        hmrc_sector_weights = {}
        total_hmrc_sector = 0
        
        for _, row in hmrc_sector_clean.iterrows():
            sic_code = row['Trade_Sector']
            hmrc_count = row['2023-24']
            
            if sic_code != 'Total' and not pd.isna(sic_code) and sic_code != '' and hmrc_count > 0:
                hmrc_sector_weights[sic_code] = hmrc_count
                total_hmrc_sector += hmrc_count
        
        # Normalize weights
        for sic in hmrc_sector_weights:
            hmrc_sector_weights[sic] = hmrc_sector_weights[sic] / total_hmrc_sector
        
        # Generate negative/zero firms (optimized)
        print(f"Generating {negative_zero_target:,} negative/zero firms...")
        negative_firms_data = []
        for sic_code, weight in hmrc_sector_weights.items():
            firms_for_sector = int(negative_zero_target * weight)
            if firms_for_sector > 0:
                turnovers = np.random.uniform(-50, 0, firms_for_sector)
                sic_codes = [sic_code] * firms_for_sector
                negative_firms_data.extend(list(zip(sic_codes, turnovers)))
        
        # Convert to list format efficiently
        negative_firms_list = []
        for sic_code, turnover in negative_firms_data:
            negative_firms_list.append({
                'sic_code': sic_code,
                'annual_turnover_k': turnover
            })
        
        # Assign employment to negative firms
        assign_employment_to_firms(negative_firms_list, ons_employment_df)
        all_calibrated_firms.extend(negative_firms_list)
        
        print(f"Added {len(all_calibrated_firms):,} negative/zero turnover firms")
    
    # Step 5: Two-Stage Calibration - Stage 1: Perfect Sector Matching
    print(f"\nStep 5: Two-Stage Calibration...")
    print("Stage 1: Sector-stratified sampling for perfect sector matching")
    
    remaining_target = target_total - len(all_calibrated_firms)
    print(f"Target positive firms: {remaining_target:,}")
    
    # Get HMRC sector targets (excluding negative/zero which we already added)
    sector_targets = {}
    for _, row in hmrc_sector_clean.iterrows():
        sic_code = row['Trade_Sector']
        hmrc_count = row['2023-24']
        
        if sic_code != 'Total' and not pd.isna(sic_code) and sic_code != '' and hmrc_count > 0:
            # Subtract the negative/zero firms already allocated to this sector
            negative_allocated = int(negative_zero_target * hmrc_sector_weights.get(sic_code, 0))
            positive_target = max(0, hmrc_count - negative_allocated)
            if positive_target > 0:
                sector_targets[sic_code] = positive_target
    
    print(f"Sector-stratified sampling for {len(sector_targets)} sectors...")
    
    # Sample within each sector separately
    stage1_firms = []
    total_allocated = 0
    
    for sic_code, target_count in sector_targets.items():
        # Get available firms for this sector
        sector_firms = base_df[base_df['sic_code'] == sic_code]
        
        if len(sector_firms) == 0:
            print(f"  Warning: No ONS firms found for SIC {sic_code}")
            continue
        
        # Sample from available firms (with replacement if needed)
        if len(sector_firms) >= target_count:
            # Sample without replacement
            sector_weights = sector_firms['calibration_weight'].values
            sector_probs = sector_weights / sector_weights.sum()
            
            selected_indices = np.random.choice(
                len(sector_firms), 
                size=target_count, 
                replace=False, 
                p=sector_probs
            )
            selected_firms = sector_firms.iloc[selected_indices]
        else:
            # Sample with replacement if not enough firms
            sector_weights = sector_firms['calibration_weight'].values
            sector_probs = sector_weights / sector_weights.sum()
            
            selected_indices = np.random.choice(
                len(sector_firms), 
                size=target_count, 
                replace=True, 
                p=sector_probs
            )
            selected_firms = sector_firms.iloc[selected_indices]
        
        # Add to stage 1 results
        for _, firm in selected_firms.iterrows():
            stage1_firms.append({
                'sic_code': firm['sic_code'],
                'annual_turnover_k': firm['annual_turnover_k'],
                'employment': firm['employment']
            })
        
        total_allocated += target_count
        if len(sector_targets) <= 10 or sic_code in list(sector_targets.keys())[:5]:
            print(f"  SIC {sic_code}: {target_count:,} firms allocated")
    
    print(f"Stage 1 complete: {total_allocated:,} firms allocated via sector-stratified sampling")
    
    # Add Stage 1 firms to calibrated firms
    all_calibrated_firms.extend(stage1_firms)
    
    # Step 6: Two-Stage Calibration - Stage 2: Adjust Turnover Bands
    print(f"\nStep 6: Stage 2 - Adjust turnover values to improve band accuracy...")
    
    stage2_df = pd.DataFrame(all_calibrated_firms)
    stage2_df['hmrc_band'] = stage2_df['annual_turnover_k'].apply(map_to_hmrc_band)
    current_bands = stage2_df['hmrc_band'].value_counts()
    
    print("Current band distribution after Stage 1:")
    band_adjustments_needed = {}
    for band, hmrc_target in hmrc_bands.items():
        current_count = current_bands.get(band, 0)
        difference = hmrc_target - current_count
        accuracy = 1 - abs(difference) / hmrc_target if hmrc_target > 0 else 1.0
        band_adjustments_needed[band] = difference
        print(f"  {band:>20}: {current_count:>8,} vs {hmrc_target:>8,} (diff: {difference:+6,}, acc: {accuracy:>5.1%})")
    
    # Apply targeted adjustments for worst-performing bands
    print(f"\nApplying targeted band adjustments...")
    
    # Identify bands that need the most adjustment
    sorted_adjustments = sorted(band_adjustments_needed.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
    
    adjustment_count = 0
    for band, needed_change in sorted_adjustments[:3]:  # Fix top 3 worst bands
        if abs(needed_change) < 1000:  # Skip small adjustments
            continue
            
        print(f"Adjusting {band}: need {needed_change:+,} firms")
        
        if needed_change > 0:
            # Need more firms in this band - find firms in over-represented bands
            # and adjust their turnover to move them to this band
            over_bands = [b for b, change in band_adjustments_needed.items() if change < -500]
            
            for source_band in over_bands:
                if needed_change <= 0:
                    break
                
                # Find firms in source band that we can move
                source_firms = stage2_df[stage2_df['hmrc_band'] == source_band]
                move_count = min(abs(needed_change), len(source_firms), abs(band_adjustments_needed[source_band]))
                
                if move_count > 0:
                    # Randomly select firms to adjust
                    firms_to_adjust = source_firms.sample(move_count, random_state=42)
                    
                    # Generate new turnover values for target band
                    if band == '£1_to_Threshold':
                        new_turnovers = np.random.uniform(1, 84, move_count)
                    elif band == '£Threshold_to_£150k':
                        new_turnovers = np.random.uniform(85, 150, move_count)
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
                    stage2_df.loc[firms_to_adjust.index, 'annual_turnover_k'] = new_turnovers
                    stage2_df.loc[firms_to_adjust.index, 'hmrc_band'] = band
                    
                    needed_change -= move_count
                    band_adjustments_needed[source_band] += move_count
                    adjustment_count += move_count
                    
                    print(f"  Moved {move_count:,} firms from {source_band} to {band}")
    
    print(f"Stage 2 complete: Adjusted {adjustment_count:,} firms")
    
    # Step 7: Final validation and quality checks
    print(f"\nStep 7: Final validation and quality checks...")
    
    # Recalculate band distribution
    stage2_df['hmrc_band'] = stage2_df['annual_turnover_k'].apply(map_to_hmrc_band)
    final_bands = stage2_df['hmrc_band'].value_counts()
    
    print("Final band accuracy after Stage 2:")
    for band, hmrc_target in hmrc_bands.items():
        current_count = final_bands.get(band, 0)
        accuracy = 1 - abs(current_count - hmrc_target) / hmrc_target if hmrc_target > 0 else 1.0
        print(f"  {band:>20}: {current_count:>8,} vs {hmrc_target:>8,} ({accuracy:>6.1%})")
    
    # Sector quality check
    final_sectors = stage2_df['sic_code'].value_counts()
    extreme_sectors = []
    for _, row in hmrc_sector_clean.iterrows():
        sic_code = row['Trade_Sector']
        if sic_code == 'Total' or pd.isna(sic_code) or sic_code == '':
            continue
        
        hmrc_count = row['2023-24']
        current_count = final_sectors.get(sic_code, 0)
        
        if hmrc_count > 0:
            accuracy = 1 - abs(current_count - hmrc_count) / hmrc_count
            if accuracy < 0.5:  # Less than 50% accuracy
                extreme_sectors.append((sic_code, current_count, hmrc_count, accuracy))
    
    if extreme_sectors:
        print(f"\nSectors with <50% accuracy ({len(extreme_sectors)} sectors):")
        for sic_code, current, target, acc in sorted(extreme_sectors, key=lambda x: x[3])[:5]:
            print(f"  SIC {sic_code}: {current:>6,} vs {target:>6,} ({acc:>5.1%})")
    
    print(f"\nFinal: Generated {len(stage2_df):,} firms with two-stage calibration")
    print(f"Target: {target_total:,}")
    
    return stage2_df[['sic_code', 'annual_turnover_k', 'employment']]

def validate_against_hmrc(synthetic_df, hmrc_turnover_df, hmrc_sector_df):
    """Validate synthetic data against both HMRC datasets."""
    print("\n" + "=" * 60)
    print("VALIDATING AGAINST HMRC DATA")
    print("=" * 60)
    
    # === SECTOR VALIDATION ===
    print("\n1. SECTOR VALIDATION:")
    print("-" * 30)
    
    synthetic_by_sic = synthetic_df.groupby('sic_code').size().reset_index(name='synthetic_count')
    
    hmrc_sector_clean = hmrc_sector_df.copy()
    hmrc_sector_clean['Trade_Sector'] = hmrc_sector_clean['Trade_Sector'].astype(str)
    
    sector_comparison = synthetic_by_sic.merge(
        hmrc_sector_clean, 
        left_on='sic_code', 
        right_on='Trade_Sector', 
        how='outer'
    ).fillna(0)
    
    sector_comparison['hmrc_count'] = sector_comparison['2023-24']
    sector_comparison['accuracy'] = np.where(
        sector_comparison['hmrc_count'] > 0,
        1 - abs(sector_comparison['synthetic_count'] - sector_comparison['hmrc_count']) / sector_comparison['hmrc_count'],
        0
    )
    
    sector_accuracy = sector_comparison['accuracy'].mean()
    print(f"Sector accuracy: {sector_accuracy:.1%}")
    
    # === TURNOVER BAND VALIDATION ===
    print("\n2. TURNOVER BAND VALIDATION:")
    print("-" * 30)
    
    def map_to_hmrc_bands(turnover_k):
        """Map turnover to HMRC bands"""
        if pd.isna(turnover_k) or turnover_k <= 0:
            return 'Negative_or_Zero'
        elif turnover_k <= 85:
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
    
    synthetic_df['hmrc_band'] = synthetic_df['annual_turnover_k'].apply(map_to_hmrc_bands)
    band_distribution = synthetic_df['hmrc_band'].value_counts()
    
    hmrc_latest = hmrc_turnover_df.iloc[-1]
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
    
    band_accuracies = []
    print("Band comparison:")
    for band, hmrc_count in hmrc_bands.items():
        synthetic_count = band_distribution.get(band, 0)
        if hmrc_count > 0:
            accuracy = 1 - abs(synthetic_count - hmrc_count) / hmrc_count
        else:
            accuracy = 1.0 if synthetic_count == 0 else 0.0
        
        band_accuracies.append(accuracy)
        print(f"  {band:>20}: {synthetic_count:>8,} vs {hmrc_count:>8,} ({accuracy:>6.1%})")
    
    band_accuracy = np.mean(band_accuracies)
    print(f"\nBand accuracy: {band_accuracy:.1%}")
    
    # === EMPLOYMENT VALIDATION ===
    print("\n3. EMPLOYMENT VALIDATION:")
    print("-" * 30)
    
    def map_to_employment_band(employment):
        """Map employment to ONS employment bands"""
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
    
    # Load ONS employment data for comparison
    project_root = Path(__file__).parent.parent
    ons_employment_path = project_root / 'data' / 'ONS_UK_business_data' / 'firm_employment.csv'
    ons_employment_df = pd.read_csv(ons_employment_path)
    
    # Get ONS employment totals by band
    ons_emp_totals = {}
    emp_bands = ['0-4', '5-9', '10-19', '20-49', '50-99', '100-249', '250+']
    
    for band in emp_bands:
        if band in ons_employment_df.columns:
            # Sum across all sectors (excluding total row)
            sector_rows = ons_employment_df[~ons_employment_df['Description'].str.contains('Total', na=False)]
            ons_emp_totals[band] = sector_rows[band].fillna(0).sum()
        else:
            ons_emp_totals[band] = 0
    
    # Get synthetic employment distribution
    synthetic_df['employment_band'] = synthetic_df['employment'].apply(map_to_employment_band)
    synthetic_emp_counts = synthetic_df['employment_band'].value_counts()
    
    print("Employment band comparison:")
    emp_accuracies = []
    for band in emp_bands:
        ons_count = ons_emp_totals.get(band, 0)
        synthetic_count = synthetic_emp_counts.get(band, 0)
        
        if ons_count > 0:
            accuracy = 1 - abs(synthetic_count - ons_count) / ons_count
        else:
            accuracy = 1.0 if synthetic_count == 0 else 0.0
        
        emp_accuracies.append(accuracy)
        print(f"  {band:>8}: {synthetic_count:>8,} vs {ons_count:>8,} ({accuracy:>6.1%})")
    
    emp_accuracy = np.mean(emp_accuracies)
    print(f"\nEmployment accuracy: {emp_accuracy:.1%}")
    
    # === OVERALL VALIDATION ===
    overall_accuracy = (sector_accuracy + band_accuracy + emp_accuracy) / 3
    
    print(f"\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"Sector accuracy:     {sector_accuracy:.1%}")
    print(f"Turnover accuracy:   {band_accuracy:.1%}")
    print(f"Employment accuracy: {emp_accuracy:.1%}")
    print(f"Overall accuracy:    {overall_accuracy:.1%}")
    
    return overall_accuracy, sector_accuracy, band_accuracy, emp_accuracy

def main():
    """Main function."""
    print("SYNTHETIC FIRM DATA GENERATION")
    print("Using ONS firm_turnover.csv as base")
    print("Validating against HMRC datasets")
    
    # Load data
    ons_df, ons_employment_df, hmrc_turnover_df, hmrc_sector_df, ons_total = load_data()
    
    # Generate calibrated synthetic firms
    synthetic_df = generate_synthetic_firms_calibrated(ons_df, ons_employment_df, ons_total, hmrc_turnover_df, hmrc_sector_df)
    
    # Validate against HMRC
    overall_acc, sector_acc, band_acc, emp_acc = validate_against_hmrc(
        synthetic_df, hmrc_turnover_df, hmrc_sector_df
    )
    
    # Save results
    print(f"\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    output_path = Path(__file__).parent / 'synthetic_firms_turnover.h5'
    
    try:
        synthetic_df.to_hdf(output_path, key='firms', mode='w', complevel=9, complib='zlib')
        file_size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"Saved {len(synthetic_df):,} firms to {output_path}")
        print(f"File size: {file_size_mb:.1f} MB")
        print(f"Overall accuracy: {overall_acc:.1%}")
    except Exception as e:
        print(f"HDF5 save failed: {e}")
        print("Saving as CSV instead...")
        csv_path = Path(__file__).parent / 'synthetic_firms_turnover.csv'
        synthetic_df.to_csv(csv_path, index=False)
        print(f"Saved to: {csv_path}")

if __name__ == "__main__":
    main()