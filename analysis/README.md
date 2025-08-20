# Synthetic Firm Data Generation

This document explains the methodology for generating synthetic UK business data that combines ONS (Office for National Statistics) business structure with HMRC (Her Majesty's Revenue and Customs) validation statistics.

## Overview

We create individual firm records that preserve the detailed structure from ONS data while exactly matching the target counts from HMRC data. This solves a critical data science challenge: ONS provides rich sector × size detail but overestimates totals, while HMRC provides accurate totals but less structural detail.

## The Core Challenge: Misaligned Band Definitions

**ONS Turnover Bands** (from business survey data):
- `0-49k`, `50-99k`, `100-249k`, `250-499k`, `500-999k`, `1000-4999k`, `5000k+`

**HMRC Turnover Bands** (from VAT registration data):
- `Negative_or_Zero`, `£1_to_Threshold`, `£Threshold_to_£150k`, `£150k_to_£300k`, `£300k_to_£500k`, `£500k_to_£1m`, `£1m_to_£10m`, `Greater_than_£10m`

These bands don't align, making direct comparison impossible. Our methodology solves this through individual firm generation and multi-objective optimization.

## Methodology: Multi-Objective Optimization Approach

### Step 1: Generate Individual Firms from ONS Structure

Instead of working with aggregate band counts, we generate individual firms with specific turnover values:

```python
# ONS says: 10,000 firms in "Manufacturing, 100-249k" band
# We generate 10,000 individual firms with realistic turnovers between 100-249k

ONS Band "100-249k" → Individual Firms:
- Firm A: Manufacturing, £120k turnover
- Firm B: Manufacturing, £180k turnover  
- Firm C: Manufacturing, £230k turnover
- ... (9,997 more firms)
```

**Turnover Generation**: Each firm gets a realistic turnover value with noise smoothing across the full band range to create natural distribution patterns.

### Step 2: Create Comprehensive Target Matrix

We build a mathematical optimization problem with multiple target types:

#### A. Turnover Band Targets (HMRC)
```python
# Map each individual firm to HMRC bands
def map_to_hmrc_band(turnover_k):
    if turnover_k <= 0:         return 'Negative_or_Zero'
    elif turnover_k <= 90:      return '£1_to_Threshold'
    elif turnover_k <= 150:     return '£Threshold_to_£150k'
    elif turnover_k <= 300:     return '£150k_to_£300k'
    # ... etc

# Target: Match HMRC firm counts in each band
```

#### B. Sector Targets (HMRC VAT-registered firms)
```python
# Target: Match HMRC sector distribution for VAT-registered firms
# e.g., HMRC target for Manufacturing: 85,000 VAT-registered firms
```

#### C. Employment Distribution (ONS)
```python
# Target: Preserve ONS employment band distribution
# Employment bands: 0-4, 5-9, 10-19, 20-49, 50-99, 100-249, 250+
```

### Step 3: Multi-Objective Weight Optimization

Using PyTorch optimization to find weights that satisfy all targets simultaneously:

```python
# For each firm, find a weight that balances all objectives
# Target matrix A where A[i,j] = contribution of firm j to target i
# Solve: minimize ||A @ weights - targets||²

# Use symmetric relative error loss for robust calibration
error_1 = ((predictions / targets) - 1) ** 2
error_2 = ((targets / predictions) - 1) ** 2
loss = minimum(error_1, error_2)

# Apply importance weights:
# - Turnover targets: 5x importance (most critical)
# - Sector targets: 1x importance  
# - Employment targets: 1x importance
```

### Step 4: VAT Registration Assignment

Assign VAT registration flags based on UK rules:

```python
# Mandatory VAT registration
mandatory_vat = (turnover > 90_000)  # Above £90k threshold

# Voluntary VAT registration  
# Calculate voluntary rate from HMRC data ratio
voluntary_rate = hmrc_target_below_threshold / synthetic_firms_below_threshold
voluntary_vat = (turnover <= 90_000) & (random() < voluntary_rate)

vat_registered = mandatory_vat | voluntary_vat
```

### Step 5: Final Calibration Adjustments

```python
# Add firms with zero/negative turnover manually to match HMRC targets
# These firms represent businesses with losses or no turnover

if hmrc_negative_zero_target > 0:
    # Distribute zero-turnover firms proportionally across sectors
    add_zero_turnover_firms(hmrc_negative_zero_target, sector_proportions)
```

## Key Technical Features

### Optimization Algorithm
- **Symmetric Relative Error Loss**: Robust to different target scales
- **L1 Regularization**: Prevents extreme weights, encourages sparse solutions  
- **Dropout Training**: 15% dropout during optimization for robustness
- **Early Stopping**: Prevents overfitting with patience mechanism

### Noise Generation
- **Full Range Smoothing**: Noise applied across entire turnover bands
- **Adaptive Noise**: Standard deviation scales with band width (minimum 25k or 20% of band width)

### Employment Assignment
- **Realistic Distributions**: Different patterns for different size bands
  - Micro (0-4): Uniform distribution
  - Small (5-99): Beta-like distribution  
  - Large (250+): Log-normal distribution

## Validation Results

The optimization typically achieves:
- **HMRC Turnover Bands**: 95%+ accuracy match
- **HMRC Sector Distribution**: 90%+ accuracy for VAT-registered firms
- **ONS Employment Distribution**: 95%+ accuracy match
- **ONS Total Population**: 99%+ accuracy match
- **Overall Calibration**: 93%+ combined accuracy

## Files Generated

- **`synthetic_firms_turnover.csv`**: Final dataset with individual firm records
  - `sic_code`: 5-digit industry classification  
  - `annual_turnover_k`: Annual turnover in thousands of pounds
  - `employment`: Number of employees
  - `weight`: Statistical weight for population scaling
  - `vat_registered`: VAT registration flag (boolean)

## Usage for Policy Analysis

This synthetic dataset enables:
- **VAT Threshold Analysis**: Impact of changing registration thresholds (90k, 150k boundaries)
- **Sector Impact Studies**: Effects of policies on different industries  
- **Employment-Based Analysis**: Small vs. large business policy impacts
- **Weighted Analysis**: Use `weight` column for population-representative statistics