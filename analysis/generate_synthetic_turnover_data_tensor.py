#!/usr/bin/env python3
"""
Tensor-based synthetic firm data generation using PyTorch.
Follows microcalibrate patterns for efficient tensor operations.

This script generates individual firm records from ONS structure,
calibrated to HMRC statistics using PyTorch tensors for efficiency.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SyntheticFirmGenerator:
    """
    Comprehensive synthetic firm data generator following SPI calibration methodology.
    
    Generates complete firm population (VAT-registered + non-VAT) with flags to identify
    which firms appear in HMRC statistics, similar to how Enhanced FRS includes all
    households but flags tax unit visibility.
    """
    
    def __init__(
        self,
        device: str = "cpu",
        batch_size: int = 10000,
        vat_voluntary_rate: float = 0.15,
        random_seed: int = 42
    ):
        """Initialize the comprehensive synthetic firm generator.
        
        Args:
            device: PyTorch device ('cpu', 'cuda', 'mps')
            batch_size: Batch size for tensor operations
            vat_voluntary_rate: Proportion of sub-threshold firms that voluntarily register for VAT
            random_seed: Random seed for reproducibility
        """
        self.device = device
        self.batch_size = batch_size
        self.vat_voluntary_rate = vat_voluntary_rate
        self.random_seed = random_seed
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        logger.info(f"Initialized SyntheticFirmGenerator on device: {device}")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
        """Load ONS and HMRC data files from standardized CSV sources.
        
        Returns:
            Tuple containing ONS turnover data, ONS employment data, 
            HMRC turnover bands, HMRC sector data, and ONS total firm count
        """
        logger.info("Loading data files...")
        
        # Define paths to data files
        project_root = Path(__file__).parent.parent
        ons_path = project_root / 'data' / 'ONS_UK_business_data' / 'firm_turnover.csv'
        ons_employment_path = project_root / 'data' / 'ONS_UK_business_data' / 'firm_employment.csv'
        hmrc_turnover_path = project_root / 'data' / 'HMRC_VAT_annual_statistics' / 'vat_population_by_turnover_band.csv'
        hmrc_sector_path = project_root / 'data' / 'HMRC_VAT_annual_statistics' / 'vat_population_by_sector.csv'
        
        # Load CSV files
        ons_df = pd.read_csv(ons_path)
        ons_employment_df = pd.read_csv(ons_employment_path)
        hmrc_turnover_df = pd.read_csv(hmrc_turnover_path)
        hmrc_sector_df = pd.read_csv(hmrc_sector_path)
        
        logger.info(f"Loaded ONS turnover data: {len(ons_df)} rows")
        logger.info(f"Loaded ONS employment data: {len(ons_employment_df)} rows")
        logger.info(f"Loaded HMRC turnover data: {len(hmrc_turnover_df)} rows")
        logger.info(f"Loaded HMRC sector data: {len(hmrc_sector_df)} rows")
        
        # Extract ONS total
        ons_total_row = ons_df[ons_df['SIC Code'].isna() | (ons_df['SIC Code'] == '')]
        if len(ons_total_row) > 0:
            ons_total = ons_total_row.iloc[0]['Total']
        else:
            sector_rows = ons_df[~ons_df['Description'].str.contains('Total', na=False)]
            ons_total = sector_rows['Total'].sum()
        
        logger.info(f"ONS total firms: {ons_total:,}")
        
        return ons_df, ons_employment_df, hmrc_turnover_df, hmrc_sector_df, ons_total
    
    def generate_base_firms_tensor(self, ons_df: pd.DataFrame) -> Tuple[Tensor, Tensor]:
        """Generate base firm records using tensor operations.
        
        Args:
            ons_df: ONS turnover data
            
        Returns:
            Tuple of (sic_codes_tensor, turnover_tensor)
        """
        logger.info("Generating base firms using tensor operations...")
        
        # ONS turnover band parameters (min, max, midpoint in £k)
        band_params = {
            '0-49': (0, 49, 24.5),
            '50-99': (50, 99, 74.5),
            '100-249': (100, 249, 174.5),
            '250-499': (250, 499, 374.5),
            '500-999': (500, 999, 749.5),
            '1000-4999': (1000, 4999, 2999.5),
            '5000+': (5000, 50000, 15000)
        }
        
        all_sic_codes = []
        all_turnovers = []
        
        # Process each sector
        for _, row in ons_df.iterrows():
            sic_code = row['SIC Code']
            
            # Skip summary rows
            if pd.isna(sic_code) or sic_code == '' or sic_code == 'Total':
                continue
            
            sic_formatted = str(int(sic_code)).zfill(5)
            
            # Generate firms for each turnover band
            for band, (min_val, max_val, midpoint) in band_params.items():
                if band in row and pd.notna(row[band]) and row[band] > 0:
                    count = int(row[band])
                    
                    if count > 0:
                        # Generate turnover values using tensor operations
                        turnovers = self._generate_values_tensor(
                            band, count, min_val, max_val, midpoint
                        )
                        
                        # Store results
                        all_sic_codes.extend([sic_formatted] * count)
                        all_turnovers.extend(turnovers.cpu().numpy())
        
        # Convert to tensors
        sic_codes_tensor = torch.tensor([int(sic) for sic in all_sic_codes], 
                                       dtype=torch.int64, device=self.device)
        turnover_tensor = torch.tensor(all_turnovers, 
                                      dtype=torch.float32, device=self.device)
        
        logger.info(f"Generated {len(all_sic_codes):,} base firms as tensors")
        
        return sic_codes_tensor, turnover_tensor
    
    def _generate_values_tensor(self, band: str, count: int, min_val: float, 
                               max_val: float, midpoint: float) -> Tensor:
        """Generate values within a band using tensor operations.
        
        Args:
            band: Band name
            count: Number of values to generate
            min_val: Minimum value
            max_val: Maximum value
            midpoint: Midpoint for distribution
            
        Returns:
            Tensor of generated values
        """
        if count == 0:
            return torch.empty(0, device=self.device)
        
        # Use different distributions based on band size
        if band in ['5000+', '1000-4999', '500-999']:
            # Log-normal for large bands
            log_mean = torch.log(torch.tensor(midpoint, device=self.device))
            log_std = 0.8 if band == '5000+' else 0.6
            values = torch.normal(log_mean, log_std, (count,), device=self.device).exp()
            values = torch.clamp(values, min_val, max_val)
        else:
            # Beta distribution for smaller bands
            alpha, beta = 2.0, 4.0 if band in ['0-49', '50-99'] else 2.0
            # Use proper beta distribution instead of approximation
            from torch.distributions import Beta
            beta_dist = Beta(alpha, beta)
            uniform_values = beta_dist.sample((count,))
            values = min_val + uniform_values * (max_val - min_val)
        
        return values
    
    def assign_vat_registration_flags(self, turnover_tensor: Tensor) -> Tensor:
        """Assign VAT registration flags to identify HMRC-visible firms.
        
        VAT registration triggers:
        1. Mandatory: Annual turnover > £90k
        2. Voluntary: Random subset of firms below threshold (15% rate)
        
        Args:
            turnover_tensor: Tensor of turnover values
            
        Returns:
            Boolean tensor indicating VAT registration status
        """
        logger.info("Assigning VAT registration flags...")
        
        # Mandatory registration above threshold
        mandatory_vat = turnover_tensor > 90.0
        
        # Voluntary registration below threshold
        below_threshold = turnover_tensor <= 90.0
        n_below_threshold = below_threshold.sum().item()
        
        if n_below_threshold > 0:
            # Random selection for voluntary VAT registration
            voluntary_mask = torch.rand(len(turnover_tensor), device=self.device) < self.vat_voluntary_rate
            voluntary_vat = below_threshold & voluntary_mask
        else:
            voluntary_vat = torch.zeros_like(below_threshold)
        
        vat_registered = mandatory_vat | voluntary_vat
        
        logger.info(f"VAT registration: {mandatory_vat.sum():.0f} mandatory + {voluntary_vat.sum():.0f} voluntary = {vat_registered.sum():.0f} total")
        logger.info(f"Non-VAT registered: {(~vat_registered).sum():.0f} firms")
        
        return vat_registered
    
    def create_spi_style_target_matrix(self, turnover_tensor: Tensor, sic_codes_tensor: Tensor,
                                      hmrc_bands: Dict[str, int], hmrc_sector_df: pd.DataFrame, ons_employment_df: pd.DataFrame, ons_total: int) -> Tuple[Tensor, Tensor]:
        """Create comprehensive target matrix following SPI methodology.
        
        Creates targets for ALL HMRC turnover bands, similar to how Enhanced FRS
        includes all SPI income targets. The optimization will determine which firms
        contribute to which targets based on VAT registration flags.
        
        Args:
            turnover_tensor: Tensor of turnover values
            sic_codes_tensor: Tensor of SIC codes
            hmrc_bands: Dictionary of all HMRC targets by band
            hmrc_sector_df: HMRC sector data for ratio targets
            ons_employment_df: ONS employment data for ratio targets
            ons_total: Total firm count target from ONS
            
        Returns:
            Tuple of (target_matrix, target_values)
        """
        logger.info("Creating SPI-style target matrix following pure methodology...")
        
        n_firms = len(turnover_tensor)
        # Get sector data and calculate ratios
        sector_rows = hmrc_sector_df[hmrc_sector_df['Trade_Sector'] != 'Total'].copy()
        hmrc_total = hmrc_sector_df[hmrc_sector_df['Trade_Sector'] == 'Total']['2023-24'].iloc[0]
        n_sectors = len(sector_rows)
        
        # Get employment data and calculate ratios
        emp_bands = ['0-4', '5-9', '10-19', '20-49', '50-99', '100-249', '250+']
        n_employment_bands = len(emp_bands)
        
        n_targets = 7 + n_sectors + n_employment_bands  # 7 turnover targets + sector ratio targets + employment ratio targets
        
        # Initialize target matrix  
        target_matrix = torch.zeros(n_targets, n_firms, device=self.device)
        
        # Map all firms to HMRC bands
        all_band_indices = self._map_to_hmrc_bands_tensor(turnover_tensor)
        
        # Rows 0-6: Turnover band targets
        # Row 0: £1_to_Threshold - keep ONS structure
        firms_in_threshold = (all_band_indices == 1)
        target_matrix[0, firms_in_threshold] = 1.0
        
        # Row 1: £Threshold_to_£150k - calibrate to HMRC target  
        firms_150k = (all_band_indices == 2)
        target_matrix[1, firms_150k] = 1.0
        
        # Rows 2-6: Individual HMRC bands above £150k
        for i, band_idx in enumerate([3, 4, 5, 6, 7], start=2):  # £150k_to_£300k, etc.
            firms_in_band = (all_band_indices == band_idx)
            target_matrix[i, firms_in_band] = 1.0
        
        # Rows 7 to 7+n_sectors-1: Sector ratio targets
        for i, (_, sector_row) in enumerate(sector_rows.iterrows(), start=7):
            trade_sector = sector_row['Trade_Sector']
            # Convert HMRC Trade_Sector to SIC code (00001 -> 1, 00002 -> 2, etc.)
            sic_code = int(trade_sector)
            
            # Find firms in this sector
            firms_in_sector = (sic_codes_tensor == sic_code)
            target_matrix[i, firms_in_sector] = 1.0
        
        # Generate employment assignments for target matrix (temporary assignments for optimization)
        employment_tensor = self.assign_employment_tensor(n_firms, ons_employment_df)
        
        # Map employment to bands
        def map_employment_to_band_idx(emp_val):
            if emp_val <= 4:
                return 0  # '0-4'
            elif emp_val <= 9:
                return 1  # '5-9'
            elif emp_val <= 19:
                return 2  # '10-19'
            elif emp_val <= 49:
                return 3  # '20-49'
            elif emp_val <= 99:
                return 4  # '50-99'
            elif emp_val <= 249:
                return 5  # '100-249'
            else:
                return 6  # '250+'
        
        employment_band_indices = torch.tensor([map_employment_to_band_idx(emp.item()) for emp in employment_tensor], 
                                             dtype=torch.long, device=self.device)
        
        # Rows 7+n_sectors to 7+n_sectors+n_employment_bands-1: Employment ratio targets
        emp_start_row = 7 + n_sectors
        for band_idx, band_name in enumerate(emp_bands):
            row_idx = emp_start_row + band_idx
            firms_in_emp_band = (employment_band_indices == band_idx)
            target_matrix[row_idx, firms_in_emp_band] = 1.0
        
        # Calculate targets
        # £1_to_Threshold: Use ONS structure (current count from generation)
        ons_threshold_count = firms_in_threshold.sum().item()
        
        # Turnover band targets (absolute numbers)
        turnover_targets = [
            ons_threshold_count,                     # £1_to_Threshold: Keep ONS structure
            hmrc_bands['£Threshold_to_£150k'],       # £Threshold_to_£150k: HMRC
            hmrc_bands['£150k_to_£300k'],            # £150k_to_£300k: HMRC
            hmrc_bands['£300k_to_£500k'],            # £300k_to_£500k: HMRC
            hmrc_bands['£500k_to_£1m'],              # £500k_to_£1m: HMRC
            hmrc_bands['£1m_to_£10m'],               # £1m_to_£10m: HMRC
            hmrc_bands['Greater_than_£10m']          # Greater_than_£10m: HMRC
        ]
        
        # Sector ratio targets (proportions × total synthetic firms)
        current_total_firms = len(turnover_tensor)
        sector_targets = []
        for _, sector_row in sector_rows.iterrows():
            sector_count = sector_row['2023-24']
            sector_ratio = sector_count / hmrc_total  # Calculate ratio from HMRC data
            target_count = sector_ratio * current_total_firms  # Apply ratio to synthetic total
            sector_targets.append(target_count)
        
        # Employment ratio targets (proportions × total synthetic firms from ONS employment data)
        employment_targets = []
        # Get ONS employment totals
        ons_emp_totals = {}
        for band in emp_bands:
            if band in ons_employment_df.columns:
                sector_rows_emp = ons_employment_df[~ons_employment_df['Description'].str.contains('Total', na=False)]
                ons_emp_totals[band] = sector_rows_emp[band].fillna(0).sum()
            else:
                ons_emp_totals[band] = 0
        
        total_ons_emp_firms = sum(ons_emp_totals.values())
        
        for band in emp_bands:
            emp_count = ons_emp_totals[band]
            emp_ratio = emp_count / total_ons_emp_firms if total_ons_emp_firms > 0 else 0
            target_count = emp_ratio * current_total_firms
            employment_targets.append(target_count)
        
        target_values_list = turnover_targets + sector_targets + employment_targets
        
        target_values = torch.tensor(
            target_values_list, 
            dtype=torch.float32, device=self.device
        )
        
        logger.info(f"SPI-style target matrix shape: {target_matrix.shape}")
        logger.info(f"Enhanced SPI targets (turnover + sector ratios + employment ratios):")
        
        # Log turnover targets
        turnover_names = ['£1_to_Threshold (ONS)', '£Threshold_to_£150k (HMRC)', 
                         '£150k_to_£300k (HMRC)', '£300k_to_£500k (HMRC)', 
                         '£500k_to_£1m (HMRC)', '£1m_to_£10m (HMRC)', 'Greater_than_£10m (HMRC)']
        for target_name, target_val in zip(turnover_names, turnover_targets):
            logger.info(f"  {target_name}: {target_val:,.0f}")
        
        # Log sector ratio targets (summary)
        logger.info(f"  Sector ratio targets: {n_sectors} sectors from HMRC data")
        logger.info(f"  Employment ratio targets: {n_employment_bands} bands from ONS data")
        logger.info(f"  Total targets: {len(target_values_list)} (7 turnover + {n_sectors} sector ratios + {n_employment_bands} employment ratios)")
        logger.info(f"  Negative_or_Zero: MANUAL (ONS doesn't have them)")
        
        return target_matrix, target_values
    
    def optimize_weights_tensor(self, target_matrix: Tensor, target_values: Tensor,
                               n_iterations: int = 300, lr: float = 0.01) -> Tensor:
        """Optimize weights using PyTorch to match multiple targets simultaneously.
        
        Uses symmetric relative error loss like Enhanced FRS SPI calibration.
        
        Args:
            target_matrix: Matrix where A[i,j] = contribution of firm j to target i
            target_values: Vector of target values to match
            n_iterations: Number of optimization iterations
            lr: Learning rate
            
        Returns:
            Optimized weights tensor
        """
        logger.info("Starting multi-objective weight optimization...")
        
        _, n_firms = target_matrix.shape
        
        # Initialize log-weights (ensures positive weights)
        log_weights = torch.zeros(n_firms, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([log_weights], lr=lr)
        
        best_loss = float('inf')
        patience = 100
        patience_counter = 0
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Convert to positive weights
            weights = torch.exp(log_weights)
            
            # Calculate predictions: target_matrix @ weights
            predictions = torch.matmul(target_matrix, weights)
            
            # Symmetric relative error loss (robust to different target scales)
            epsilon = 1e-6
            pred_adj = predictions + epsilon
            target_adj = target_values + epsilon
            
            # SRE loss: min(|pred/target - 1|^2, |target/pred - 1|^2)
            error_1 = ((pred_adj / target_adj) - 1) ** 2
            error_2 = ((target_adj / pred_adj) - 1) ** 2
            sre_loss = torch.minimum(error_1, error_2)
            
            # Apply importance weights: turnover targets (0-6) most important, sector targets (7+) and employment targets less important
            importance_weights = torch.ones_like(sre_loss)
            importance_weights[:7] = 10.0  # 10x weight for turnover targets
            # Calculate where employment targets start (7 + number of sector targets)
            # Since we don't know exactly how many sector targets there are, we'll assume employment targets are the last 7
            n_total_targets = len(sre_loss)
            if n_total_targets > 14:  # If we have more than 14 targets, we have employment targets
                n_emp_targets = 7  # 7 employment bands
                emp_start_idx = n_total_targets - n_emp_targets
                importance_weights[7:emp_start_idx] = 1.0  # 1x weight for sector targets  
                importance_weights[emp_start_idx:] = 1.0  # 1x weight for employment targets
            else:
                importance_weights[7:] = 1.0  # 1x weight for sector targets only (no employment targets)
            
            weighted_loss = sre_loss * importance_weights
            total_loss = torch.mean(weighted_loss)
            
            # Add regularization to prevent extreme weights
            reg_loss = 0.01 * torch.mean((log_weights ** 2))
            total_loss += reg_loss
            
            total_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_([log_weights], max_norm=1.0)
            
            optimizer.step()
            
            # Early stopping
            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if iteration % 100 == 0:
                logger.info(f"Iteration {iteration}: Loss = {total_loss.item():.6f}")
            
            if patience_counter > patience:
                logger.info(f"Early stopping at iteration {iteration}")
                break
        
        final_weights = torch.exp(log_weights)
        final_predictions = torch.matmul(target_matrix, final_weights)
        
        logger.info("Optimization complete:")
        target_names = ['Negative_or_Zero', '£1_to_Threshold', '£Threshold_to_£150k', '£150k_to_£300k', 
                       '£300k_to_£500k', '£500k_to_£1m', '£1m_to_£10m', 'Greater_than_£10m']
        for i, (pred, target, name) in enumerate(zip(final_predictions, target_values, target_names)):
            if target > 0:  # Only log targets we're actually trying to match
                accuracy = 1 - abs(pred - target) / target
                logger.info(f"  {name}: {pred:.0f} vs {target:.0f} ({accuracy:.1%})")
            else:
                logger.info(f"  {name}: SKIPPED (pred: {pred:.0f})")
        
        return final_weights.detach()
    
    def apply_backup_style_calibration(self, base_sic_codes: Tensor, base_turnover: Tensor,
                                     weights_tensor: Tensor, hmrc_bands: Dict[str, int]) -> Tuple[Tensor, Tensor, Tensor]:
        """Apply calibration following backup.py approach.
        
        Key elements from backup.py:
        1. Keep ALL base firms (no sampling)
        2. Add zero/negative turnover firms manually (lines 502-525)
        3. Set £1_to_Threshold weights = 1.0 (lines 490-494)
        4. Apply calibration weights to other bands
        
        Args:
            base_sic_codes: Original base SIC codes
            base_turnover: Original base turnover values  
            weights_tensor: Calibration weights from optimization
            hmrc_bands: HMRC target bands
            
        Returns:
            Tuple of final (sic_codes, turnover, weights) tensors
        """
        logger.info("Applying backup.py style calibration...")
        
        all_final_firms = []
        
        # Step 1: Add ALL base firms with their weights
        hmrc_band_indices = self._map_to_hmrc_bands_tensor(base_turnover)
        
        logger.info(f"Adding {len(base_sic_codes):,} base firms...")
        for i in range(len(base_sic_codes)):
            band_idx = hmrc_band_indices[i].item()
            
            # Use calibrated weights from optimization for all bands
            weight = weights_tensor[i].item()
            
            all_final_firms.append({
                'sic_code': base_sic_codes[i].item(),
                'annual_turnover_k': base_turnover[i].item(),
                'weight': weight
            })
        
        # Step 2: Manually add zero/negative turnover firms (backup.py lines 502-525)
        negative_zero_target = hmrc_bands['Negative_or_Zero']
        
        if negative_zero_target > 0:
            # Get current sector distribution for proportional allocation
            unique_sics, counts = torch.unique(base_sic_codes, return_counts=True)
            total_current_firms = len(base_sic_codes)
            
            logger.info(f"Manually adding {negative_zero_target:,} zero turnover firms from HMRC...")
            zero_firms_count = 0
            
            for sic, count in zip(unique_sics, counts):
                sector_weight = count.float() / total_current_firms
                firms_for_sector = int((negative_zero_target * sector_weight).item())
                if firms_for_sector > 0:
                    for _ in range(firms_for_sector):
                        all_final_firms.append({
                            'sic_code': sic.item(),
                            'annual_turnover_k': 0.0,
                            'weight': 1.0
                        })
                        zero_firms_count += 1
            
            logger.info(f"Added {zero_firms_count:,} zero turnover firms")
        
        # Convert back to tensors
        if all_final_firms:
            final_sics = torch.tensor([f['sic_code'] for f in all_final_firms], 
                                    dtype=torch.int64, device=self.device)
            final_turnover = torch.tensor([f['annual_turnover_k'] for f in all_final_firms], 
                                        dtype=torch.float32, device=self.device)
            final_weights = torch.tensor([f['weight'] for f in all_final_firms], 
                                       dtype=torch.float32, device=self.device)
            
            logger.info(f"Final dataset: {len(all_final_firms):,} firms")
            logger.info(f"Total weighted population: {final_weights.sum():.0f}")
        else:
            # Empty tensors if no firms generated
            final_sics = torch.empty(0, dtype=torch.int64, device=self.device)
            final_turnover = torch.empty(0, dtype=torch.float32, device=self.device) 
            final_weights = torch.empty(0, dtype=torch.float32, device=self.device)
        
        return final_sics, final_turnover, final_weights
    
    def _map_to_hmrc_bands_tensor(self, turnover_tensor: Tensor) -> Tensor:
        """Map turnover values to HMRC band indices using tensor operations.
        
        Args:
            turnover_tensor: Tensor of turnover values
            
        Returns:
            Tensor of HMRC band indices (0-7 for 8 HMRC bands)
        """
        # Initialize with highest band (Greater_than_£10m = 7)
        band_indices = torch.full_like(turnover_tensor, 7, dtype=torch.long)
        
        # Assign bands based on turnover thresholds
        band_indices = torch.where(turnover_tensor <= 0, 0, band_indices)  # Negative_or_Zero
        band_indices = torch.where((turnover_tensor > 0) & (turnover_tensor <= 90), 1, band_indices)  # £1_to_Threshold
        band_indices = torch.where((turnover_tensor > 90) & (turnover_tensor <= 150), 2, band_indices)  # £Threshold_to_£150k
        band_indices = torch.where((turnover_tensor > 150) & (turnover_tensor <= 300), 3, band_indices)  # £150k_to_£300k
        band_indices = torch.where((turnover_tensor > 300) & (turnover_tensor <= 500), 4, band_indices)  # £300k_to_£500k
        band_indices = torch.where((turnover_tensor > 500) & (turnover_tensor <= 1000), 5, band_indices)  # £500k_to_£1m
        band_indices = torch.where((turnover_tensor > 1000) & (turnover_tensor <= 10000), 6, band_indices)  # £1m_to_£10m
        
        return band_indices
    
    
    def assign_employment_tensor(self, num_firms: int, 
                               ons_employment_df: pd.DataFrame) -> Tensor:
        """Assign employment using tensor operations and ONS distribution.
        
        Args:
            num_firms: Number of firms to assign employment to
            ons_employment_df: ONS employment data
            
        Returns:
            Tensor of employment values
        """
        logger.info("Assigning employment using tensor operations...")
        
        # Employment bands and parameters
        emp_bands = ['0-4', '5-9', '10-19', '20-49', '50-99', '100-249', '250+']
        band_params = {
            '0-4': (1, 4, 2.5),
            '5-9': (5, 9, 7),
            '10-19': (10, 19, 14.5),
            '20-49': (20, 49, 34.5),
            '50-99': (50, 99, 74.5),
            '100-249': (100, 249, 174.5),
            '250+': (250, 2000, 400)
        }
        
        # Calculate ONS employment band totals
        total_ons_counts = {}
        for band in emp_bands:
            if band in ons_employment_df.columns:
                sector_rows = ons_employment_df[~ons_employment_df['Description'].str.contains('Total', na=False)]
                total_ons_counts[band] = int(sector_rows[band].fillna(0).sum())
            else:
                total_ons_counts[band] = 0
        
        total_ons_firms = sum(total_ons_counts.values())
        
        # Calculate target counts for each band
        employment_values = []
        for band in emp_bands:
            target_count = int(round(num_firms * total_ons_counts[band] / total_ons_firms))
            if target_count > 0:
                min_val, max_val, midpoint = band_params[band]
                
                # Generate employment values using tensor operations
                if band == '0-4':
                    # Uniform for micro businesses
                    values = torch.randint(1, 5, (target_count,), device=self.device)
                elif band == '250+':
                    # Log-normal for large firms
                    log_mean = torch.log(torch.tensor(midpoint, device=self.device))
                    values = torch.normal(log_mean, 0.8, (target_count,), device=self.device).exp()
                    values = torch.clamp(values, min_val, max_val).round()
                else:
                    # Beta distribution for others
                    uniform = torch.rand(target_count, device=self.device)
                    beta_values = uniform.pow(0.5) * (1 - uniform).pow(2.0)  # Beta approximation
                    values = min_val + beta_values * (max_val - min_val)
                    values = values.round()
                
                employment_values.extend(values.cpu().numpy())
        
        # Shuffle and pad/trim to exact size
        np.random.shuffle(employment_values)
        if len(employment_values) < num_firms:
            employment_values.extend([1] * (num_firms - len(employment_values)))
        elif len(employment_values) > num_firms:
            employment_values = employment_values[:num_firms]
        
        employment_tensor = torch.tensor(employment_values, 
                                       dtype=torch.float32, device=self.device)
        
        logger.info(f"Assigned employment to {num_firms:,} firms")
        
        return employment_tensor
    
    def validate_comprehensive_accuracy(self, synthetic_df: pd.DataFrame, hmrc_bands: Dict[str, int],
                                       ons_total: int, ons_employment_df: pd.DataFrame, hmrc_sector_df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
        """Comprehensive validation against HMRC VAT, ONS total, employment, sector, and employment ratio targets.
        
        Args:
            synthetic_df: Generated synthetic data with VAT flags
            hmrc_bands: HMRC VAT-registered firm targets by band
            ons_total: ONS total firm count target
            ons_employment_df: ONS employment data for validation
            hmrc_sector_df: HMRC sector data for validation
            
        Returns:
            Tuple of (hmrc_accuracy, ons_accuracy, employment_accuracy, sector_accuracy, emp_ratio_accuracy)
        """
        logger.info("Comprehensive validation against HMRC, ONS, and employment targets...")
        
        # 1. HMRC VAT Firm Validation (VAT-registered firms only)
        def map_to_hmrc_band(turnover_k):
            if turnover_k <= 0:
                return 'Negative_or_Zero'
            elif turnover_k <= 90:
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
        
        # Map all firms to HMRC bands for validation
        synthetic_df['hmrc_band'] = synthetic_df['annual_turnover_k'].apply(map_to_hmrc_band)
        all_bands = synthetic_df.groupby('hmrc_band')['weight'].sum().round().astype(int)
        
        print("\n" + "="*80)
        print("SIMPLE SPI METHODOLOGY VALIDATION")
        print("£1_to_Threshold: ONS | £Threshold+: HMRC | Negative_or_Zero: HMRC")
        print("="*80)
        print(f"{'Band':>25} {'Synthetic':>12} {'Target':>12} {'Source':>8} {'Accuracy':>10}")
        print("-" * 75)
        
        hmrc_accuracies = []
        # Negative_or_Zero: HMRC target 
        band = 'Negative_or_Zero'
        synthetic = all_bands.get(band, 0)
        target = hmrc_bands[band]
        accuracy = 1 - abs(synthetic - target) / target if target > 0 else (1.0 if synthetic == 0 else 0.0)
        hmrc_accuracies.append(accuracy)
        status = "✓" if accuracy > 0.90 else "⚠" if accuracy > 0.80 else "✗"
        print(f"  {status} {band:>22}: {synthetic:>10,} vs {target:>10,} {'HMRC':>8} ({accuracy:>6.1%})")
        
        # £1_to_Threshold: ONS structure (no specific target - just report)
        band = '£1_to_Threshold'
        synthetic = all_bands.get(band, 0)
        hmrc_ref = hmrc_bands[band]
        print(f"  ○ {band:>22}: {synthetic:>10,} vs {hmrc_ref:>10,} {'Based on ONS':>8} ")
        
        # £Threshold_to_£150k and above: HMRC targets
        hmrc_above_bands = ['£Threshold_to_£150k', '£150k_to_£300k', '£300k_to_£500k', 
                           '£500k_to_£1m', '£1m_to_£10m', 'Greater_than_£10m']
        for band in hmrc_above_bands:
            synthetic = all_bands.get(band, 0)
            target = hmrc_bands[band]
            accuracy = 1 - abs(synthetic - target) / target if target > 0 else (1.0 if synthetic == 0 else 0.0)
            hmrc_accuracies.append(accuracy)
            status = "✓" if accuracy > 0.90 else "⚠" if accuracy > 0.80 else "✗"
            print(f"  {status} {band:>22}: {synthetic:>10,} vs {target:>10,} {'HMRC':>8} ({accuracy:>6.1%})")
        
        hmrc_accuracy = np.mean(hmrc_accuracies)
        print("-" * 75)
        print(f"HMRC CALIBRATION ACCURACY: {hmrc_accuracy:.1%}")
        
        # 2. ONS Total Population Validation (All firms)
        total_synthetic = synthetic_df['weight'].sum()
        ons_accuracy = 1 - abs(total_synthetic - ons_total) / ons_total
        
        print(f"\nONS TOTAL POPULATION VALIDATION:")
        print("-" * 50)
        print(f"Total Synthetic: {total_synthetic:,.0f}")
        print(f"ONS Target:      {ons_total:,}")
        print(f"Difference:      {total_synthetic - ons_total:+,.0f}")
        print(f"ONS Accuracy:    {ons_accuracy:.1%}")
        
        print(f"\nTOTAL POPULATION: {total_synthetic:,.0f} firms")
        
        # Employment Band Validation
        def map_to_employment_band(employment):
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
        
        # Get ONS employment totals
        emp_bands = ['0-4', '5-9', '10-19', '20-49', '50-99', '100-249', '250+']
        ons_emp_totals = {}
        
        for band in emp_bands:
            if band in ons_employment_df.columns:
                sector_rows = ons_employment_df[~ons_employment_df['Description'].str.contains('Total', na=False)]
                ons_emp_totals[band] = sector_rows[band].fillna(0).sum()
            else:
                ons_emp_totals[band] = 0
        
        synthetic_df['employment_band'] = synthetic_df['employment'].apply(map_to_employment_band)
        synthetic_emp = synthetic_df.groupby('employment_band')['weight'].sum().round().astype(int)
        
        print(f"\nEMPLOYMENT BAND VALIDATION RESULTS:")
        print("-" * 65)
        print(f"{'Band':>8} {'Synthetic':>10} {'ONS Target':>11} {'Accuracy':>10}")
        print("-" * 65)
        
        emp_accuracies = []
        emp_accuracy_95_plus = 0
        emp_accuracy_90_95 = 0
        emp_accuracy_80_90 = 0 
        emp_accuracy_below_80 = 0
        
        for band in emp_bands:
            target = ons_emp_totals.get(band, 0)
            synthetic = synthetic_emp.get(band, 0)
            accuracy = 1 - abs(synthetic - target) / target if target > 0 else (1.0 if synthetic == 0 else 0.0)
            emp_accuracies.append(accuracy)
            
            # Count accuracy levels
            if accuracy >= 0.95:
                emp_accuracy_95_plus += 1
                status = "✓"
            elif accuracy >= 0.90:
                emp_accuracy_90_95 += 1  
                status = "✓"
            elif accuracy >= 0.80:
                emp_accuracy_80_90 += 1
                status = "⚠"
            else:
                emp_accuracy_below_80 += 1
                status = "✗"
            
            print(f"  {status} {band:>6}: {synthetic:>8,} vs {target:>8,} ({accuracy:>6.1%})")
        
        employment_accuracy = np.mean(emp_accuracies)
        n_emp_bands = len(emp_accuracies)
        print("-" * 65)
        print(f"EMPLOYMENT OVERALL ACCURACY: {employment_accuracy:.1%}")
        print(f"Accuracy breakdown: ≥95%: {emp_accuracy_95_plus}/{n_emp_bands}, 90-95%: {emp_accuracy_90_95}/{n_emp_bands}, 80-90%: {emp_accuracy_80_90}/{n_emp_bands}, <80%: {emp_accuracy_below_80}/{n_emp_bands}")
        
        # 4. Sector Target Validation (subset of HMRC sectors used in optimization)
        print(f"\nSECTOR TARGET VALIDATION RESULTS:")
        print("-" * 65)
        
        # Get sector data and calculate expected targets
        sector_rows = hmrc_sector_df[hmrc_sector_df['Trade_Sector'] != 'Total'].copy()
        hmrc_total = hmrc_sector_df[hmrc_sector_df['Trade_Sector'] == 'Total']['2023-24'].iloc[0]
        current_total_firms = len(synthetic_df)
        
        # Calculate synthetic sector distribution
        synthetic_df['sic_numeric'] = synthetic_df['sic_code'].astype(int)
        synthetic_sectors = synthetic_df.groupby('sic_numeric')['weight'].sum().round().astype(int)
        
        sector_accuracies = []
        accuracy_95_plus = 0
        accuracy_90_95 = 0
        accuracy_80_90 = 0 
        accuracy_below_80 = 0
        
        print(f"{'SIC':>5} {'Synthetic':>10} {'Expected':>10} {'Accuracy':>10}")
        print("-" * 65)
        
        for _, sector_row in sector_rows.iterrows():
            sic_code = int(sector_row['Trade_Sector'])
            sector_count = sector_row['2023-24']
            sector_ratio = sector_count / hmrc_total
            expected_count = sector_ratio * current_total_firms
            
            synthetic_count = synthetic_sectors.get(sic_code, 0)
            accuracy = 1 - abs(synthetic_count - expected_count) / expected_count if expected_count > 0 else (1.0 if synthetic_count == 0 else 0.0)
            sector_accuracies.append(accuracy)
            
            # Count accuracy levels
            if accuracy >= 0.95:
                accuracy_95_plus += 1
                status = "✓"
            elif accuracy >= 0.90:
                accuracy_90_95 += 1  
                status = "✓"
            elif accuracy >= 0.80:
                accuracy_80_90 += 1
                status = "⚠"
            else:
                accuracy_below_80 += 1
                status = "✗"
            
            print(f"  {status} {sic_code:>3}: {synthetic_count:>8,} vs {expected_count:>8,.0f} ({accuracy:>6.1%})")
        
        sector_accuracy = np.mean(sector_accuracies) if sector_accuracies else 0.0
        n_sectors = len(sector_accuracies)
        
        print("-" * 65)
        print(f"SECTOR OVERALL ACCURACY: {sector_accuracy:.1%}")
        print(f"Accuracy breakdown: ≥95%: {accuracy_95_plus}/{n_sectors}, 90-95%: {accuracy_90_95}/{n_sectors}, 80-90%: {accuracy_80_90}/{n_sectors}, <80%: {accuracy_below_80}/{n_sectors}")
        
        # 5. Employment Ratio Target Validation (subset of employment ratio targets used in optimization)
        print(f"\nEMPLOYMENT RATIO TARGET VALIDATION RESULTS:")
        print("-" * 65)
        
        # Get employment data and calculate expected ratios
        emp_bands = ['0-4', '5-9', '10-19', '20-49', '50-99', '100-249', '250+']
        ons_emp_totals = {}
        for band in emp_bands:
            if band in ons_employment_df.columns:
                sector_rows_emp = ons_employment_df[~ons_employment_df['Description'].str.contains('Total', na=False)]
                ons_emp_totals[band] = sector_rows_emp[band].fillna(0).sum()
            else:
                ons_emp_totals[band] = 0
        
        total_ons_emp_firms = sum(ons_emp_totals.values())
        
        # Calculate synthetic employment ratio distribution
        def map_employment_to_band(employment):
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
        
        synthetic_df['employment_band'] = synthetic_df['employment'].apply(map_employment_to_band)
        synthetic_emp_ratios = synthetic_df.groupby('employment_band')['weight'].sum().round().astype(int)
        
        emp_ratio_accuracies = []
        emp_ratio_accuracy_95_plus = 0
        emp_ratio_accuracy_90_95 = 0
        emp_ratio_accuracy_80_90 = 0 
        emp_ratio_accuracy_below_80 = 0
        
        print(f"{'Band':>8} {'Synthetic':>10} {'Expected':>10} {'Accuracy':>10}")
        print("-" * 65)
        
        for band in emp_bands:
            emp_count = ons_emp_totals[band]
            emp_ratio = emp_count / total_ons_emp_firms if total_ons_emp_firms > 0 else 0
            expected_count = emp_ratio * current_total_firms
            
            synthetic_count = synthetic_emp_ratios.get(band, 0)
            accuracy = 1 - abs(synthetic_count - expected_count) / expected_count if expected_count > 0 else (1.0 if synthetic_count == 0 else 0.0)
            emp_ratio_accuracies.append(accuracy)
            
            # Count accuracy levels
            if accuracy >= 0.95:
                emp_ratio_accuracy_95_plus += 1
                status = "✓"
            elif accuracy >= 0.90:
                emp_ratio_accuracy_90_95 += 1  
                status = "✓"
            elif accuracy >= 0.80:
                emp_ratio_accuracy_80_90 += 1
                status = "⚠"
            else:
                emp_ratio_accuracy_below_80 += 1
                status = "✗"
            
            print(f"  {status} {band:>6}: {synthetic_count:>8,} vs {expected_count:>8,.0f} ({accuracy:>6.1%})")
        
        emp_ratio_accuracy = np.mean(emp_ratio_accuracies) if emp_ratio_accuracies else 0.0
        n_emp_ratio_bands = len(emp_ratio_accuracies)
        
        print("-" * 65)
        print(f"EMPLOYMENT RATIO OVERALL ACCURACY: {emp_ratio_accuracy:.1%}")
        print(f"Accuracy breakdown: ≥95%: {emp_ratio_accuracy_95_plus}/{n_emp_ratio_bands}, 90-95%: {emp_ratio_accuracy_90_95}/{n_emp_ratio_bands}, 80-90%: {emp_ratio_accuracy_80_90}/{n_emp_ratio_bands}, <80%: {emp_ratio_accuracy_below_80}/{n_emp_ratio_bands}")
        
        # Overall summary (now including sector accuracy and employment ratio accuracy)
        overall_accuracy = (hmrc_accuracy + ons_accuracy + employment_accuracy + sector_accuracy + emp_ratio_accuracy) / 5
        print(f"\n" + "="*80)
        print("SIMPLE SPI METHODOLOGY SUMMARY")
        print("="*80)
        print(f"HMRC calibration accuracy: {hmrc_accuracy:.1%} (HMRC-targeted bands)")
        print(f"ONS total accuracy:        {ons_accuracy:.1%} (total population)")
        # print(f"Employment accuracy:       {employment_accuracy:.1%} (employment distribution)")
        print(f"Sector accuracy:           {sector_accuracy:.1%} (sector distribution)")
        print(f"Employment accuracy:       {emp_ratio_accuracy:.1%} (employment distribution)")
        print(f"Combined accuracy:         {overall_accuracy:.1%}")
        
        print(f"\nSIMPLE SPI METHODOLOGY VALIDATION:")
        if overall_accuracy >= 0.95:
            print("✓ EXCELLENT: Simple SPI synthetic data very closely matches all targets")
        elif overall_accuracy >= 0.90:
            print("✓ GOOD: Simple SPI synthetic data closely matches all targets")
        elif overall_accuracy >= 0.80:
            print("⚠ FAIR: Simple SPI synthetic data reasonably matches targets")
        else:
            print("✗ POOR: Simple SPI synthetic data needs improvement")
        
        print(f"\nMETHODOLOGY SUMMARY:")
        print(f"• Start with ONS synthetic data: {ons_total:,} firms")
        print(f"• Total synthetic population: {total_synthetic:,.0f} firms")
        print(f"• Calibrate £Threshold+ to HMRC targets")
        print(f"• Keep £1_to_Threshold from ONS structure")
        print(f"• Add Negative_or_Zero manually from HMRC")
        
        return hmrc_accuracy, ons_accuracy, employment_accuracy, sector_accuracy, emp_ratio_accuracy
    
    

    def generate_synthetic_firms(self) -> pd.DataFrame:
        """Main function to generate comprehensive synthetic firms population.
        
        Creates complete firm dataset with VAT registration flags, following
        SPI calibration methodology for Enhanced FRS generation.
        
        Returns:
            DataFrame with synthetic firms data including VAT registration flags
        """
        logger.info("Starting comprehensive synthetic firm generation...")
        
        # Load data
        ons_df, ons_employment_df, hmrc_turnover_df, hmrc_sector_df, ons_total = self.load_data()
        
        # Extract HMRC targets (VAT-registered firms only)
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
        
        logger.info(f"Target populations:")
        logger.info(f"  ONS total firms: {ons_total:,} (includes all businesses)")
        logger.info(f"  HMRC VAT firms: {sum(hmrc_bands.values()):,} (VAT-registered only)")
        
        # Generate base firms from ONS structure
        base_sic_codes, base_turnover = self.generate_base_firms_tensor(ons_df)
        
        # Create target matrix for multi-objective optimization (enhanced SPI methodology)
        target_matrix, target_values = self.create_spi_style_target_matrix(
            base_turnover, base_sic_codes, hmrc_bands, hmrc_sector_df, ons_employment_df, ons_total
        )
        
        # Optimize weights to match calibration targets
        weights_tensor = self.optimize_weights_tensor(target_matrix, target_values)
        
        # Apply backup.py style calibration (add zero firms manually)
        sic_codes_tensor, turnover_tensor, weights_tensor = self.apply_backup_style_calibration(
            base_sic_codes, base_turnover, weights_tensor, hmrc_bands
        )
        
        # Assign employment to final firms
        employment_tensor = self.assign_employment_tensor(len(sic_codes_tensor), ons_employment_df)
        
        # Convert to DataFrame
        logger.info("Converting to comprehensive DataFrame...")
        sic_codes_np = sic_codes_tensor.cpu().numpy().astype(int)
        synthetic_df = pd.DataFrame({
            'sic_code': [str(sic).zfill(5) for sic in sic_codes_np],
            'annual_turnover_k': turnover_tensor.cpu().numpy(),
            'employment': employment_tensor.cpu().numpy().astype(int),
            'weight': weights_tensor.cpu().numpy()
        })
        
        logger.info(f"Generated simplified firm dataset:")
        logger.info(f"  Total firms: {len(synthetic_df):,}")
        logger.info(f"  Weighted population: {synthetic_df['weight'].sum():,.0f}")
        
        # Comprehensive validation
        self.validate_comprehensive_accuracy(synthetic_df, hmrc_bands, ons_total, ons_employment_df, hmrc_sector_df)
        
        return synthetic_df


def main():
    """Main execution function."""
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    logger.info("TENSOR-BASED SYNTHETIC FIRM DATA GENERATION")
    logger.info("Using PyTorch tensors following microcalibrate patterns")
    
    # Initialize comprehensive generator
    generator = SyntheticFirmGenerator(
        device="cpu",  # Use CPU for compatibility
        vat_voluntary_rate=0.15  # 15% of sub-threshold firms voluntarily register for VAT
    )
    
    # Generate synthetic data
    synthetic_df = generator.generate_synthetic_firms()
    
    # Save results
    output_path = Path(__file__).parent / 'synthetic_firms_turnover_tensor.csv'
    synthetic_df.to_csv(output_path, index=False)
    
    file_size_mb = output_path.stat().st_size / 1024 / 1024
    logger.info(f"Saved {len(synthetic_df):,} firms to {output_path}")
    logger.info(f"File size: {file_size_mb:.1f} MB")
    logger.info(f"Columns: {list(synthetic_df.columns)}")
    
    # Show sample
    logger.info("Sample data:")
    print(synthetic_df.head())
    
    logger.info("Tensor-based generation complete!")


if __name__ == "__main__":
    main()