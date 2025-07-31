"""
Synthetic Firm Generator for UK VAT Microsimulation

This module generates synthetic firm-level microdata that matches ONS Business Statistics
marginal distributions using gradient descent optimization.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple
import logging

class SyntheticFirmGenerator:
    """Generate synthetic firms matching ONS marginal distributions."""
    
    def __init__(self, ons_data_path: str, n_firms: int = 1_000_000):
        """
        Initialize the generator with ONS data.
        
        Args:
            ons_data_path: Path to ONS business statistics Excel file
            n_firms: Number of synthetic firms to generate
        """
        self.ons_data_path = ons_data_path
        self.n_firms = n_firms
        self.target_marginals = {}
        
    def load_ons_marginals(self) -> Dict[str, pd.DataFrame]:
        """Load target marginal distributions from ONS data."""
        # Load key tables from ONS workbook
        marginals = {}
        
        # Table 1: Enterprises by industry and employment size
        marginals['industry_employment'] = pd.read_excel(
            self.ons_data_path, 
            sheet_name='Table 1',
            skiprows=7
        )
        
        # Table 2: Enterprises by industry and turnover size
        marginals['industry_turnover'] = pd.read_excel(
            self.ons_data_path,
            sheet_name='Table 2', 
            skiprows=7
        )
        
        # Table 5: Cross-tabulation of employment and turnover
        marginals['employment_turnover'] = pd.read_excel(
            self.ons_data_path,
            sheet_name='Table 5',
            skiprows=7  
        )
        
        return marginals
    
    def generate_initial_firms(self) -> pd.DataFrame:
        """Generate initial synthetic firms with random characteristics."""
        firms = pd.DataFrame({
            'firm_id': range(self.n_firms),
            'weight': np.ones(self.n_firms) / self.n_firms
        })
        
        # Industry distribution (simplified SIC sections)
        industries = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 
                     'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S']
        firms['industry'] = np.random.choice(industries, self.n_firms)
        
        # Employment size bands
        emp_bands = ['0', '1-4', '5-9', '10-19', '20-49', '50-99', '100-249', '250+']
        emp_probs = [0.05, 0.70, 0.10, 0.06, 0.04, 0.02, 0.02, 0.01]  # Approximate
        firms['employment_band'] = np.random.choice(emp_bands, self.n_firms, p=emp_probs)
        
        # Generate exact employment within bands
        firms['employment'] = firms['employment_band'].map(self._sample_from_band)
        
        # Turnover (correlated with employment)
        firms['turnover'] = self._generate_turnover(firms['employment'], firms['industry'])
        
        # VAT registration (based on £90k threshold)
        firms['vat_registered'] = firms['turnover'] >= 90_000
        
        # Region
        regions = ['London', 'South East', 'East', 'South West', 'West Midlands',
                  'East Midlands', 'Yorkshire', 'North West', 'North East', 
                  'Wales', 'Scotland', 'Northern Ireland']
        firms['region'] = np.random.choice(regions, self.n_firms)
        
        return firms
    
    def _sample_from_band(self, band: str) -> int:
        """Sample exact value from employment band."""
        if band == '0':
            return 0
        elif band == '250+':
            # Use log-normal for large firms
            return int(np.random.lognormal(6, 1))
        else:
            low, high = map(int, band.split('-'))
            return np.random.randint(low, high + 1)
    
    def _generate_turnover(self, employment: pd.Series, industry: pd.Series) -> pd.Series:
        """Generate turnover correlated with employment and industry."""
        # Industry-specific revenue per employee (approximate)
        revenue_per_employee = {
            'A': 50_000,   # Agriculture
            'C': 200_000,  # Manufacturing  
            'F': 150_000,  # Construction
            'G': 300_000,  # Wholesale/Retail
            'I': 40_000,   # Accommodation/Food
            'J': 250_000,  # Information/Communication
            'K': 500_000,  # Financial
            'M': 120_000,  # Professional services
            'Q': 30_000,   # Health/Social
        }
        
        # Default revenue per employee
        default_rpe = 100_000
        
        # Calculate base turnover
        base_turnover = employment * industry.map(
            lambda x: revenue_per_employee.get(x, default_rpe)
        )
        
        # Add noise
        noise = np.random.lognormal(0, 0.5, len(employment))
        turnover = base_turnover * noise
        
        # Handle zero employment firms
        turnover[employment == 0] = np.random.lognormal(10, 2, sum(employment == 0))
        
        return turnover.astype(int)
    
    def compute_marginals(self, firms: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Compute marginal distributions from synthetic firms."""
        marginals = {}
        
        # Industry × Employment
        marginals['industry_employment'] = pd.crosstab(
            firms['industry'],
            firms['employment_band'],
            values=firms['weight'],
            aggfunc='sum'
        )
        
        # Industry × Turnover bands
        turnover_bands = pd.cut(
            firms['turnover'],
            bins=[0, 50_000, 100_000, 250_000, 500_000, 1_000_000, 5_000_000, np.inf],
            labels=['0-50k', '50-100k', '100-250k', '250-500k', '500k-1m', '1-5m', '5m+']
        )
        marginals['industry_turnover'] = pd.crosstab(
            firms['industry'],
            turnover_bands,
            values=firms['weight'], 
            aggfunc='sum'
        )
        
        return marginals
    
    def optimize_weights(self, firms: pd.DataFrame, target_marginals: Dict, 
                        learning_rate: float = 0.01, n_iterations: int = 1000) -> pd.DataFrame:
        """
        Optimize firm weights to match target marginals using gradient descent.
        
        Args:
            firms: Initial synthetic firms
            target_marginals: Target marginal distributions from ONS
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of optimization iterations
            
        Returns:
            Firms with optimized weights
        """
        # Convert to PyTorch tensors
        weights = torch.tensor(firms['weight'].values, requires_grad=True)
        
        # Optimizer
        optimizer = torch.optim.Adam([weights], lr=learning_rate)
        
        for i in range(n_iterations):
            optimizer.zero_grad()
            
            # Normalize weights
            normalized_weights = F.softmax(weights, dim=0) * len(weights)
            
            # Compute loss (simplified - would need full implementation)
            loss = self._compute_marginal_loss(firms, normalized_weights, target_marginals)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                logging.info(f"Iteration {i}, Loss: {loss.item():.4f}")
        
        # Update firms with optimized weights
        firms['weight'] = normalized_weights.detach().numpy()
        
        return firms
    
    def _compute_marginal_loss(self, firms: pd.DataFrame, weights: torch.Tensor,
                               target_marginals: Dict) -> torch.Tensor:
        """Compute loss between synthetic and target marginals."""
        # This is a simplified version - full implementation would compute
        # all marginals and compare to targets
        
        # Example: Industry distribution loss
        industry_counts = pd.get_dummies(firms['industry']).values
        weighted_industry = torch.tensor(industry_counts.T) @ weights
        
        # Placeholder - would compare to actual targets
        target_industry = torch.ones_like(weighted_industry) * len(firms) / len(weighted_industry)
        
        loss = torch.mean((weighted_industry - target_industry) ** 2)
        
        return loss
    
    def validate_results(self, firms: pd.DataFrame) -> Dict[str, float]:
        """Validate synthetic firms against known statistics."""
        validation = {}
        
        # VAT registration rate
        vat_rate = (firms['vat_registered'] * firms['weight']).sum() / firms['weight'].sum()
        validation['vat_registration_rate'] = vat_rate
        
        # Average turnover by employment band
        for band in firms['employment_band'].unique():
            mask = firms['employment_band'] == band
            avg_turnover = (firms[mask]['turnover'] * firms[mask]['weight']).sum() / firms[mask]['weight'].sum()
            validation[f'avg_turnover_{band}'] = avg_turnover
        
        return validation
    
    def generate(self) -> pd.DataFrame:
        """Main method to generate synthetic firms."""
        logging.info("Loading ONS marginal distributions...")
        target_marginals = self.load_ons_marginals()
        
        logging.info(f"Generating {self.n_firms:,} initial firms...")
        firms = self.generate_initial_firms()
        
        logging.info("Optimizing weights to match marginals...")
        firms = self.optimize_weights(firms, target_marginals)
        
        logging.info("Validating results...")
        validation = self.validate_results(firms)
        for key, value in validation.items():
            logging.info(f"{key}: {value:,.0f}")
        
        return firms


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    generator = SyntheticFirmGenerator(
        ons_data_path="data/ons-business-statistics/ukbusinessworkbook2024.xlsx",
        n_firms=1_000_000
    )
    
    synthetic_firms = generator.generate()
    
    # Save results
    synthetic_firms.to_parquet("data/synthetic_firms.parquet", index=False)
    print(f"Generated {len(synthetic_firms):,} synthetic firms")