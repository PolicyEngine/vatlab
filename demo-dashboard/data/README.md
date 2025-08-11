# VAT Microsimulation Data Sources

This directory contains data sources used for the VAT microsimulation model.

## ONS Business Statistics (`ons-business-statistics/`)

### Source
UK business: activity, size and location  
Office for National Statistics (ONS)  
URL: https://www.ons.gov.uk/businessindustryandtrade/business/activitysizeandlocation/datasets/ukbusinessactivitysizeandlocation

### Description
Numbers of enterprises and local units produced from a snapshot of the Inter-Departmental Business Register (IDBR) taken on 8 March 2024.

### Key Variables
- **Industry**: Standard Industrial Classification (UK SIC 2007)
- **Employment size bands**: 0, 1-4, 5-9, 10-19, 20-49, 50-99, 100-249, 250+
- **Turnover size bands**: Various bands from £0 to £50m+
- **Legal status**: Company, Sole proprietor, Partnership, Public corporation/nationalised body, Non-profit body
- **Geography**: UK regions and countries

### Data Files
- `ukbusinessworkbook2024.xlsx` - 2024 edition containing enterprises by industry, size, and location
- Download from: https://www.ons.gov.uk/file?uri=/businessindustryandtrade/business/activitysizeandlocation/datasets/ukbusinessactivitysizeandlocation/2024/ukbusinessworkbook2024.xlsx

### License
Open Government Licence v3.0

### Contact
Business Registers Strategy and Outputs team: idbrdas@ons.gov.uk

## Synthetic Firm Generation Plan

### Overview
Generate synthetic firm-level microdata that matches ONS marginal distributions using gradient descent optimization.

### Methodology

1. **Target Marginals**
   - Industry × Employment size
   - Industry × Turnover bands  
   - Employment × Turnover cross-tabulation
   - Regional distribution by industry

2. **Initial Sample Generation**
   - Generate N synthetic firms with random characteristics
   - Assign initial weights w_i = 1/N

3. **Gradient Descent Optimization**
   ```python
   # Objective: Minimize deviation from target marginals
   loss = sum((observed_marginal - weighted_marginal)^2)
   
   # Update weights iteratively
   w_i = w_i - learning_rate * gradient(loss, w_i)
   
   # Constraints: w_i >= 0, sum(w_i) = 1
   ```

4. **Firm Attributes**
   - Industry (SIC code)
   - Employment count
   - Annual turnover
   - VAT registration status (imputed based on £90k threshold)
   - Region
   - Legal form

5. **Validation**
   - Compare weighted synthetic marginals to ONS targets
   - Ensure realistic joint distributions
   - Validate against known VAT registration rates by sector

### Implementation Notes
- Use PyTorch/JAX for differentiable optimization
- Consider stratified sampling by major industry groups
- Apply post-processing to ensure coherent firm characteristics
- Store as Parquet for efficient access

### Usage
The synthetic firms will be used to:
- Simulate VAT policy impacts at firm level
- Calculate revenue changes by industry/size
- Model behavioral responses to threshold changes
- Generate distributional statistics