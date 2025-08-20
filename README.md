# PolicyEngine VAT Lab

A comprehensive microsimulation framework for analyzing UK Value Added Tax (VAT) policy reforms and their economic impacts on businesses and government revenue.

## Project Overview

We will model the revenue and business impacts of VAT reforms using a firm-level microsimulation approach that captures effects by sector and firm size. The project centres on developing an interactive web tool that enables policymakers to design custom VAT policies and immediately visualise their impacts across different business segments.

We will produce a comprehensive report analysing four specific reform scenarios:

1. **Higher VAT threshold**: Raising the registration threshold from £90,000 to £100,000, exempting additional small businesses from VAT registration

2. **Split-rate by sector**: Implementing a 10% rate for labour-intensive services (following [EU definitions](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=legissum:l31043) for hairdressing, repairs, cleaning) while maintaining 20% standard rate for other sectors. This approach follows the Netherlands model, where a 9% rate applies to labour-intensive services to support employment in sectors where human labour comprises the majority of value added.

3. **Graduated threshold (Moderate Taper 1)**: Creating a transition from £65,000 to £110,000 where effective VAT liability increases incrementally from 0% to 20%

4. **Graduated threshold (Moderate Taper 2)**: Alternative tapering from £90,000 to £135,000, with VAT liability increasing incrementally across this range

### Methodology

Microsimulation represents the optimal approach for analysing VAT reforms as it captures the heterogeneous impacts across thousands of firms with different characteristics. Unlike aggregate models that rely on average effects, microsimulation models individual firms' responses to policy changes based on their specific turnover, sector, and size. This granular approach reveals distributional impacts that would otherwise remain hidden - identifying precisely which types of businesses gain or lose under different reforms. For threshold policies particularly, microsimulation captures the non-linear incentives firms face, enabling realistic modelling of bunching behaviour and growth suppression that aggregate approaches miss.

We will construct synthetic firm microdata calibrated to [ONS UK Business statistics](https://www.ons.gov.uk/businessindustryandtrade/business/activitysizeandlocation), capturing the distribution of firms by industry classification, turnover bands, and employee counts. We will validate this synthetic dataset against [HMRC VAT statistics](https://www.gov.uk/government/statistics/value-added-tax-vat-annual-statistics) to ensure our model accurately represents sectoral receipts and registration patterns.

For behavioural modelling, we will conduct a comprehensive review of the empirical literature on VAT threshold responses, including [Liu, Lockwood & Tam (2024)](https://oxfordtax.sbs.ox.ac.uk/files/wp22-21-liu-lockwood-tampdf), [Bellon, Copestake & Zhang (2024)](https://matthieubellon.com/docs/Heterogeneous_VAT_Pass_Through.pdf), [Ross & Warwick (2021)](https://www.taxdev.org/sites/default/files/2021-12/Ross%20Warwick_tax%20processes%20or%20tax%20payments.pdf), and [Benedek et al. (2015)](https://www.imf.org/external/pubs/ft/wp/2015/wp15214.pdf). These studies document bunching at VAT thresholds, growth suppression effects, and differential pass-through rates across sectors. Based on the breadth of evidence, we will implement a justified turnover elasticity that captures both threshold bunching effects and smooth responses to graduated systems. Importantly, users can adjust this behavioural parameter in our interactive tool to explore how results vary under different assumptions. Our report will include detailed sensitivity analysis showing how revenue and distributional impacts change across the range of elasticities found in the literature.

Our impact analysis will calculate both static and behavioural effects for each scenario, including revenue changes, the number of firms experiencing tax increases or decreases by sector and size, shifts in VAT registration patterns across industries, and effective tax rates throughout the firm distribution. We will run all simulations for fiscal years 2025-26 through 2029-30 by aging the synthetic firm microdata in accordance with OBR projections, providing the Committee with medium-term projections of policy impacts.

## How ONS and HMRC Data Are Used

**ONS Data:** Provides realistic economic structure - uses all SIC sectors with exact firm counts and turnover distributions to generate base firms with authentic size patterns and employment relationships.

**HMRC Data:** Provides calibration targets - sector totals for VAT-registered firms, turnover band targets across all bands, and adds firms with negative/zero turnover missing from ONS surveys.

**Integration:** Multi-objective optimization process - ONS generates realistic firm structure, then mathematical optimization finds weights that simultaneously match HMRC targets while preserving authentic distributions and relationships.