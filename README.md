# PolicyEngine VAT Lab

A comprehensive microsimulation framework for analyzing UK Value Added Tax (VAT) policy reforms and their economic impacts on businesses and government revenue.

## Project Overview

We will model the revenue and business impacts of VAT reforms using a firm-level microsimulation approach that captures effects by sector and firm size. The project centres on developing an interactive web tool that enables policymakers to design custom VAT policies and immediately visualise their impacts across different business segments.

We will deliver two core outputs. First, an interactive VAT policy calculator that provides real-time visualisation of how policy changes affect total VAT revenue, the distribution of tax burden changes across firms, and the number of VAT-registered businesses. Users can adjust key parameters including the registration threshold, sector-specific rates, and tapering designs to explore different policy configurations. We have produced an interactive mock-up of this concept, with fake data, at https://policyengine.github.io/vatlab/, displayed in Figure 1 below.

**Figure 1: PolicyEngine VATLab mockup, display a mix of options 2 and 3**

Second, we will produce a comprehensive report analysing four specific reform scenarios:

1. **Higher VAT threshold**: Raising the registration threshold from £90,000 to £100,000, exempting additional small businesses from VAT registration

2. **Split-rate by sector**: Implementing a 10% rate for labour-intensive services (following EU definitions for hairdressing, repairs, cleaning) while maintaining 20% standard rate for other sectors. This approach follows the Netherlands model, where a 9% rate applies to labour-intensive services to support employment in sectors where human labour comprises the majority of value added.

3. **Graduated threshold (Moderate Taper 1)**: Creating a transition from £65,000 to £110,000 where effective VAT liability increases incrementally from 0% to 20%

4. **Graduated threshold (Moderate Taper 2)**: Alternative tapering from £90,000 to £135,000, with VAT liability increasing incrementally across this range

This report will include a thorough literature review, detailed methodology, and sensitivity analysis examining how results vary with different behavioural assumptions.

Evidence from the Federation of Small Businesses and National Hair and Beauty Federation demonstrates that many businesses deliberately suppress turnover by reducing hours or turning away clients to remain under the VAT threshold. This behaviour particularly affects labour-intensive service sectors, limiting both economic productivity and job creation.

### Methodology

Microsimulation represents the optimal approach for analysing VAT reforms as it captures the heterogeneous impacts across thousands of firms with different characteristics. Unlike aggregate models that rely on average effects, microsimulation models individual firms' responses to policy changes based on their specific turnover, sector, and size. This granular approach reveals distributional impacts that would otherwise remain hidden - identifying precisely which types of businesses gain or lose under different reforms. For threshold policies particularly, microsimulation captures the non-linear incentives firms face, enabling realistic modelling of bunching behaviour and growth suppression that aggregate approaches miss.

We will construct synthetic firm microdata calibrated to ONS UK Business statistics, capturing the distribution of firms by industry classification, turnover bands, and employee counts. We will validate this synthetic dataset against HMRC VAT statistics to ensure our model accurately represents sectoral receipts and registration patterns.

For behavioural modelling, we will conduct a comprehensive review of the empirical literature on VAT threshold responses, including Liu, Lockwood & Tam (2024), Bellon, Copestake & Zhang (2024), Ross & Warwick (2021), and Benedek et al. (2015). These studies document bunching at VAT thresholds, growth suppression effects, and differential pass-through rates across sectors. Based on the breadth of evidence, we will implement a justified turnover elasticity that captures both threshold bunching effects and smooth responses to graduated systems. Importantly, users can adjust this behavioural parameter in our interactive tool to explore how results vary under different assumptions. Our report will include detailed sensitivity analysis showing how revenue and distributional impacts change across the range of elasticities found in the literature.

Our impact analysis will calculate both static and behavioural effects for each scenario, including revenue changes, the number of firms experiencing tax increases or decreases by sector and size, shifts in VAT registration patterns across industries, and effective tax rates throughout the firm distribution. We will run all simulations for fiscal years 2025-26 through 2029-30 by aging the synthetic firm microdata in accordance with OBR projections, providing the Committee with medium-term projections of policy impacts.