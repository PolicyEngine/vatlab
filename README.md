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

## 🚀 Quick Start

### Prerequisites
- Node.js 22+
- npm
- Python 3.8+ (for data processing)

### Installation & Running

```bash
# Clone the repository
git clone https://github.com/PolicyEngine/PolicyEngine_VATLab.git
cd PolicyEngine_VATLab

# Navigate to the dashboard
cd demo-dashboard/vatlab

# Install dependencies
npm install

# Start development server
npm run dev
```

Visit `http://localhost:3000` to access the VAT Policy Analysis Dashboard.

## 📁 Project Structure

```
PolicyEngine_VATLab/
├── README.md                    # Main project documentation
├── data/                        # Processed data files
│   ├── UK_business_data/
│   │   ├── table8.csv          # UK business enterprises by SIC & turnover
│   │   ├── firm_employment.csv # UK business local units by SIC & employment
│   │   └── ukbusinessworkbook2024.xlsx  # Original ONS data
│   └── HMRC_VAT_annual_statistics/
│       ├── vat_population_by_turnover_band.csv # VAT registration by turnover
│       └── Annual_UK_VAT_Statistics_2023-24.xls # Original HMRC data
└── demo-dashboard/              # Main application directory
    ├── README.md               # Detailed dashboard documentation
    ├── data/                   # Raw data and processing scripts
    │   ├── extract_sectors.py  # Data extraction utilities
    │   ├── synthetic_firm_generator.py  # Synthetic firm generation
    │   └── ons-business-statistics/  # ONS data source info
    ├── vat-dashboard/          # Alternative dashboard (legacy)
    └── vatlab/                 # Main Next.js application
        ├── components/         # React components
        ├── pages/             # Application pages
        ├── styles/            # CSS styles
        ├── __tests__/         # Test files
        └── public/            # Static assets
```

## 🎯 Features

### Policy Simulation
- **Registration Thresholds**: Model changes to VAT registration requirements
- **Graduated Systems**: Analyze smooth transition thresholds
- **Industry-Specific Rates**: Split rates for labor-intensive sectors
- **Behavioral Responses**: Account for firm adaptation to policy changes

### Data Analysis
- **Official Statistics Integration**: Built on ONS and HMRC data
- **Synthetic Microdata**: Representative firm-level data generation
- **Real-time Calculations**: Interactive policy impact analysis
- **Visual Analytics**: Charts and graphs for policy insights

### Dashboard Components
- **Policy Configuration**: Interactive parameter adjustment
- **Impact Visualization**: Revenue projections and distributional effects
- **Sectoral Analysis**: Industry-specific impacts
- **Firm-Level Results**: Winners and losers identification

## 📊 Data Sources

### Primary Data
- **ONS UK Business Statistics 2024**: Enterprise counts by industry, size, and location
- **HMRC VAT Statistics 2023-24**: Registration data and revenue figures by turnover bands
- **Living Costs and Food Survey**: Household spending patterns

### Processed Data Files
- `data/UK_business_data/table8.csv`: Clean UK business enterprise data by turnover bands
  - **SIC Code**: Standard Industrial Classification codes
  - **Description**: Industry descriptions
  - **Turnover Bands**: Enterprise counts by revenue size (£000s)
  - Columns: 0-49, 50-99, 100-249, 250-499, 500-999, 1000-4999, 5000+, Total

- `data/UK_business_data/firm_employment.csv`: UK business local units by employment size
  - **SIC Code**: Standard Industrial Classification codes
  - **Description**: Industry descriptions
  - **Employment Bands**: Local unit counts by employee size
  - Columns: 0-4, 5-9, 10-19, 20-49, 50-99, 100-249, 250+, Total

- `data/HMRC_VAT_annual_statistics/vat_population_by_turnover_band.csv`: VAT registration by turnover
  - **Financial_Year**: Tax years from 2004-05 to 2023-24
  - **Turnover Bands**: VAT registered businesses by revenue size
  - Historical time series of VAT population by threshold categories

## 🧪 Development

### Running Tests
```bash
cd demo-dashboard/vatlab
npm test
```

### Building for Production
```bash
npm run build
npm start
```

### Code Quality
```bash
npm run lint
```

### Development Workflow
1. Make changes to components or pages
2. Test locally with `npm run dev`
3. Run tests with `npm test`
4. Check code quality with `npm run lint`
5. Build for production with `npm run build`

## 🔬 Technical Architecture

### Frontend Stack
- **Next.js**: React framework with SSR/SSG
- **React**: Component-based UI
- **Plotly.js**: Interactive data visualization
- **Framer Motion**: Smooth animations

### Data Processing
- **Python**: Data extraction and transformation
- **Pandas**: Data manipulation
- **Synthetic Data Generation**: Gradient descent optimization for realistic firm distributions

### Testing
- **Jest**: Unit testing framework
- **Testing Library**: React component testing
- **ESLint**: Code quality and consistency

## 📈 Usage Guide

1. **Launch Application**: Start the development server or access deployed version
2. **Configure Baseline**: Set current UK VAT policy parameters
3. **Design Reform**: Define your policy reform scenario
4. **Run Analysis**: Execute microsimulation calculations
5. **Review Results**: Examine revenue impacts, firm distributions, and sectoral effects
6. **Export Data**: Save results for further analysis

## 🎛️ Key Policy Parameters

- **Registration Threshold**: Turnover level requiring VAT registration (current: £90,000)
- **Graduated Threshold**: Optional upper threshold for smooth transition
- **Standard Rate**: Main VAT rate (current: 20%)
- **Reduced Rates**: Lower rates for specific sectors
- **Labor-Intensive Industries**: Sectors eligible for preferential treatment
- **Behavioral Elasticity**: Firm response sensitivity to policy changes

## 🚀 Deployment

### Local Development
```bash
npm run dev  # Development server on localhost:3000
```

### Production Build
```bash
npm run build  # Create optimized build
npm start      # Start production server
```

### Static Export (GitHub Pages)
```bash
npm run export  # Generate static files
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Development Guidelines
- Follow existing code style and patterns
- Write tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Office for National Statistics (ONS)**: UK business statistics data
- **HM Revenue & Customs (HMRC)**: VAT registration and revenue data
- **PolicyEngine Team**: Framework development and maintenance

## 📞 Support

- **Issues**: Report bugs or request features via GitHub Issues
- **Documentation**: See `demo-dashboard/README.md` for detailed technical documentation
- **Data Sources**: Check `demo-dashboard/data/README.md` for data methodology

## 🔗 Related Projects

- [PolicyEngine UK](https://github.com/PolicyEngine/policyengine-uk): UK tax-benefit microsimulation
- [PolicyEngine Core](https://github.com/PolicyEngine/policyengine-core): Core microsimulation framework

---

**Note**: This tool uses synthetic data for demonstration purposes. Results should be interpreted as illustrative rather than precise policy predictions. The methodology is calibrated to official UK statistics but involves modeling assumptions and approximations.

