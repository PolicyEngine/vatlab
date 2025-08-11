# VATLab - PolicyEngine VAT Microsimulation Dashboard

A web-based microsimulation tool for analyzing UK Value Added Tax (VAT) policy reforms and their economic impacts.

## Overview

VATLab allows users to simulate various VAT policy scenarios including:
- Changes to registration thresholds
- Graduated threshold systems
- Split rates for labor-intensive industries
- Behavioral responses to policy changes

## Features

- **Interactive Policy Configuration**: Adjust VAT thresholds, rates, and industry classifications
- **Real-time Impact Analysis**: See revenue projections and firm-level impacts
- **Data Visualization**: Charts showing revenue over time, winner/loser distributions, and sectoral impacts
- **Calibration to Official Statistics**: Model validated against ONS and HMRC data
- **Synthetic Firm Generation**: Creates representative microdata matching UK business distributions

## Quick Start

### Prerequisites
- Node.js 22+ 
- npm

### Installation

```bash
# Clone the repository
git clone https://github.com/PolicyEngine/vatlab.git
cd vatlab

# Install dependencies
cd vatlab
npm install

# Start development server
npm run dev
```

Visit http://localhost:3000/vat-analysis to use the dashboard.

## Project Structure

```
vatlab/
├── vatlab/                # Next.js web application
│   ├── components/         # React components
│   ├── pages/             # Application pages
│   └── styles/            # CSS styles
├── data/                  # Data sources and generators
│   ├── ons-business-statistics/  # ONS business data
│   └── synthetic_firm_generator.py  # Synthetic data generation
└── README.md
```

## Data Sources

The model uses official UK government statistics:
- **ONS Business Statistics**: Enterprise counts by industry, size, and location
- **HMRC VAT Statistics**: Registration data and revenue figures
- **Living Costs and Food Survey**: Household spending patterns

See `data/README.md` for detailed information about data sources and synthetic firm generation methodology.

## Usage Guide

1. **Choose Baseline**: Configure the baseline VAT policy (current UK policy or custom)
2. **Set Reform Parameters**: Define your reform scenario in the Reform tab
3. **Run Analysis**: Click "Analyse VAT Policy" to generate results
4. **Review Impacts**: Explore revenue changes, firm distributions, and sectoral effects

## Key Policy Parameters

- **Registration Threshold**: Annual turnover above which firms must register for VAT
- **Graduated Threshold**: Optional upper threshold for smooth transition
- **Labor-Intensive Industries**: Sectors eligible for reduced rates
- **VAT Rates**: Separate rates for labor-intensive and standard businesses
- **Elasticity**: Behavioral response parameter

## Development

### Running Tests
```bash
npm test
```

### Building for Production
```bash
npm run build
npm start
```

### Code Style
- React: Functional components with hooks
- Formatting: Prettier + ESLint
- Run `npm run lint` before committing

## Deployment

The application can be deployed to any Node.js hosting platform. For GitHub Pages deployment, see `.github/workflows/deploy.yml`.

## License

Open source under the MIT License.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## Contact

For questions or support, please open an issue on GitHub.

## Acknowledgments

This tool uses synthetic data for demonstration purposes. The methodology is based on official UK statistics from ONS and HMRC.