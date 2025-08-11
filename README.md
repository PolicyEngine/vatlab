# PolicyEngine VAT Lab

A comprehensive microsimulation framework for analyzing UK Value Added Tax (VAT) policy reforms and their economic impacts on businesses and government revenue.

## 🚀 Quick Start

### Prerequisites
- Node.js 22+
- npm
- Python 3.8+ (for data processing)

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
- **HMRC VAT Statistics**: Registration data and revenue figures
- **Living Costs and Food Survey**: Household spending patterns

### Processed Data Files
- `data/UK_business_data/table8.csv`: Clean UK business enterprise data
  - **SIC Code**: Standard Industrial Classification codes
  - **Description**: Industry descriptions
  - **Turnover Bands**: Enterprise counts by revenue size (£000s)
  - Columns: 0-49, 50-99, 100-249, 250-499, 500-999, 1000-4999, 5000+, Total

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