# VAT Analysis Dashboard

A standalone PolicyEngine VAT policy analysis dashboard built with Next.js and React.

## Features

- **Interactive VAT Policy Simulation**: Configure VAT thresholds, rates, and analyze economic impacts
- **Multiple Analysis Tabs**:
  - Policy Reform Impact
  - Calibration & Official Statistics
  - Simulation Guide
  - Replication Code
- **Real-time Data Visualization**: Charts powered by Plotly.js
- **Responsive Design**: Works on desktop and mobile devices

## Quick Start

### Prerequisites

- Node.js 16.x or later
- npm or yarn

### Installation

1. Clone or download this repository
2. Install dependencies:

```bash
npm install
```

3. Run the development server:

```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

### Build for Production

```bash
npm run build
npm start
```

## Project Structure

```
vat-dashboard-standalone/
├── components/          # React components
│   ├── Layout.js       # Main layout with navigation
│   ├── Loading.js      # Loading component
│   ├── Tabs.js         # Tab navigation
│   ├── VATAnalysisSidebar.js  # Policy configuration sidebar
│   └── VATChart.js     # Chart components
├── pages/              # Next.js pages
│   ├── _app.js         # App configuration
│   ├── index.js        # Home page (redirects to VAT analysis)
│   └── vat-analysis.js # Main VAT analysis dashboard
├── styles/
│   └── globals.css     # Global styles
├── public/             # Static assets
├── package.json        # Dependencies and scripts
├── next.config.js      # Next.js configuration
└── README.md          # This file
```

## Usage

1. **Configure Policy Parameters**: Use the sidebar to set VAT thresholds, rates, and analysis parameters
2. **Switch Tabs**: Navigate between different analysis views
3. **Interactive Charts**: Hover over charts for detailed information
4. **Export Results**: Use the "Results CSV" button to download analysis data

## Data Disclaimer

⚠️ **This dashboard uses simulated data for demonstration purposes only.**

The data shown is not real government statistics and should not be used for actual policy decisions. For real PolicyEngine analysis, visit [policyengine.org](https://policyengine.org).

## Technology Stack

- **Frontend**: Next.js, React, Framer Motion
- **Charts**: Plotly.js, React-Plotly.js
- **Styling**: CSS with CSS variables
- **Build Tool**: Next.js built-in bundler

## Contributing

This is a demonstration project. For contributing to the actual PolicyEngine platform, visit the [PolicyEngine GitHub organization](https://github.com/PolicyEngine).

## License

MIT License - see the original PolicyEngine project for full license details.
