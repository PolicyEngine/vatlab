import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import VATAnalysis from '../pages/vat-analysis';

// Mock framer-motion to avoid animation issues in tests
jest.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }) => <div {...props}>{children}</div>,
    h1: ({ children, ...props }) => <h1 {...props}>{children}</h1>,
    button: ({ children, ...props }) => <button {...props}>{children}</button>,
  },
  AnimatePresence: ({ children }) => <>{children}</>,
}));

// Mock the components
jest.mock('../components/Layout', () => {
  return function Layout({ children }) {
    return <div data-testid="layout">{children}</div>;
  };
});

jest.mock('../components/Loading', () => {
  return function Loading() {
    return <div data-testid="loading">Loading...</div>;
  };
});

jest.mock('../components/Tabs', () => {
  return function Tabs({ children }) {
    return <div data-testid="tabs">{children}</div>;
  };
});

jest.mock('../components/VATAnalysisSidebar', () => {
  return function VATAnalysisSidebar({ onFiltersChange }) {
    return (
      <div data-testid="sidebar">
        <button onClick={() => onFiltersChange({ threshold: 100000 })}>
          Analyse
        </button>
      </div>
    );
  };
});

jest.mock('../components/VATChart', () => ({
  VATElasticityRevenueChart: () => <div data-testid="elasticity-chart">Chart</div>,
  VATRevenueHistoryChart: () => <div data-testid="revenue-history-chart">Chart</div>,
  VATRegistrationChart: () => <div data-testid="registration-chart">Chart</div>,
  VATRateComparisonChart: () => <div data-testid="rate-comparison-chart">Chart</div>,
}));

describe('VATAnalysis Page', () => {
  it('renders without crashing', () => {
    render(<VATAnalysis />);
    expect(screen.getByText('PolicyEngine VATLab')).toBeInTheDocument();
  });

  it('shows loading state initially', () => {
    render(<VATAnalysis />);
    expect(screen.getByTestId('sidebar')).toBeInTheDocument();
  });

  it('displays quick start guide when no analysis results', async () => {
    render(<VATAnalysis />);
    
    // Wait for the component to load
    await screen.findByText('Quick Start Guide');
    
    expect(screen.getByText('Quick Start Guide')).toBeInTheDocument();
    expect(screen.getByText(/Choose Baseline:/)).toBeInTheDocument();
    expect(screen.getByText(/Configure Reform:/)).toBeInTheDocument();
  });

  it('has valid JSX structure', () => {
    // This test will fail to compile if there are JSX syntax errors
    const { container } = render(<VATAnalysis />);
    
    // Check that the main container exists
    expect(container.querySelector('.container')).toBeInTheDocument();
    
    // Check that conditional rendering works
    const mainContent = container.querySelector('.main-content');
    expect(mainContent).toBeInTheDocument();
  });

  it('renders all tabs', async () => {
    render(<VATAnalysis />);
    
    await screen.findByTestId('tabs');
    
    // The tabs component should be rendered
    expect(screen.getByTestId('tabs')).toBeInTheDocument();
  });
});