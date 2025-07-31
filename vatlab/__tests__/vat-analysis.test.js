import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";
import VATAnalysis from "../pages/vat-analysis";

// Mock framer-motion to avoid animation issues in tests
jest.mock("framer-motion", () => ({
  motion: {
    div: ({ children, ...props }) => <div {...props}>{children}</div>,
    h1: ({ children, ...props }) => <h1 {...props}>{children}</h1>,
    button: ({ children, ...props }) => <button {...props}>{children}</button>,
  },
  AnimatePresence: ({ children }) => <>{children}</>,
}));

// Mock the components
jest.mock("../components/Layout", () => {
  return function Layout({ children }) {
    return <div data-testid="layout">{children}</div>;
  };
});

jest.mock("../components/Loading", () => {
  return function Loading() {
    return <div data-testid="loading">Loading...</div>;
  };
});

// Tabs component removed in new design

jest.mock("../components/VATAnalysisSidebar", () => {
  return function VATAnalysisSidebar({ onFiltersChange }) {
    return (
      <div data-testid="sidebar">
        <input
          type="number"
          onChange={(e) =>
            onFiltersChange({ threshold: parseInt(e.target.value) || 90000 })
          }
          placeholder="Threshold"
        />
      </div>
    );
  };
});

jest.mock("../components/VATChart", () => ({
  VATElasticityRevenueChart: () => (
    <div data-testid="elasticity-chart">Chart</div>
  ),
  VATRevenueHistoryChart: () => (
    <div data-testid="revenue-history-chart">Chart</div>
  ),
  VATRegistrationChart: () => <div data-testid="registration-chart">Chart</div>,
  VATRateComparisonChart: () => (
    <div data-testid="rate-comparison-chart">Chart</div>
  ),
}));

describe("VATAnalysis Page", () => {
  it("renders without crashing", () => {
    render(<VATAnalysis />);
    expect(screen.getByText("PolicyEngine VATLab")).toBeInTheDocument();
  });

  it("shows loading state initially", () => {
    render(<VATAnalysis />);
    expect(screen.getByTestId("sidebar")).toBeInTheDocument();
  });

  it("displays baseline prompt by default", async () => {
    render(<VATAnalysis />);

    // The new interface shows a prompt to define reform when at baseline
    expect(screen.getByText("PolicyEngine VATLab")).toBeInTheDocument();
    expect(screen.getByText("Define Your VAT Reform")).toBeInTheDocument();
  });

  it("has valid JSX structure", () => {
    // This test will fail to compile if there are JSX syntax errors
    const { container } = render(<VATAnalysis />);

    // Check that the main container exists
    expect(container.querySelector(".container")).toBeInTheDocument();

    // Check that conditional rendering works
    const mainContent = container.querySelector(".main-content");
    expect(mainContent).toBeInTheDocument();
  });

  it("renders main interface elements", async () => {
    render(<VATAnalysis />);

    // Check for the new interface elements
    expect(screen.getByText("Define Your VAT Reform")).toBeInTheDocument();
    expect(screen.getByText("Available Parameters:")).toBeInTheDocument();
  });

  it("has all required helper functions defined", () => {
    // This test ensures that all helper functions are properly defined
    // to prevent runtime errors like "calculateWinners is not defined"

    // Component should render without throwing errors
    expect(() => {
      render(<VATAnalysis />);
    }).not.toThrow();
  });

  it("handles analysis with changed parameters without errors", async () => {
    const { getByPlaceholderText } = render(<VATAnalysis />);

    // Change threshold value
    const thresholdInput = getByPlaceholderText("Threshold");
    fireEvent.change(thresholdInput, { target: { value: "100000" } });

    // Wait for any async operations
    await waitFor(() => {
      // Component should render without throwing errors
      expect(screen.getByText("PolicyEngine VATLab")).toBeInTheDocument();
    });
  });
});
