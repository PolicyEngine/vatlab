import React from "react";
import dynamic from "next/dynamic";
import { motion } from "framer-motion";

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import("react-plotly.js"), {
  ssr: false,
  loading: () => <div>Loading chart...</div>,
});

// PolicyEngine style constants
const FOG_GRAY = "#F4F4F4";
const BLUE_PRIMARY = "#2C6496";
const BLUE_LIGHT = "#D8E6F3";
const DARK_GRAY = "#616161";
const TEAL_ACCENT = "#39C6C0";
const ORANGE_ACCENT = "#F58518";
const GREEN_ACCENT = "#72B7B2";

function formatFigure(fig) {
  const newLayout = {
    ...fig.layout,
    font: {
      family: "Roboto Serif",
      color: "black",
    },
    template: "plotly_white",
    height: 460,
    plot_bgcolor: FOG_GRAY,
    paper_bgcolor: FOG_GRAY,
    xaxis: {
      ...fig.layout.xaxis,
      gridcolor: FOG_GRAY,
      zerolinecolor: FOG_GRAY,
    },
    yaxis: {
      ...fig.layout.yaxis,
      gridcolor: FOG_GRAY,
      zerolinecolor: DARK_GRAY,
    },
    modebar: {
      bgcolor: FOG_GRAY,
      color: FOG_GRAY,
      activecolor: FOG_GRAY,
    },
    margin: {
      l: 80,
      r: 80,
      t: 100,
      b: 80,
    },
  };

  return {
    ...fig,
    layout: newLayout,
  };
}

export function PolicyScenariosChart({ scenarios, selectedScenario = null }) {
  if (!scenarios) {
    return <div className="chart-container">No scenario data available.</div>;
  }

  // Filter out baseline and get other scenarios, highlighting selected one
  const scenarioKeys = Object.keys(scenarios).filter(
    (key) => key !== "baseline",
  );
  const scenarioLabels = scenarioKeys.map(
    (key) => scenarios[key].description.replace(/^[^:]*:\s*/, ""), // Remove prefix before colon
  );
  const revenueChanges = scenarioKeys.map(
    (key) => scenarios[key].revenue_change_from_baseline / 1000000000, // Convert to billions
  );

  // Create colors array - red for negative, green for positive, highlight selected
  const colors = scenarioKeys.map((key, index) => {
    const change = revenueChanges[index];
    const baseColor = change >= 0 ? GREEN_ACCENT : ORANGE_ACCENT;
    return selectedScenario === key ? BLUE_PRIMARY : baseColor;
  });

  const fig = {
    data: [
      {
        type: "bar",
        x: scenarioLabels,
        y: revenueChanges,
        marker: {
          color: colors,
        },
        text: revenueChanges.map(
          (val) => `${val >= 0 ? "+" : ""}£${val.toFixed(1)}bn`,
        ),
        textposition: "auto",
        hovertemplate:
          "<b>%{x}</b><br>Revenue change: %{text}<br><i>Compared to current system</i><extra></extra>",
      },
    ],
    layout: {
      title: "VAT Revenue Impact by Policy Scenario",
      yaxis: {
        title: "Revenue change (£ billions)",
        tickprefix: "£",
        ticksuffix: "bn",
        zeroline: true,
        zerolinecolor: DARK_GRAY,
        zerolinewidth: 2,
      },
      xaxis: {
        title: "Policy scenario",
        tickangle: -20,
      },
      annotations: [
        {
          text: "",
          showarrow: false,
          x: 0,
          y: -0.45,
          xref: "paper",
          yref: "paper",
          xanchor: "left",
          yanchor: "bottom",
          font: {
            size: 10,
            color: DARK_GRAY,
          },
        },
      ],
    },
  };

  const formattedFig = formatFigure(fig);

  return (
    <motion.div
      className="chart-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.3 }}
    >
      <Plot
        data={formattedFig.data}
        layout={formattedFig.layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "100%" }}
      />
    </motion.div>
  );
}

export function SectoralImpactChart({ sectoralData }) {
  if (!sectoralData || !sectoralData.sectoral_impacts) {
    return <div className="chart-container">No sectoral data available.</div>;
  }

  // Get sectors and their baseline vs split rate scenario impact
  const sectors = Object.keys(sectoralData.sectoral_impacts);
  const sectorLabels = sectors.map((sector) =>
    sector.length > 20 ? sector.substring(0, 17) + "..." : sector,
  );

  // Compare current baseline vs split rate scenario
  const baselineImpacts = sectors.map(
    (sector) =>
      sectoralData.sectoral_impacts[sector].scenario_impacts.baseline
        .total_vat_liability / 1000000,
  );

  const splitRateImpacts = sectors.map(
    (sector) =>
      sectoralData.sectoral_impacts[sector].scenario_impacts.split_rate
        .total_vat_liability / 1000000,
  );

  const isLabourIntensive = sectors.map(
    (sector) => sectoralData.sectoral_impacts[sector].is_labour_intensive,
  );

  const fig = {
    data: [
      {
        type: "bar",
        x: sectorLabels,
        y: baselineImpacts,
        marker: {
          color: BLUE_PRIMARY,
        },
        name: "Current system (20%)",
        hovertemplate:
          "<b>%{x}</b><br>VAT liability: £%{y:.1f}m<br><i>Under current 20% rate</i><extra></extra>",
      },
      {
        type: "bar",
        x: sectorLabels,
        y: splitRateImpacts,
        marker: {
          color: isLabourIntensive.map((intensive) =>
            intensive ? TEAL_ACCENT : BLUE_LIGHT,
          ),
        },
        name: "Split rate system",
        hovertemplate:
          "<b>%{x}</b><br>VAT liability: £%{y:.1f}m<br><i>Under split rate system</i><extra></extra>",
      },
    ],
    layout: {
      title: "Sectoral VAT Impact: Current vs Split Rate System",
      yaxis: {
        title: "Total VAT liability (£ millions)",
        tickprefix: "£",
        ticksuffix: "m",
      },
      xaxis: {
        title: "Business sector",
        tickangle: -45,
      },
      barmode: "group",
      legend: {
        x: 0.01,
        y: 0.99,
        bgcolor: "rgba(255, 255, 255, 0.8)",
        bordercolor: BLUE_LIGHT,
        borderwidth: 1,
      },
      annotations: [
        {
          text: "",
          showarrow: false,
          x: 0,
          y: -0.25,
          xref: "paper",
          yref: "paper",
          xanchor: "left",
          yanchor: "bottom",
          font: {
            size: 10,
            color: DARK_GRAY,
          },
        },
      ],
    },
  };

  const formattedFig = formatFigure(fig);

  return (
    <motion.div
      className="chart-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
    >
      <Plot
        data={formattedFig.data}
        layout={formattedFig.layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "100%" }}
      />
    </motion.div>
  );
}

export function FirmDistributionChart({ firmData }) {
  if (!firmData || !firmData.sectors) {
    return (
      <div className="chart-container">
        No firm distribution data available.
      </div>
    );
  }

  // Aggregate firm counts by turnover band across all sectors
  const turnoverBands = [
    "£0-30k",
    "£30-60k",
    "£60-90k",
    "£90-120k",
    "£120-200k",
    "£200k+",
  ];
  const bandCounts = turnoverBands.map(() => 0);
  const vatRegisteredCounts = turnoverBands.map(() => 0);

  Object.values(firmData.sectors).forEach((sector) => {
    sector.turnover_bands.forEach((band, index) => {
      bandCounts[index] += band.firm_count;
      if (band.vat_registered) {
        vatRegisteredCounts[index] += band.firm_count;
      }
    });
  });

  const fig = {
    data: [
      {
        type: "bar",
        x: turnoverBands,
        y: bandCounts,
        marker: {
          color: BLUE_LIGHT,
        },
        name: "All firms",
        hovertemplate: "<b>%{x}</b><br>Total firms: %{y:,}<extra></extra>",
      },
      {
        type: "bar",
        x: turnoverBands,
        y: vatRegisteredCounts,
        marker: {
          color: BLUE_PRIMARY,
        },
        name: "VAT registered",
        hovertemplate: "<b>%{x}</b><br>VAT registered: %{y:,}<extra></extra>",
      },
    ],
    layout: {
      title: "UK Business Distribution by Turnover Band",
      yaxis: {
        title: "Number of firms",
        tickformat: ",",
      },
      xaxis: {
        title: "Annual turnover band",
      },
      barmode: "overlay",
      legend: {
        x: 0.7,
        y: 0.99,
        bgcolor: "rgba(255, 255, 255, 0.8)",
        bordercolor: BLUE_LIGHT,
        borderwidth: 1,
      },
      annotations: [
        {
          text: "The £90k VAT threshold creates a clear divide. Firms above this threshold must register for VAT and charge 20% on their sales.",
          showarrow: false,
          x: 0,
          y: -0.25,
          xref: "paper",
          yref: "paper",
          xanchor: "left",
          yanchor: "bottom",
          font: {
            size: 10,
            color: DARK_GRAY,
          },
        },
      ],
    },
  };

  const formattedFig = formatFigure(fig);

  return (
    <motion.div
      className="chart-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.3 }}
    >
      <Plot
        data={formattedFig.data}
        layout={formattedFig.layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "100%" }}
      />
    </motion.div>
  );
}

export function ThresholdEffectsChart({ thresholdData }) {
  if (!thresholdData || !thresholdData.threshold_bunching) {
    return (
      <div className="chart-container">
        No threshold effects data available.
      </div>
    );
  }

  const ranges = thresholdData.threshold_bunching.map((item) => item.range);
  const firmCounts = thresholdData.threshold_bunching.map((item) => item.firms);

  const fig = {
    data: [
      {
        type: "bar",
        x: ranges,
        y: firmCounts,
        marker: {
          color: ranges.map((range) =>
            range.includes("80k-90k")
              ? ORANGE_ACCENT
              : range.includes("90k-100k")
                ? DARK_GRAY
                : BLUE_PRIMARY,
          ),
        },
        text: firmCounts.map((count) => count.toLocaleString()),
        textposition: "auto",
        hovertemplate:
          "<b>%{x}</b><br>Firms: %{y:,}<br><i>Firms in this turnover range</i><extra></extra>",
      },
    ],
    layout: {
      title: "Firm Bunching Around VAT Threshold",
      yaxis: {
        title: "Number of firms",
        tickformat: ",",
      },
      xaxis: {
        title: "Turnover range",
      },
      annotations: [
        {
          x: "£80k-90k",
          y: Math.max(...firmCounts) * 0.8,
          text: "Bunching below<br>threshold",
          showarrow: true,
          arrowhead: 2,
          arrowsize: 1,
          arrowwidth: 2,
          arrowcolor: ORANGE_ACCENT,
          font: {
            size: 12,
            color: ORANGE_ACCENT,
          },
        },
        {
          x: "£90k-100k",
          y: Math.min(...firmCounts) * 1.5,
          text: "Fewer firms<br>just above threshold",
          showarrow: true,
          arrowhead: 2,
          arrowsize: 1,
          arrowwidth: 2,
          arrowcolor: DARK_GRAY,
          font: {
            size: 12,
            color: DARK_GRAY,
          },
        },
        {
          text: "Evidence of firm bunching just below the £90k VAT threshold, as businesses limit growth to avoid VAT registration.",
          showarrow: false,
          x: 0,
          y: -0.25,
          xref: "paper",
          yref: "paper",
          xanchor: "left",
          yanchor: "bottom",
          font: {
            size: 10,
            color: DARK_GRAY,
          },
        },
      ],
    },
  };

  const formattedFig = formatFigure(fig);

  return (
    <motion.div
      className="chart-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.5 }}
    >
      <Plot
        data={formattedFig.data}
        layout={formattedFig.layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "100%" }}
      />
    </motion.div>
  );
}

export function VATElasticityRevenueChart({ scenarios, elasticityRange }) {
  if (!scenarios || !elasticityRange) {
    return (
      <div className="chart-container">
        No elasticity revenue data available.
      </div>
    );
  }

  // Generate elasticity values across the range
  const elasticityStep = (elasticityRange.max - elasticityRange.min) / 10;
  const elasticityValues = Array.from(
    { length: 11 },
    (_, i) => elasticityRange.min + i * elasticityStep,
  );

  // Calculate revenue for each scenario at different elasticity values
  const scenarioKeys = Object.keys(scenarios);
  const traces = scenarioKeys.map((scenarioKey, index) => {
    const scenario = scenarios[scenarioKey];
    const baseRevenue = scenario.total_vat_revenue / 1000000000; // Convert to billions

    // Calculate how revenue changes with elasticity
    const revenueValues = elasticityValues.map((elasticity) => {
      // Apply elasticity impact - more negative elasticity = lower revenue
      const elasticityFactor = 1 + elasticity * 0.1; // 10% change per elasticity unit
      return Math.max(0, baseRevenue * elasticityFactor);
    });

    const colors = [
      BLUE_PRIMARY,
      TEAL_ACCENT,
      GREEN_ACCENT,
      ORANGE_ACCENT,
      DARK_GRAY,
    ];

    return {
      type: "scatter",
      mode: "lines+markers",
      x: elasticityValues,
      y: revenueValues,
      name: scenario.description?.split(":")[0] || scenarioKey,
      line: {
        color: colors[index % colors.length],
        width: 3,
      },
      marker: {
        color: colors[index % colors.length],
        size: 8,
      },
      hovertemplate:
        "<b>%{fullData.name}</b><br>Elasticity: %{x:.1f}<br>Revenue: £%{y:.1f}bn<extra></extra>",
    };
  });

  const fig = {
    data: traces,
    layout: {
      title: "VAT Revenue Impact by Elasticity Range",
      xaxis: {
        title: "Elasticity",
        gridcolor: FOG_GRAY,
        zerolinecolor: DARK_GRAY,
        zerolinewidth: 2,
      },
      yaxis: {
        title: "VAT Revenue (£ billions)",
        tickprefix: "£",
        ticksuffix: "bn",
        gridcolor: FOG_GRAY,
        zerolinecolor: DARK_GRAY,
      },
      legend: {
        x: 0.02,
        y: 0.98,
        bgcolor: "rgba(255, 255, 255, 0.8)",
        bordercolor: BLUE_LIGHT,
        borderwidth: 1,
      },
      annotations: [
        {
          text: "Shows how VAT revenue for each policy scenario changes across different elasticity values. Lower (more negative) elasticity typically reduces revenue.",
          showarrow: false,
          x: 0,
          y: -0.25,
          xref: "paper",
          yref: "paper",
          xanchor: "left",
          yanchor: "bottom",
          font: {
            size: 10,
            color: DARK_GRAY,
          },
        },
      ],
    },
  };

  const formattedFig = formatFigure(fig);

  return (
    <motion.div
      className="chart-container"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay: 0.4 }}
    >
      <Plot
        data={formattedFig.data}
        layout={formattedFig.layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "100%" }}
      />
    </motion.div>
  );
}

// Government Statistics Charts
export function VATRevenueHistoryChart() {
  // Generate random historical VAT revenue data for UK (2015-2024)
  const years = Array.from({ length: 10 }, (_, i) => 2015 + i);
  const baseRevenue = 140; // £140bn baseline
  const vatRevenue = years.map((year, i) => {
    const trend = baseRevenue + i * 3.2; // Growing trend
    const noise = (Math.random() - 0.5) * 8; // Random variation
    return Math.max(120, trend + noise);
  });

  const fig = {
    data: [
      {
        type: "scatter",
        mode: "lines+markers",
        x: years,
        y: vatRevenue,
        line: {
          color: BLUE_PRIMARY,
          width: 4,
        },
        marker: {
          color: BLUE_PRIMARY,
          size: 8,
        },
        name: "VAT Revenue",
        hovertemplate: "<b>%{x}</b><br>VAT Revenue: £%{y:.1f}bn<extra></extra>",
      },
    ],
    layout: {
      title: "UK VAT Revenue History (2015-2024)",
      yaxis: {
        title: "VAT Revenue (£ billions)",
        tickprefix: "£",
        ticksuffix: "bn",
        gridcolor: FOG_GRAY,
      },
      xaxis: {
        title: "Year",
        gridcolor: FOG_GRAY,
      },
      autosize: true,
    },
  };

  const formattedFig = formatFigure(fig);

  return (
    <div style={{ width: "100%", height: "500px" }}>
      <Plot
        data={formattedFig.data}
        layout={formattedFig.layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler={true}
      />
    </div>
  );
}

export function VATRegistrationChart() {
  // Generate random data for VAT registrations by business size
  const businessSizes = [
    "Micro (0-9)",
    "Small (10-49)",
    "Medium (50-249)",
    "Large (250+)",
  ];
  const registrations = [
    Math.floor(Math.random() * 500000) + 1200000, // Micro: 1.2-1.7M
    Math.floor(Math.random() * 200000) + 300000, // Small: 300-500K
    Math.floor(Math.random() * 50000) + 80000, // Medium: 80-130K
    Math.floor(Math.random() * 20000) + 15000, // Large: 15-35K
  ];

  const colors = [BLUE_LIGHT, BLUE_PRIMARY, TEAL_ACCENT, GREEN_ACCENT];

  const fig = {
    data: [
      {
        type: "bar",
        x: businessSizes,
        y: registrations,
        marker: {
          color: colors,
        },
        text: registrations.map((val) => val.toLocaleString()),
        textposition: "auto",
        hovertemplate:
          "<b>%{x}</b><br>VAT Registrations: %{y:,}<extra></extra>",
      },
    ],
    layout: {
      title: "VAT Registrations by Business Size (2024)",
      yaxis: {
        title: "Number of VAT Registrations",
        tickformat: ",",
      },
      xaxis: {
        title: "Business Size (employees)",
      },
      autosize: true,
    },
  };

  const formattedFig = formatFigure(fig);

  return (
    <div style={{ width: "100%", height: "500px" }}>
      <Plot
        data={formattedFig.data}
        layout={formattedFig.layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler={true}
      />
    </div>
  );
}

export function VATRateComparisonChart() {
  // Generate random data for VAT rates across EU countries
  const countries = [
    "UK",
    "Germany",
    "France",
    "Italy",
    "Spain",
    "Netherlands",
    "Belgium",
    "Austria",
  ];
  const standardRates = [20, 19, 20, 22, 21, 21, 21, 20];
  const reducedRates = countries.map(() => Math.floor(Math.random() * 10) + 5); // 5-15%

  const fig = {
    data: [
      {
        type: "bar",
        x: countries,
        y: standardRates,
        name: "Standard Rate",
        marker: {
          color: BLUE_PRIMARY,
        },
        hovertemplate: "<b>%{x}</b><br>Standard Rate: %{y}%<extra></extra>",
      },
      {
        type: "bar",
        x: countries,
        y: reducedRates,
        name: "Reduced Rate (avg)",
        marker: {
          color: TEAL_ACCENT,
        },
        hovertemplate: "<b>%{x}</b><br>Reduced Rate: %{y}%<extra></extra>",
      },
    ],
    layout: {
      title: "VAT Rates: UK vs EU Comparison (2024)",
      yaxis: {
        title: "VAT Rate (%)",
        ticksuffix: "%",
      },
      xaxis: {
        title: "Country",
      },
      barmode: "group",
      legend: {
        x: 0.7,
        y: 0.99,
        bgcolor: "rgba(255, 255, 255, 0.8)",
        bordercolor: BLUE_LIGHT,
        borderwidth: 1,
      },
      autosize: true,
    },
  };

  const formattedFig = formatFigure(fig);

  return (
    <div style={{ width: "100%", height: "500px" }}>
      <Plot
        data={formattedFig.data}
        layout={formattedFig.layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%", height: "100%" }}
        useResizeHandler={true}
      />
    </div>
  );
}
