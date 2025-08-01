import { useState, useEffect } from "react";
import Layout from "../components/Layout";
import VATAnalysisSidebar from "../components/VATAnalysisSidebar";
import Loading from "../components/Loading";
import { motion } from "framer-motion";

export default function VATAnalysis() {
  const [filters, setFilters] = useState({
    threshold: 90000,
    graduatedEndThreshold: "",
    fullRateLaborIntensive: 20,
    fullRateNonLaborIntensive: 20,
    year: 2026,
    elasticity: -0.01,
  });

  const [analysisResults, setAnalysisResults] = useState(null);
  const [breakdownType, setBreakdownType] = useState("sector");
  const [weightingType, setWeightingType] = useState("firms");
  const [showTaxChange, setShowTaxChange] = useState(false);

  // Helper functions
  const isBaselinePolicy = (params) => {
    return (
      params.threshold === 90000 &&
      params.fullRateLaborIntensive === 20 &&
      params.fullRateNonLaborIntensive === 20 &&
      !params.graduatedEndThreshold
    );
  };

  const calculateReformRevenue = (params) => {
    const baseRevenue = 164200000000;
    const thresholdChange = (params.threshold - 90000) / 90000;
    const avgRateChange =
      ((params.fullRateLaborIntensive + params.fullRateNonLaborIntensive) / 2 -
        20) /
      20;

    const thresholdImpact = baseRevenue * thresholdChange * -0.02;
    const rateImpact = baseRevenue * avgRateChange * 0.9;

    return baseRevenue + thresholdImpact + rateImpact;
  };

  const calculateBusinessImpact = (params) => {
    const thresholdIncrease = params.threshold > 90000;
    const avgRate =
      (params.fullRateLaborIntensive + params.fullRateNonLaborIntensive) / 2;
    const rateDecrease = avgRate < 20;

    let payingMore, payingLess, unaffected;
    if (thresholdIncrease && rateDecrease) {
      payingLess = 62.3;
      payingMore = 35.7;
      unaffected = 2.0;
    } else if (thresholdIncrease || rateDecrease) {
      payingLess = 54.2;
      payingMore = 43.8;
      unaffected = 2.0;
    } else if (params.threshold < 90000 || avgRate > 20) {
      payingLess = 28.5;
      payingMore = 69.5;
      unaffected = 2.0;
    } else {
      payingLess = 0.0;
      payingMore = 0.0;
      unaffected = 100.0;
    }

    return { payingMore, payingLess, unaffected };
  };

  const calculateSectorTaxChanges = (params) => {
    const thresholdEffect = (params.threshold - 90000) / 90000;
    const avgRate =
      (params.fullRateLaborIntensive + params.fullRateNonLaborIntensive) / 2;
    const rateEffect = (avgRate - 20) / 20;

    // Base tax amounts by sector (in millions)
    const baseTaxBySector = {
      "Wholesale & Retail Trade": 45000,
      "Accommodation & Food": 12000,
      "Professional & Scientific": 28000,
      Construction: 18000,
      Manufacturing: 35000,
      "Hair & Beauty": 3000,
    };

    // Tax change percentages based on policy
    const taxChangePercent = {
      "Wholesale & Retail Trade": thresholdEffect * -0.08 + rateEffect * 0.85,
      "Accommodation & Food": thresholdEffect * -0.12 + rateEffect * 0.75,
      "Professional & Scientific": thresholdEffect * -0.05 + rateEffect * 0.9,
      Construction: thresholdEffect * 0.02 + rateEffect * 0.95,
      Manufacturing: thresholdEffect * 0.03 + rateEffect * 0.98,
      "Hair & Beauty": thresholdEffect * -0.18 + rateEffect * 0.7,
    };

    const result = {};
    for (const [sector, baseTax] of Object.entries(baseTaxBySector)) {
      const changePercent = taxChangePercent[sector] * 100;
      const changeAmount = baseTax * taxChangePercent[sector];
      result[sector] = {
        changePercent: changePercent,
        changeAmount: changeAmount,
      };
    }

    return result;
  };

  const calculateSectorBreakdown = (params) => {
    const thresholdEffect = (params.threshold - 90000) / 90000;
    const avgRate =
      (params.fullRateLaborIntensive + params.fullRateNonLaborIntensive) / 2;
    const rateEffect = (avgRate - 20) / 20;

    // Base percentages by sector (at baseline)
    const basePayingLess = {
      "Wholesale & Retail Trade": 0,
      "Accommodation & Food": 0,
      "Professional & Scientific": 0,
      Construction: 0,
      Manufacturing: 0,
      "Hair & Beauty": 0,
    };

    // Adjust based on policy changes
    const adjustments = {
      "Wholesale & Retail Trade": thresholdEffect * 30 - rateEffect * 10, // Benefits from higher threshold
      "Accommodation & Food": thresholdEffect * 25 - rateEffect * 15, // Labor intensive, mixed effect
      "Professional & Scientific": thresholdEffect * 10 - rateEffect * 20, // Less threshold sensitive
      Construction: thresholdEffect * -10 - rateEffect * 25, // Larger firms, hurt by changes
      Manufacturing: thresholdEffect * -15 - rateEffect * 30, // Capital intensive, hurt most
      "Hair & Beauty": thresholdEffect * 35 - rateEffect * 8, // Small businesses, benefit most from threshold
    };

    const result = {};
    for (const [sector, base] of Object.entries(basePayingLess)) {
      const payingLess = Math.max(0, Math.min(90, base + adjustments[sector]));
      const payingMore = Math.max(0, Math.min(90, 100 - payingLess - 2));
      result[sector] = {
        payingLess: payingLess,
        payingMore: payingMore,
        unaffected: 100 - payingLess - payingMore,
      };
    }

    return result;
  };

  const calculateSizeTaxChanges = (params) => {
    const thresholdEffect = (params.threshold - 90000) / 90000;
    const avgRate =
      (params.fullRateLaborIntensive + params.fullRateNonLaborIntensive) / 2;
    const rateEffect = (avgRate - 20) / 20;

    // Base tax amounts by size (in millions)
    const baseTaxBySize = {
      "1-9 employees": 25000,
      "10-49 employees": 48000,
      "50+ employees": 91000,
    };

    // Tax change percentages based on policy
    const taxChangePercent = {
      "1-9 employees": thresholdEffect * -0.15 + rateEffect * 0.6,
      "10-49 employees": thresholdEffect * -0.05 + rateEffect * 0.8,
      "50+ employees": thresholdEffect * 0.05 + rateEffect * 0.95,
    };

    const result = {};
    for (const [size, baseTax] of Object.entries(baseTaxBySize)) {
      const changePercent = taxChangePercent[size] * 100;
      const changeAmount = baseTax * taxChangePercent[size];
      result[size] = {
        changePercent: changePercent,
        changeAmount: changeAmount,
      };
    }

    return result;
  };

  const formatCurrency = (value) => {
    if (Math.abs(value) >= 1e9) {
      return `£${(value / 1e9).toFixed(1)}bn`;
    } else if (Math.abs(value) >= 1e6) {
      return `£${(value / 1e6).toFixed(1)}m`;
    } else if (Math.abs(value) >= 1e3) {
      return `£${(value / 1e3).toFixed(0)}k`;
    } else {
      return `£${value.toFixed(0)}`;
    }
  };

  const formatPercent = (value) => {
    return `${value > 0 ? "+" : ""}${value.toFixed(1)}%`;
  };

  // Calculate current state values
  const currentRevenue = calculateReformRevenue(filters);
  const baselineRevenue = 164200000000;
  const revenueChange = currentRevenue - baselineRevenue;
  const revenueChangePercent = (revenueChange / baselineRevenue) * 100;
  const businessImpact = calculateBusinessImpact(filters);
  const isBaseline = isBaselinePolicy(filters);

  const handleFiltersChange = (newFilters) => {
    // Instant update
    setAnalysisResults(newFilters);
    setFilters(newFilters);
  };

  return (
    <Layout>
      {/* Watermark - only show when displaying data */}
      {!isBaseline && (
        <div
          style={{
            position: "fixed",
            top: "50%",
            left: "60%",
            transform: "translate(-50%, -50%)",
            fontSize: "6rem",
            fontWeight: "bold",
            color: "rgba(255, 0, 0, 0.08)",
            zIndex: 1000,
            pointerEvents: "none",
            userSelect: "none",
            textShadow: "2px 2px 4px rgba(0,0,0,0.05)",
            whiteSpace: "nowrap",
          }}
        >
          FAKE DATA
        </div>
      )}

      <div className="container">
        <VATAnalysisSidebar
          onFiltersChange={handleFiltersChange}
          initialFilters={filters}
        />

        <div className="main-content">
          {/* Title and Controls Header */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            style={{ 
              position: "relative",
              marginBottom: "2rem" 
            }}
          >
            <h1 style={{ fontSize: "2.5rem", margin: 0, textAlign: "center" }}>
              PolicyEngine VATLab
            </h1>
            {!isBaseline && (
              <div style={{ 
                position: "absolute", 
                top: "50%", 
                right: 0, 
                transform: "translateY(-50%)",
                display: "flex", 
                gap: "0.75rem", 
                alignItems: "center" 
              }}>
                <select
                  value={weightingType}
                  onChange={(e) => setWeightingType(e.target.value)}
                  style={{
                    padding: "0.4rem 0.8rem",
                    borderRadius: "20px",
                    border: "1px solid var(--medium-light-gray)",
                    fontSize: "0.8rem",
                    cursor: "pointer",
                    backgroundColor: "var(--white)",
                    color: "var(--dark-gray)",
                    outline: "none",
                  }}
                >
                  <option value="firms">Firms</option>
                  <option value="revenue">Revenue</option>
                  <option value="employees">Employees</option>
                </select>
                <select
                  value={filters.year}
                  onChange={(e) =>
                    handleFiltersChange({
                      ...filters,
                      year: parseInt(e.target.value),
                    })
                  }
                  style={{
                    padding: "0.4rem 0.8rem",
                    borderRadius: "20px",
                    border: "1px solid var(--medium-light-gray)",
                    fontSize: "0.8rem",
                    cursor: "pointer",
                    backgroundColor: "var(--white)",
                    color: "var(--dark-gray)",
                    outline: "none",
                  }}
                >
                  {[2024, 2025, 2026, 2027, 2028, 2029, 2030].map((year) => (
                    <option key={year} value={year}>
                      {year}
                    </option>
                  ))}
                </select>
              </div>
            )}
          </motion.div>

          {/* Conditional content based on whether parameters have changed */}
          {isBaseline ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                minHeight: "400px",
                padding: "3rem",
              }}
            >
              <div
                style={{
                  backgroundColor: "var(--blue-98)",
                  border: "2px solid var(--blue-95)",
                  borderRadius: "12px",
                  padding: "3rem",
                  maxWidth: "600px",
                  textAlign: "center",
                }}
              >
                <h2
                  style={{
                    color: "var(--darkest-blue)",
                    marginBottom: "1.5rem",
                  }}
                >
                  Define Your VAT Reform
                </h2>
                <p
                  style={{
                    fontSize: "1.1rem",
                    lineHeight: 1.6,
                    marginBottom: "2rem",
                    color: "var(--dark-gray)",
                  }}
                >
                  Use the sidebar controls to modify VAT policy parameters and
                  see how your reforms would impact UK businesses and revenue.
                </p>
                <div
                  style={{
                    backgroundColor: "var(--white)",
                    padding: "1.5rem",
                    borderRadius: "8px",
                    border: "1px solid var(--medium-dark-gray)",
                    textAlign: "left",
                  }}
                >
                  <h3
                    style={{
                      fontSize: "1rem",
                      marginBottom: "1rem",
                      color: "var(--darkest-blue)",
                    }}
                  >
                    Available Parameters:
                  </h3>
                  <ul
                    style={{
                      margin: 0,
                      paddingLeft: "1.5rem",
                      fontSize: "0.95rem",
                      lineHeight: 1.8,
                      color: "var(--dark-gray)",
                    }}
                  >
                    <li>Registration threshold (currently £90,000)</li>
                    <li>VAT rates by sector type</li>
                    <li>Graduated threshold options</li>
                    <li>Implementation year and elasticity</li>
                  </ul>
                </div>
                <div
                  style={{
                    marginTop: "2rem",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: "0.5rem",
                  }}
                >
                  <span style={{ fontSize: "1.5rem" }}>👈</span>
                  <span style={{ fontSize: "1rem", color: "var(--gray)" }}>
                    Adjust parameters in the sidebar to begin
                  </span>
                </div>
              </div>
            </motion.div>
          ) : (
            <>
              {/* Main Results Banner */}
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.1 }}
                style={{
                  background:
                    "linear-gradient(135deg, var(--blue-primary) 0%, var(--blue-pressed) 100%)",
                  color: "var(--white)",
                  marginBottom: "2rem",
                  border: "none",
                  boxShadow: "0 10px 30px rgba(0,0,0,0.1)",
                }}
              >
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(3, 1fr)",
                    gap: "2rem",
                    padding: "1.5rem",
                  }}
                >
                  <div style={{ textAlign: "center" }}>
                    <div
                      style={{
                        fontSize: "2.2rem",
                        fontWeight: "bold",
                        marginBottom: "0.25rem",
                      }}
                    >
                      {formatPercent(revenueChangePercent)}
                    </div>
                    <div style={{ fontSize: "0.9rem", opacity: 0.9 }}>
                      Revenue Change
                    </div>
                  </div>

                  <div style={{ textAlign: "center" }}>
                    <div
                      style={{
                        fontSize: "2.2rem",
                        fontWeight: "bold",
                        marginBottom: "0.25rem",
                      }}
                    >
                      {businessImpact.payingLess.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: "0.9rem", opacity: 0.9 }}>
                      of{" "}
                      {weightingType === "firms"
                        ? "Firms"
                        : weightingType === "revenue"
                          ? "Revenue"
                          : "Employees"}{" "}
                      Paying Less Tax
                    </div>
                  </div>

                  <div style={{ textAlign: "center" }}>
                    <div
                      style={{
                        fontSize: "2.2rem",
                        fontWeight: "bold",
                        marginBottom: "0.25rem",
                      }}
                    >
                      {businessImpact.payingMore.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: "0.9rem", opacity: 0.9 }}>
                      of{" "}
                      {weightingType === "firms"
                        ? "Firms"
                        : weightingType === "revenue"
                          ? "Revenue"
                          : "Employees"}{" "}
                      Paying More Tax
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* Business Impact Distribution */}
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: 0.3 }}
              >
                <div
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "1.5rem",
                  }}
                >
                  <h3 style={{ margin: 0 }}>Business Impact Distribution</h3>
                  <div
                    style={{
                      display: "flex",
                      gap: "1rem",
                      alignItems: "center",
                    }}
                  >
                    <button
                      onClick={() => setShowTaxChange(!showTaxChange)}
                      style={{
                        padding: "0.5rem 1rem",
                        borderRadius: "6px",
                        border: "2px solid var(--medium-light-gray)",
                        fontSize: "0.9rem",
                        cursor: "pointer",
                        backgroundColor: showTaxChange
                          ? "var(--blue-primary)"
                          : "var(--white)",
                        color: showTaxChange
                          ? "var(--white)"
                          : "var(--darkest-blue)",
                        transition: "all 0.2s ease",
                      }}
                    >
                      {showTaxChange ? "Show Distribution" : "Show Tax Change"}
                    </button>
                    <select
                      value={breakdownType}
                      onChange={(e) => setBreakdownType(e.target.value)}
                      style={{
                        padding: "0.5rem 1rem",
                        borderRadius: "6px",
                        border: "2px solid var(--medium-light-gray)",
                        fontSize: "0.9rem",
                        cursor: "pointer",
                        backgroundColor: "var(--white)",
                      }}
                    >
                      <option value="sector">By Sector</option>
                      <option value="size">By Employee Count</option>
                    </select>
                  </div>
                </div>

                <div style={{ marginBottom: "2rem" }}>
                  <div
                    style={{
                      display: "flex",
                      height: "60px",
                      borderRadius: "8px",
                      overflow: "hidden",
                      boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                    }}
                  >
                    <div
                      style={{
                        width: `${businessImpact.payingMore}%`,
                        backgroundColor: "var(--gray)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "white",
                        fontWeight: "bold",
                        fontSize: "1.1rem",
                      }}
                    >
                      {businessImpact.payingMore > 10 &&
                        `${businessImpact.payingMore.toFixed(1)}%`}
                    </div>
                    <div
                      style={{
                        width: `${businessImpact.unaffected}%`,
                        backgroundColor: "var(--fog-gray)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "var(--dark-gray)",
                        fontWeight: "bold",
                        fontSize: "1.1rem",
                      }}
                    >
                      {businessImpact.unaffected > 5 &&
                        `${businessImpact.unaffected.toFixed(1)}%`}
                    </div>
                    <div
                      style={{
                        width: `${businessImpact.payingLess}%`,
                        backgroundColor: "var(--teal-accent)",
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        color: "white",
                        fontWeight: "bold",
                        fontSize: "1.1rem",
                      }}
                    >
                      {businessImpact.payingLess > 10 &&
                        `${businessImpact.payingLess.toFixed(1)}%`}
                    </div>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      marginTop: "1rem",
                    }}
                  >
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem",
                      }}
                    >
                      <div
                        style={{
                          width: "16px",
                          height: "16px",
                          backgroundColor: "var(--gray)",
                          borderRadius: "4px",
                        }}
                      ></div>
                      <span style={{ fontSize: "0.9rem" }}>
                        Paying more tax
                      </span>
                    </div>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem",
                      }}
                    >
                      <div
                        style={{
                          width: "16px",
                          height: "16px",
                          backgroundColor: "var(--fog-gray)",
                          borderRadius: "4px",
                        }}
                      ></div>
                      <span style={{ fontSize: "0.9rem" }}>Unaffected</span>
                    </div>
                    <div
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "0.5rem",
                      }}
                    >
                      <div
                        style={{
                          width: "16px",
                          height: "16px",
                          backgroundColor: "var(--teal-accent)",
                          borderRadius: "4px",
                        }}
                      ></div>
                      <span style={{ fontSize: "0.9rem" }}>
                        Paying less tax
                      </span>
                    </div>
                  </div>
                </div>

                {/* Breakdown Details */}
                {breakdownType === "sector" && !showTaxChange && (
                  <div>
                    {(() => {
                      const sectorData = calculateSectorBreakdown(filters);
                      return Object.entries(sectorData).map(
                        ([sector, data]) => (
                          <div key={sector} style={{ marginBottom: "1rem" }}>
                            <div
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                marginBottom: "0.5rem",
                              }}
                            >
                              <span style={{ fontSize: "0.9rem" }}>
                                {sector}
                              </span>
                              <span
                                style={{
                                  fontSize: "0.9rem",
                                  color:
                                    data.payingLess > data.payingMore
                                      ? "var(--teal-accent)"
                                      : "var(--gray)",
                                  fontWeight: "bold",
                                }}
                              >
                                {data.payingLess > data.payingMore
                                  ? `${data.payingLess.toFixed(0)}% paying less`
                                  : `${data.payingMore.toFixed(0)}% paying more`}
                              </span>
                            </div>
                            <div
                              style={{
                                height: "8px",
                                backgroundColor: "var(--fog-gray)",
                                borderRadius: "4px",
                                overflow: "hidden",
                                display: "flex",
                              }}
                            >
                              <div
                                style={{
                                  width: `${data.payingMore}%`,
                                  height: "100%",
                                  backgroundColor: "var(--gray)",
                                }}
                              ></div>
                              <div
                                style={{
                                  width: `${data.unaffected}%`,
                                  height: "100%",
                                  backgroundColor: "var(--fog-gray)",
                                }}
                              ></div>
                              <div
                                style={{
                                  width: `${data.payingLess}%`,
                                  height: "100%",
                                  backgroundColor: "var(--teal-accent)",
                                }}
                              ></div>
                            </div>
                          </div>
                        ),
                      );
                    })()}
                  </div>
                )}

                {breakdownType === "sector" && showTaxChange && (
                  <div>
                    {(() => {
                      const taxChanges = calculateSectorTaxChanges(filters);
                      return Object.entries(taxChanges).map(
                        ([sector, data]) => (
                          <div key={sector} style={{ marginBottom: "1.5rem" }}>
                            <div
                              style={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "baseline",
                                marginBottom: "0.5rem",
                              }}
                            >
                              <span style={{ fontSize: "0.9rem" }}>
                                {sector}
                              </span>
                              <div style={{ textAlign: "right" }}>
                                <span
                                  style={{
                                    fontSize: "1.1rem",
                                    fontWeight: "bold",
                                    color:
                                      data.changePercent < 0
                                        ? "var(--teal-accent)"
                                        : data.changePercent > 0
                                          ? "var(--gray)"
                                          : "var(--dark-gray)",
                                  }}
                                >
                                  {data.changePercent > 0 ? "+" : ""}
                                  {data.changePercent.toFixed(1)}%
                                </span>
                                <span
                                  style={{
                                    fontSize: "0.8rem",
                                    color: "var(--dark-gray)",
                                    marginLeft: "0.5rem",
                                  }}
                                >
                                  ({data.changeAmount > 0 ? "+" : ""}
                                  {formatCurrency(data.changeAmount)})
                                </span>
                              </div>
                            </div>
                            <div
                              style={{
                                height: "8px",
                                backgroundColor: "var(--fog-gray)",
                                borderRadius: "4px",
                                overflow: "hidden",
                                position: "relative",
                              }}
                            >
                              <div
                                style={{
                                  width: `${Math.min(100, Math.abs(data.changePercent) * 5)}%`,
                                  height: "100%",
                                  backgroundColor:
                                    data.changePercent < 0
                                      ? "var(--teal-accent)"
                                      : "var(--gray)",
                                  transition: "width 0.3s ease",
                                }}
                              ></div>
                            </div>
                          </div>
                        ),
                      );
                    })()}
                  </div>
                )}

                {breakdownType === "size" && !showTaxChange && (
                  <div>
                    <div style={{ marginBottom: "1rem" }}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "0.5rem",
                        }}
                      >
                        <span style={{ fontSize: "0.9rem" }}>
                          1-9 employees
                        </span>
                        <span
                          style={{
                            fontSize: "0.9rem",
                            color: "var(--teal-accent)",
                            fontWeight: "bold",
                          }}
                        >
                          78% paying less
                        </span>
                      </div>
                      <div
                        style={{
                          height: "8px",
                          backgroundColor: "var(--fog-gray)",
                          borderRadius: "4px",
                          overflow: "hidden",
                          display: "flex",
                        }}
                      >
                        <div
                          style={{
                            width: "20%",
                            height: "100%",
                            backgroundColor: "var(--gray)",
                          }}
                        ></div>
                        <div
                          style={{
                            width: "2%",
                            height: "100%",
                            backgroundColor: "var(--fog-gray)",
                          }}
                        ></div>
                        <div
                          style={{
                            width: "78%",
                            height: "100%",
                            backgroundColor: "var(--teal-accent)",
                          }}
                        ></div>
                      </div>
                    </div>

                    <div style={{ marginBottom: "1rem" }}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "0.5rem",
                        }}
                      >
                        <span style={{ fontSize: "0.9rem" }}>
                          10-49 employees
                        </span>
                        <span
                          style={{
                            fontSize: "0.9rem",
                            color: "var(--teal-accent)",
                            fontWeight: "bold",
                          }}
                        >
                          52% paying less
                        </span>
                      </div>
                      <div
                        style={{
                          height: "8px",
                          backgroundColor: "var(--fog-gray)",
                          borderRadius: "4px",
                          overflow: "hidden",
                          display: "flex",
                        }}
                      >
                        <div
                          style={{
                            width: "46%",
                            height: "100%",
                            backgroundColor: "var(--gray)",
                          }}
                        ></div>
                        <div
                          style={{
                            width: "2%",
                            height: "100%",
                            backgroundColor: "var(--fog-gray)",
                          }}
                        ></div>
                        <div
                          style={{
                            width: "52%",
                            height: "100%",
                            backgroundColor: "var(--teal-accent)",
                          }}
                        ></div>
                      </div>
                    </div>

                    <div style={{ marginBottom: "1rem" }}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          marginBottom: "0.5rem",
                        }}
                      >
                        <span style={{ fontSize: "0.9rem" }}>
                          50+ employees
                        </span>
                        <span
                          style={{
                            fontSize: "0.9rem",
                            color: "var(--gray)",
                            fontWeight: "bold",
                          }}
                        >
                          65% paying more
                        </span>
                      </div>
                      <div
                        style={{
                          height: "8px",
                          backgroundColor: "var(--fog-gray)",
                          borderRadius: "4px",
                          overflow: "hidden",
                          display: "flex",
                        }}
                      >
                        <div
                          style={{
                            width: "65%",
                            height: "100%",
                            backgroundColor: "var(--gray)",
                          }}
                        ></div>
                        <div
                          style={{
                            width: "2%",
                            height: "100%",
                            backgroundColor: "var(--fog-gray)",
                          }}
                        ></div>
                        <div
                          style={{
                            width: "33%",
                            height: "100%",
                            backgroundColor: "var(--teal-accent)",
                          }}
                        ></div>
                      </div>
                    </div>
                  </div>
                )}

                {breakdownType === "size" && showTaxChange && (
                  <div>
                    {(() => {
                      const taxChanges = calculateSizeTaxChanges(filters);
                      return Object.entries(taxChanges).map(([size, data]) => (
                        <div key={size} style={{ marginBottom: "1.5rem" }}>
                          <div
                            style={{
                              display: "flex",
                              justifyContent: "space-between",
                              alignItems: "baseline",
                              marginBottom: "0.5rem",
                            }}
                          >
                            <span style={{ fontSize: "0.9rem" }}>{size}</span>
                            <div style={{ textAlign: "right" }}>
                              <span
                                style={{
                                  fontSize: "1.1rem",
                                  fontWeight: "bold",
                                  color:
                                    data.changePercent < 0
                                      ? "var(--teal-accent)"
                                      : data.changePercent > 0
                                        ? "var(--gray)"
                                        : "var(--dark-gray)",
                                }}
                              >
                                {data.changePercent > 0 ? "+" : ""}
                                {data.changePercent.toFixed(1)}%
                              </span>
                              <span
                                style={{
                                  fontSize: "0.8rem",
                                  color: "var(--dark-gray)",
                                  marginLeft: "0.5rem",
                                }}
                              >
                                ({data.changeAmount > 0 ? "+" : ""}
                                {formatCurrency(data.changeAmount)})
                              </span>
                            </div>
                          </div>
                          <div
                            style={{
                              height: "8px",
                              backgroundColor: "var(--fog-gray)",
                              borderRadius: "4px",
                              overflow: "hidden",
                              position: "relative",
                            }}
                          >
                            <div
                              style={{
                                width: `${Math.min(100, Math.abs(data.changePercent) * 5)}%`,
                                height: "100%",
                                backgroundColor:
                                  data.changePercent < 0
                                    ? "var(--teal-accent)"
                                    : "var(--gray)",
                                transition: "width 0.3s ease",
                              }}
                            ></div>
                          </div>
                        </div>
                      ));
                    })()}
                  </div>
                )}
              </motion.div>
            </>
          )}

          <div className="helper-text" style={{ marginTop: "2rem" }}>
            <span className="info-icon" title="About this analysis">
              ⓘ
            </span>
            <span>
              This dashboard is a demo created with fake data. It shows
              PolicyEngine&apos;s analysis of VAT reform options for the UK. Adjust
              parameters in the sidebar to see how different policies would
              impact revenue and businesses.
            </span>
          </div>
        </div>
      </div>
    </Layout>
  );
}
