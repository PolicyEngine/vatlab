import { useState } from "react";
import { motion } from "framer-motion";

export default function VATAnalysisSidebar({
  onFiltersChange,
  initialFilters,
}) {
  const [filters, setFilters] = useState(
    initialFilters || {
      threshold: 90000,
      graduatedEndThreshold: "",
      fullRateLaborIntensive: 20,
      fullRateNonLaborIntensive: 20,
      year: 2026,
      elasticity: -0.01,
    },
  );

  const years = Array.from({ length: 21 }, (_, i) => 2020 + i); // 2020-2040

  const handleFilterChange = (key, value) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
    // Automatically trigger analysis
    onFiltersChange(newFilters);
  };

  return (
    <motion.div
      className="sidebar"
      initial={{ x: -320, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.5 }}
    >
      <div className="top-logo">
        <img src="/images/logo.png" alt="PolicyEngine Logo" />
      </div>

      <div
        className="sidebar-card"
        style={{
          overflow: "auto",
          maxHeight: "calc(100vh - 100px)",
          padding: "1.5rem",
        }}
      >
        <h3
          className="sidebar-title"
          style={{ textAlign: "center", marginBottom: "1rem" }}
        >
          VAT Policy Reform
        </h3>

        {/* Current UK Policy Info Box */}
        <div
          style={{
            backgroundColor: "var(--blue-98)",
            padding: "1rem",
            borderRadius: "8px",
            marginBottom: "1.5rem",
            border: "1px solid var(--blue-primary)",
            fontSize: "0.85rem",
          }}
        >
          <strong style={{ color: "var(--blue-pressed)" }}>
            Baseline: Current UK Policy
          </strong>
          <div style={{ marginTop: "0.5rem", color: "var(--dark-gray)" }}>
            • £90,000 registration threshold
            <br />• 20% standard VAT rate for all sectors
          </div>
        </div>

        {/* Policy Parameters Section */}
        <div
          style={{
            fontSize: "0.85rem",
            fontWeight: "600",
            color: "var(--gray)",
            marginBottom: "0.75rem",
            marginTop: "1rem",
            textTransform: "uppercase",
            letterSpacing: "0.05em",
          }}
        >
          Policy Parameters
        </div>

        <div
          className="form-group"
          style={{ display: "flex", alignItems: "center", gap: "1rem" }}
        >
          <div style={{ flex: 1 }}>
            <label style={{ marginBottom: "0.25rem" }}>Year Applies</label>
            <div
              style={{
                fontSize: "0.75rem",
                color: "var(--gray)",
                lineHeight: 1.3,
              }}
            >
              Year when reform takes effect
            </div>
          </div>
          <select
            value={filters.year}
            onChange={(e) =>
              handleFilterChange("year", parseInt(e.target.value))
            }
            className="select-dropdown"
            style={{ width: "120px", flexShrink: 0 }}
          >
            {years.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </div>

        <div
          className="form-group"
          style={{ display: "flex", alignItems: "center", gap: "1rem" }}
        >
          <div style={{ flex: 1 }}>
            <label style={{ marginBottom: "0.25rem" }}>
              Registration Threshold
            </label>
            <div
              style={{
                fontSize: "0.75rem",
                color: "var(--gray)",
                lineHeight: 1.3,
              }}
            >
              Annual turnover for VAT registration
            </div>
          </div>
          <input
            type="number"
            value={filters.threshold}
            onChange={(e) =>
              handleFilterChange("threshold", parseInt(e.target.value) || 0)
            }
            className="select-dropdown"
            placeholder="90000"
            min="0"
            step="1000"
            style={{
              width: "120px",
              flexShrink: 0,
              ...(filters.threshold !== 90000
                ? {
                    borderColor: "var(--blue-primary)",
                    borderWidth: "2px",
                    boxShadow: "0 0 0 2px var(--blue-light)",
                  }
                : {}),
            }}
          />
        </div>

        <div
          className="form-group"
          style={{ display: "flex", alignItems: "center", gap: "1rem" }}
        >
          <div style={{ flex: 1 }}>
            <label style={{ marginBottom: "0.25rem" }}>
              Graduated End Threshold
            </label>
            <div
              style={{
                fontSize: "0.75rem",
                color: "var(--gray)",
                lineHeight: 1.3,
              }}
            >
              Upper threshold for smooth transition (optional)
            </div>
          </div>
          <input
            type="number"
            value={filters.graduatedEndThreshold}
            onChange={(e) =>
              handleFilterChange(
                "graduatedEndThreshold",
                e.target.value ? parseInt(e.target.value) : "",
              )
            }
            className="select-dropdown"
            placeholder="None"
            min="0"
            step="1000"
            style={{ width: "120px", flexShrink: 0 }}
          />
        </div>

        <div
          className="form-group"
          style={{ display: "flex", alignItems: "center", gap: "1rem" }}
        >
          <div style={{ flex: 1 }}>
            <label style={{ marginBottom: "0.25rem" }}>
              Labor-Intensive Rate (%)
            </label>
            <div
              style={{
                fontSize: "0.75rem",
                color: "var(--gray)",
                lineHeight: 1.3,
              }}
            >
              VAT rate for labor-intensive services
            </div>
          </div>
          <input
            type="number"
            value={filters.fullRateLaborIntensive}
            onChange={(e) =>
              handleFilterChange(
                "fullRateLaborIntensive",
                parseFloat(e.target.value) || 0,
              )
            }
            className="select-dropdown"
            placeholder="20"
            min="0"
            max="100"
            step="0.1"
            style={{
              width: "120px",
              flexShrink: 0,
              ...(filters.fullRateLaborIntensive !== 20
                ? {
                    borderColor: "var(--blue-primary)",
                    borderWidth: "2px",
                    boxShadow: "0 0 0 2px var(--blue-light)",
                  }
                : {}),
            }}
          />
        </div>

        <div
          className="form-group"
          style={{ display: "flex", alignItems: "center", gap: "1rem" }}
        >
          <div style={{ flex: 1 }}>
            <label style={{ marginBottom: "0.25rem" }}>Standard Rate (%)</label>
            <div
              style={{
                fontSize: "0.75rem",
                color: "var(--gray)",
                lineHeight: 1.3,
              }}
            >
              VAT rate for other business sectors
            </div>
          </div>
          <input
            type="number"
            value={filters.fullRateNonLaborIntensive}
            onChange={(e) =>
              handleFilterChange(
                "fullRateNonLaborIntensive",
                parseFloat(e.target.value) || 0,
              )
            }
            className="select-dropdown"
            placeholder="20"
            min="0"
            max="100"
            step="0.1"
            style={{
              width: "120px",
              flexShrink: 0,
              ...(filters.fullRateNonLaborIntensive !== 20
                ? {
                    borderColor: "var(--blue-primary)",
                    borderWidth: "2px",
                    boxShadow: "0 0 0 2px var(--blue-light)",
                  }
                : {}),
            }}
          />
        </div>

        {/* Model Parameters Section */}
        <div
          style={{
            marginTop: "1.5rem",
            paddingTop: "1.5rem",
            borderTop: "2px solid var(--medium-dark-gray)",
          }}
        >
          <div
            style={{
              fontSize: "0.85rem",
              fontWeight: "600",
              color: "var(--gray)",
              marginBottom: "0.75rem",
              textTransform: "uppercase",
              letterSpacing: "0.05em",
            }}
          >
            Model Parameters
          </div>
          <div
            className="form-group"
            style={{
              display: "flex",
              alignItems: "center",
              gap: "1rem",
              marginBottom: 0,
            }}
          >
            <div style={{ flex: 1 }}>
              <label style={{ marginBottom: "0.25rem" }}>Elasticity</label>
              <div
                style={{
                  fontSize: "0.75rem",
                  color: "var(--gray)",
                  lineHeight: 1.3,
                }}
              >
                Firm response to threshold changes
              </div>
            </div>
            <input
              type="number"
              value={filters.elasticity}
              onChange={(e) =>
                handleFilterChange(
                  "elasticity",
                  parseFloat(e.target.value) || 0,
                )
              }
              className="select-dropdown"
              placeholder="-0.01"
              min="-5"
              max="5"
              step="0.1"
              style={{ width: "120px", flexShrink: 0 }}
            />
          </div>
        </div>
      </div>
    </motion.div>
  );
}
