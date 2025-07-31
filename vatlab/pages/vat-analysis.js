import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Layout from '../components/Layout';
import Loading from '../components/Loading';
import Tabs from '../components/Tabs';
import VATAnalysisSidebar from '../components/VATAnalysisSidebar';
import { 
  VATElasticityRevenueChart,
  VATRevenueHistoryChart,
  VATRegistrationChart,
  VATRateComparisonChart
} from '../components/VATChart';

export default function VATAnalysis() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedScenario, setSelectedScenario] = useState('baseline');
  const [filters, setFilters] = useState({
    threshold: 90000,
    graduatedEndThreshold: '',
    laborIntensiveIndustries: ['eu_default'],
    fullRateLaborIntensive: 20,
    fullRateNonLaborIntensive: 20,
    year: 2026,
    elasticity: -0.01
  });
  const [analysisResults, setAnalysisResults] = useState(null);
  const [filteredData, setFilteredData] = useState(null);

  useEffect(() => {
    // Initialize with mock data for demonstration
    const mockData = {
      summary: {
        current_threshold: 90000,
        scenarios_analyzed: 5
      },
      key_findings: {
        total_firms_analyzed: 5600000,
        current_vat_registered: 2300000,
        registration_rate: 0.41,
        highest_impact_scenario: 'split_rate',
        labour_intensive_sectors: 8
      },
      policy_scenarios: {
        baseline: {
          description: 'Current System: £90k threshold',
          total_vat_revenue: 164200000000,
          revenue_change_from_baseline: 0,
          firms_affected: 0
        },
        higher_threshold: {
          description: 'Higher Threshold: £100k',
          total_vat_revenue: 162800000000,
          revenue_change_from_baseline: -1400000000,
          firms_affected: 45000
        },
        graduated_threshold_1: {
          description: 'Graduated Threshold: Taper 1',
          total_vat_revenue: 165100000000,
          revenue_change_from_baseline: 900000000,
          firms_affected: 85000
        },
        graduated_threshold_2: {
          description: 'Graduated Threshold: Taper 2',
          total_vat_revenue: 164800000000,
          revenue_change_from_baseline: 600000000,
          firms_affected: 75000
        },
        split_rate: {
          description: 'Split Rate System: 10% for labor-intensive',
          total_vat_revenue: 159500000000,
          revenue_change_from_baseline: -4700000000,
          firms_affected: 125000
        }
      },
      scenarios_metadata: {
        split_rate: 'Split rate system: 10% for labor-intensive services'
      },
      sectoral_impacts: {
        'Professional Services': {
          is_labour_intensive: false,
          total_firms: 450000,
          current_vat_registered: 280000,
          scenario_impacts: {
            baseline: { total_vat_liability: 12500000000 },
            higher_threshold: { total_vat_liability: 12200000000 },
            graduated_threshold_1: { total_vat_liability: 12650000000 },
            graduated_threshold_2: { total_vat_liability: 12580000000 },
            split_rate: { total_vat_liability: 12500000000 }
          }
        },
        'Hospitality': {
          is_labour_intensive: true,
          total_firms: 180000,
          current_vat_registered: 95000,
          scenario_impacts: {
            baseline: { total_vat_liability: 8900000000 },
            higher_threshold: { total_vat_liability: 8700000000 },
            graduated_threshold_1: { total_vat_liability: 9100000000 },
            graduated_threshold_2: { total_vat_liability: 9000000000 },
            split_rate: { total_vat_liability: 4450000000 }
          }
        },
        'Construction': {
          is_labour_intensive: false,
          total_firms: 320000,
          current_vat_registered: 210000,
          scenario_impacts: {
            baseline: { total_vat_liability: 15600000000 },
            higher_threshold: { total_vat_liability: 15200000000 },
            graduated_threshold_1: { total_vat_liability: 15800000000 },
            graduated_threshold_2: { total_vat_liability: 15700000000 },
            split_rate: { total_vat_liability: 15600000000 }
          }
        },
        'Retail': {
          is_labour_intensive: false,
          total_firms: 290000,
          current_vat_registered: 185000,
          scenario_impacts: {
            baseline: { total_vat_liability: 22100000000 },
            higher_threshold: { total_vat_liability: 21600000000 },
            graduated_threshold_1: { total_vat_liability: 22400000000 },
            graduated_threshold_2: { total_vat_liability: 22250000000 },
            split_rate: { total_vat_liability: 22100000000 }
          }
        },
        'Healthcare Services': {
          is_labour_intensive: true,
          total_firms: 85000,
          current_vat_registered: 45000,
          scenario_impacts: {
            baseline: { total_vat_liability: 3200000000 },
            higher_threshold: { total_vat_liability: 3100000000 },
            graduated_threshold_1: { total_vat_liability: 3250000000 },
            graduated_threshold_2: { total_vat_liability: 3220000000 },
            split_rate: { total_vat_liability: 1600000000 }
          }
        },
        'Personal Services': {
          is_labour_intensive: true,
          total_firms: 125000,
          current_vat_registered: 75000,
          scenario_impacts: {
            baseline: { total_vat_liability: 2800000000 },
            higher_threshold: { total_vat_liability: 2700000000 },
            graduated_threshold_1: { total_vat_liability: 2850000000 },
            graduated_threshold_2: { total_vat_liability: 2820000000 },
            split_rate: { total_vat_liability: 1400000000 }
          }
        },
        'Manufacturing': {
          is_labour_intensive: false,
          total_firms: 195000,
          current_vat_registered: 145000,
          scenario_impacts: {
            baseline: { total_vat_liability: 18500000000 },
            higher_threshold: { total_vat_liability: 18100000000 },
            graduated_threshold_1: { total_vat_liability: 18700000000 },
            graduated_threshold_2: { total_vat_liability: 18600000000 },
            split_rate: { total_vat_liability: 18500000000 }
          }
        },
        'Technology Services': {
          is_labour_intensive: false,
          total_firms: 165000,
          current_vat_registered: 98000,
          scenario_impacts: {
            baseline: { total_vat_liability: 9200000000 },
            higher_threshold: { total_vat_liability: 8900000000 },
            graduated_threshold_1: { total_vat_liability: 9350000000 },
            graduated_threshold_2: { total_vat_liability: 9270000000 },
            split_rate: { total_vat_liability: 9200000000 }
          }
        }
      },
      threshold_behavior: {
        growth_effects: {
          firms_limiting_turnover: 125000,
          estimated_growth_suppression: 2.8,
          hours_reduction: 15.4,
          client_turnaway: 12.3
        },
        elasticity_effects: {
          turnover_elasticity: -0.8,
          size_reduction: -12.5,
          registration_elasticity: -0.6
        },
        threshold_bunching: [
          { range: '80k-90k', firms: 95000 }
        ]
      },
      firm_distribution: {
        by_size: [
          { size_band: '0-50k', firms: 1800000 },
          { size_band: '50k-90k', firms: 450000 },
          { size_band: '90k-150k', firms: 180000 },
          { size_band: '150k+', firms: 70000 }
        ]
      }
    };
    
    setData(mockData);
    setFilteredData(mockData);
    setLoading(false);
  }, []);

  const handleFiltersChange = (newFilters) => {
    setLoading(true);
    setFilters(newFilters);
    setAnalysisResults(newFilters);
    // Simulate analysis processing
    setTimeout(() => {
      setLoading(false);
    }, 1500);
  };

  const applyFiltersToData = (rawData, currentFilters) => {
    if (!rawData) return null;
    
    // Calculate elasticity impact factor based on selected elasticity
    const elasticityImpactFactor = Math.max(0.5, Math.min(2.0, Math.abs(currentFilters.elasticity) * 50 + 1)); // Scale between 0.5x and 2.0x
    
    // Filter data based on selected parameters
    const filtered = {
      ...rawData,
      // Apply year filtering to time-series data
      policy_scenarios: Object.fromEntries(
        Object.entries(rawData.policy_scenarios).map(([key, scenario]) => {
          // Apply elasticity impact to revenue calculations
          const adjustedRevenue = scenario.total_vat_revenue * elasticityImpactFactor;
          const adjustedChangeFromBaseline = key === 'baseline' ? 0 : 
            (adjustedRevenue - rawData.policy_scenarios.baseline.total_vat_revenue * elasticityImpactFactor);
          
          return [
            key,
            {
              ...scenario,
              total_vat_revenue: adjustedRevenue,
              revenue_change_from_baseline: adjustedChangeFromBaseline,
              firms_affected: Math.round(scenario.firms_affected * elasticityImpactFactor),
              // Filter any time-series data within scenarios
              yearly_data: scenario.yearly_data?.filter(
                item => item.year >= currentFilters.startYear && item.year <= currentFilters.endYear
              ).map(item => ({
                ...item,
                revenue: item.revenue * elasticityImpactFactor
              })) || []
            }
          ];
        })
      ),
      // Apply elasticity filtering and adjust behavioral effects
      threshold_behavior: {
        ...rawData.threshold_behavior,
        growth_effects: {
          ...rawData.threshold_behavior.growth_effects,
          firms_limiting_turnover: Math.round(rawData.threshold_behavior.growth_effects.firms_limiting_turnover * elasticityImpactFactor),
          estimated_growth_suppression: rawData.threshold_behavior.growth_effects.estimated_growth_suppression * elasticityImpactFactor,
          hours_reduction: rawData.threshold_behavior.growth_effects.hours_reduction * elasticityImpactFactor,
          client_turnaway: rawData.threshold_behavior.growth_effects.client_turnaway * elasticityImpactFactor
        },
        elasticity_effects: {
          ...rawData.threshold_behavior.elasticity_effects,
          // Filter elasticity values within range
          filtered_elasticities: rawData.threshold_behavior.elasticity_effects.elasticities?.filter(
            e => Math.abs(e - currentFilters.elasticity) <= 0.1
          ) || [],
          // Adjust elasticity metrics based on range
          turnover_elasticity: rawData.threshold_behavior.elasticity_effects.turnover_elasticity * elasticityImpactFactor,
          size_reduction: rawData.threshold_behavior.elasticity_effects.size_reduction * elasticityImpactFactor,
          registration_elasticity: rawData.threshold_behavior.elasticity_effects.registration_elasticity * elasticityImpactFactor
        }
      },
      // Adjust sectoral impacts based on elasticity
      sectoral_impacts: Object.fromEntries(
        Object.entries(rawData.sectoral_impacts).map(([sector, data]) => [
          sector,
          {
            ...data,
            scenario_impacts: Object.fromEntries(
              Object.entries(data.scenario_impacts).map(([scenarioKey, impact]) => [
                scenarioKey,
                {
                  ...impact,
                  total_vat_liability: impact.total_vat_liability * elasticityImpactFactor
                }
              ])
            )
          }
        ])
      ),
      // Add elasticity metadata
      elasticity_adjustment: {
        range: [currentFilters.elasticity - 0.1, currentFilters.elasticity + 0.1],
        impact_factor: elasticityImpactFactor,
        description: `Analysis adjusted for elasticity value ${currentFilters.elasticity}`
      }
    };
    
    return filtered;
  };

  // Update filtered data when filters change
  useEffect(() => {
    if (data) {
      const filtered = applyFiltersToData(data, filters);
      setFilteredData(filtered);
    }
  }, [data, filters]);

  const formatCurrency = (value) => {
    if (Math.abs(value) >= 1000000000) {
      return `£${(value / 1000000000).toFixed(1)}bn`;
    } else if (Math.abs(value) >= 1000000) {
      return `£${(value / 1000000).toFixed(1)}m`;
    } else {
      return new Intl.NumberFormat('en-GB', { 
        style: 'currency', 
        currency: 'GBP',
        maximumFractionDigits: 0
      }).format(value);
    }
  };

  const formatNumber = (value) => {
    return new Intl.NumberFormat('en-GB').format(value);
  };

  const tabs = [
    { 
      label: "Policy Reform Impact",
      description: "Analysis of revenue impacts over time and firm-level winners and losers from the proposed VAT policy reforms."
    },
    { 
      label: "Calibration & Official Statistics",
      description: "Model calibration parameters and official UK government VAT statistics, revenue trends, and comparative analysis with other tax systems."
    },
    { 
      label: "Simulation Guide",
      description: ""
    },
    { 
      label: "Replication Code",
      description: "Python code to replicate the VAT policy analysis and generate the results shown in this dashboard."
    }
  ];

  if (loading) {
    return (
      <Layout>
        <div className="container">
          <VATAnalysisSidebar onFiltersChange={handleFiltersChange} initialFilters={filters} loading={loading} />
          <div className="main-content">
            <h1>PolicyEngine VATLab</h1>
            <Loading />
          </div>
        </div>
      </Layout>
    );
  }

  if (!filteredData) {
    return (
      <Layout>
        <div className="container">
          <VATAnalysisSidebar onFiltersChange={handleFiltersChange} initialFilters={filters} loading={loading} />
          <div className="main-content">
            <h1>PolicyEngine VATLab</h1>
            <div className="card">
              <h3>Data not available</h3>
              <p>Unable to load VAT analysis data. Please try again later.</p>
            </div>
          </div>
        </div>
      </Layout>
    );
  }

  return (
    <Layout>
      {/* Watermark */}
      <div style={{
        position: 'fixed',
        top: '50%',
        left: '60%',
        transform: 'translate(-50%, -50%)',
        fontSize: '6rem',
        fontWeight: 'bold',
        color: 'rgba(255, 0, 0, 0.08)',
        zIndex: 1000,
        pointerEvents: 'none',
        userSelect: 'none',
        textShadow: '2px 2px 4px rgba(0,0,0,0.05)',
        whiteSpace: 'nowrap'
      }}>
        FAKE DATA
      </div>
      <div className="container">
        <VATAnalysisSidebar onFiltersChange={handleFiltersChange} initialFilters={filters} loading={loading} />
        <div className="main-content">
          <motion.h1
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
          >
            PolicyEngine VATLab
          </motion.h1>
          

          <motion.div 
            className="area-impact-summary"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
            style={{ marginBottom: '2rem' }}
          >
            {analysisResults ? (
              <div className="key-results-banner" style={{
                background: 'linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%)',
                color: 'white',
                padding: '2rem',
                borderRadius: '12px',
                marginBottom: '1.5rem',
                boxShadow: '0 4px 12px rgba(0,0,0,0.15)'
              }}>
                <h2 style={{ color: 'white', marginBottom: '1.5rem', fontSize: '1.8rem', fontWeight: '600' }}>Key Policy Impact Results</h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1.5rem' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2.5rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                      {formatCurrency(168500000000)}
                    </div>
                    <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Total VAT Revenue</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2.5rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#fbbf24' }}>
                      +£2.3bn
                    </div>
                    <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Revenue Change</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2.5rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                      125k
                    </div>
                    <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Firms Affected</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2.5rem', fontWeight: 'bold', marginBottom: '0.5rem', color: '#ef4444' }}>
                      -£4,300
                    </div>
                    <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Avg Firm Impact</div>
                  </div>
                </div>
              </div>
              <div style={{ 
                backgroundColor: '#fafafa',
                padding: '1.5rem',
                borderRadius: '8px',
                border: '1px solid #e5e7eb',
                marginTop: '1.5rem'
              }}>
                <h3 style={{ marginTop: 0, marginBottom: '1rem' }}>Policy Comparison</h3>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                  <div>
                    <h4 style={{ color: '#6b7280', fontSize: '0.9rem', marginBottom: '0.5rem' }}>BASELINE (Current UK Policy)</h4>
                    <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.9rem', lineHeight: 1.6 }}>
                      <li>£90,000 registration threshold</li>
                      <li>20% standard VAT rate for all sectors</li>
                      <li>No graduated threshold system</li>
                    </ul>
                  </div>
                  <div>
                    <h4 style={{ color: '#3b82f6', fontSize: '0.9rem', marginBottom: '0.5rem' }}>REFORM (Your Policy)</h4>
                    <ul style={{ margin: 0, paddingLeft: '1.2rem', fontSize: '0.9rem', lineHeight: 1.6 }}>
                      <li>£{analysisResults.threshold?.toLocaleString()} registration threshold{analysisResults.graduatedEndThreshold && ` (graduating to £${analysisResults.graduatedEndThreshold.toLocaleString()})`}</li>
                      <li>{analysisResults.fullRateLaborIntensive}% rate for labor-intensive sectors</li>
                      <li>{analysisResults.fullRateNonLaborIntensive}% rate for other businesses</li>
                    </ul>
                  </div>
                </div>
                <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#6b7280' }}>
                  Analysis year: {analysisResults.year} | Elasticity: {analysisResults.elasticity}
                </div>
              </div>
            ) : null}
            
            {!analysisResults ? (
              <div>
                <div style={{ 
                  backgroundColor: '#f0f9ff',
                  border: '2px solid #3b82f6',
                  borderRadius: '12px',
                  padding: '2rem',
                  marginBottom: '1.5rem'
                }}>
                  <h3 style={{ marginTop: 0, color: '#1e40af', fontSize: '1.4rem' }}>Quick Start Guide</h3>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '1rem' }}>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                      <div style={{ 
                        backgroundColor: '#3b82f6',
                        color: 'white',
                        borderRadius: '50%',
                        width: '28px',
                        height: '28px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                        fontWeight: 'bold'
                      }}>1</div>
                      <div>
                        <strong>Choose Baseline:</strong> Select the "Baseline" tab to analyze current UK VAT policy or customize baseline parameters
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                      <div style={{ 
                        backgroundColor: '#3b82f6',
                        color: 'white',
                        borderRadius: '50%',
                        width: '28px',
                        height: '28px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                        fontWeight: 'bold'
                      }}>2</div>
                      <div>
                        <strong>Configure Reform:</strong> Switch to "Reform" tab to set your policy parameters
                      </div>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'flex-start', gap: '1rem' }}>
                      <div style={{ 
                        backgroundColor: '#3b82f6',
                        color: 'white',
                        borderRadius: '50%',
                        width: '28px',
                        height: '28px',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                        fontWeight: 'bold'
                      }}>3</div>
                      <div>
                        <strong>Run Analysis:</strong> Click "Analyse VAT Policy" to see the impact
                      </div>
                    </div>
                  </div>
                </div>
                
                <div style={{ 
                  backgroundColor: '#fafafa',
                  padding: '1.5rem',
                  borderRadius: '8px',
                  border: '1px solid #e5e7eb'
                }}>
                  <h4 style={{ marginTop: 0, color: '#374151' }}>Current UK VAT Landscape</h4>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: '1rem' }}>
                    <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
                      <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#1e40af' }}>
                        {formatNumber(filteredData.key_findings.total_firms_analyzed)}
                      </div>
                      <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>Total UK Businesses</div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
                      <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#059669' }}>
                        {formatNumber(filteredData.key_findings.current_vat_registered)}
                      </div>
                      <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>VAT Registered</div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
                      <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#dc2626' }}>
                        {(filteredData.key_findings.registration_rate * 100).toFixed(1)}%
                      </div>
                      <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>Registration Rate</div>
                    </div>
                    <div style={{ textAlign: 'center', padding: '1rem', backgroundColor: 'white', borderRadius: '8px', border: '1px solid #e5e7eb' }}>
                      <div style={{ fontSize: '1.8rem', fontWeight: 'bold', color: '#7c3aed' }}>
                        £{filteredData.summary.current_threshold.toLocaleString()}
                      </div>
                      <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>VAT Threshold</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            <div className="helper-text">
              <span className="info-icon" title="About this analysis">ⓘ</span>
              <span>
                This dashboard is a demo created with fake data. This dashboard shows PolicyEngine's analysis of VAT reform options for the UK.
                {analysisResults && " Configure parameters in the sidebar and press 'Analyse VAT Policy' to update the analysis."}
              </span>
            </div>
          </motion.div>

          <Tabs tabs={tabs}>
            {/* Policy Reform Impact Tab */}
            <div>
              <div className="stats" style={{ marginBottom: '2rem' }}>
                <motion.div 
                  className="stat"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                  style={{ 
                    backgroundColor: '#f0f9ff',
                    border: '2px solid #3b82f6',
                    borderRadius: '12px',
                    padding: '1.5rem',
                    textAlign: 'center'
                  }}
                >
                  <h3 style={{ color: '#1e40af', fontSize: '1rem', marginBottom: '0.5rem' }}>Total VAT Revenue</h3>
                  <p style={{ fontSize: '2.2rem', fontWeight: 'bold', color: '#1e3a8a', margin: '0.5rem 0' }}>{formatCurrency(168500000000)}</p>
                  <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>Projected for 2026</div>
                </motion.div>
                <motion.div 
                  className="stat"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                  style={{ 
                    backgroundColor: '#f0fdf4',
                    border: '2px solid #22c55e',
                    borderRadius: '12px',
                    padding: '1.5rem',
                    textAlign: 'center'
                  }}
                >
                  <h3 style={{ color: '#166534', fontSize: '1rem', marginBottom: '0.5rem' }}>Winners</h3>
                  <p style={{ fontSize: '2.2rem', fontWeight: 'bold', color: '#059669', margin: '0.5rem 0' }}>34.2%</p>
                  <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>Firms gaining &gt;10%</div>
                </motion.div>
                <motion.div 
                  className="stat"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                  style={{ 
                    backgroundColor: '#fef2f2',
                    border: '2px solid #ef4444',
                    borderRadius: '12px',
                    padding: '1.5rem',
                    textAlign: 'center'
                  }}
                >
                  <h3 style={{ color: '#991b1b', fontSize: '1rem', marginBottom: '0.5rem' }}>Losers</h3>
                  <p style={{ fontSize: '2.2rem', fontWeight: 'bold', color: '#dc2626', margin: '0.5rem 0' }}>12.8%</p>
                  <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>Firms losing &gt;10%</div>
                </motion.div>
                <motion.div 
                  className="stat"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.4 }}
                  style={{ 
                    backgroundColor: '#fffbeb',
                    border: '2px solid #f59e0b',
                    borderRadius: '12px',
                    padding: '1.5rem',
                    textAlign: 'center'
                  }}
                >
                  <h3 style={{ color: '#92400e', fontSize: '1rem', marginBottom: '0.5rem' }}>Average Firm Impact</h3>
                  <p style={{ fontSize: '2.2rem', fontWeight: 'bold', color: '#d97706', margin: '0.5rem 0' }}>{formatCurrency(-4300)}</p>
                  <div style={{ fontSize: '0.85rem', color: '#6b7280' }}>Change in turnover</div>
                </motion.div>
              </div>
              
              {/* Revenue Over Time Chart */}
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <h3>VAT Revenue Projection (2026-2029)</h3>
                <div style={{ 
                  height: '550px', 
                  backgroundColor: '#f8f9fa', 
                  border: '1px solid #ddd', 
                  borderRadius: '8px',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  position: 'relative'
                }}>
                  <svg width="800" height="500" viewBox="0 0 800 500">
                    {/* Grid lines - horizontal */}
                    <line x1="100" y1="80" x2="720" y2="80" stroke="#e5e7eb" strokeWidth="1" strokeDasharray="2,2"/>
                    <line x1="100" y1="140" x2="720" y2="140" stroke="#e5e7eb" strokeWidth="1" strokeDasharray="2,2"/>
                    <line x1="100" y1="200" x2="720" y2="200" stroke="#e5e7eb" strokeWidth="1" strokeDasharray="2,2"/>
                    <line x1="100" y1="260" x2="720" y2="260" stroke="#e5e7eb" strokeWidth="1" strokeDasharray="2,2"/>
                    <line x1="100" y1="320" x2="720" y2="320" stroke="#e5e7eb" strokeWidth="1" strokeDasharray="2,2"/>
                    <line x1="100" y1="380" x2="720" y2="380" stroke="#e5e7eb" strokeWidth="1" strokeDasharray="2,2"/>
                    
                    {/* Grid lines - vertical */}
                    <line x1="200" y1="80" x2="200" y2="400" stroke="#e5e7eb" strokeWidth="1" strokeDasharray="2,2"/>
                    <line x1="350" y1="80" x2="350" y2="400" stroke="#e5e7eb" strokeWidth="1" strokeDasharray="2,2"/>
                    <line x1="500" y1="80" x2="500" y2="400" stroke="#e5e7eb" strokeWidth="1" strokeDasharray="2,2"/>
                    <line x1="650" y1="80" x2="650" y2="400" stroke="#e5e7eb" strokeWidth="1" strokeDasharray="2,2"/>
                    
                    {/* Axes */}
                    <line x1="100" y1="400" x2="720" y2="400" stroke="#333" strokeWidth="2"/>
                    <line x1="100" y1="80" x2="100" y2="400" stroke="#333" strokeWidth="2"/>
                    
                    {/* Y-axis labels */}
                    <text x="90" y="85" fontSize="14" textAnchor="end" fill="#333">£200bn</text>
                    <text x="90" y="145" fontSize="14" textAnchor="end" fill="#333">£160bn</text>
                    <text x="90" y="205" fontSize="14" textAnchor="end" fill="#333">£120bn</text>
                    <text x="90" y="265" fontSize="14" textAnchor="end" fill="#333">£80bn</text>
                    <text x="90" y="325" fontSize="14" textAnchor="end" fill="#333">£40bn</text>
                    <text x="90" y="385" fontSize="14" textAnchor="end" fill="#333">£0</text>
                    
                    {/* X-axis labels */}
                    <text x="200" y="430" fontSize="16" textAnchor="middle" fill="#333" fontWeight="bold">2026</text>
                    <text x="350" y="430" fontSize="16" textAnchor="middle" fill="#333" fontWeight="bold">2027</text>
                    <text x="500" y="430" fontSize="16" textAnchor="middle" fill="#333" fontWeight="bold">2028</text>
                    <text x="650" y="430" fontSize="16" textAnchor="middle" fill="#333" fontWeight="bold">2029</text>
                    
                    {/* Baseline bars - starting from 0 */}
                    <rect x="160" y="192" width="50" height="208" fill="#94a3b8" opacity="0.8" stroke="#6b7280" strokeWidth="1"/>
                    <rect x="310" y="184" width="50" height="216" fill="#94a3b8" opacity="0.8" stroke="#6b7280" strokeWidth="1"/>
                    <rect x="460" y="176" width="50" height="224" fill="#94a3b8" opacity="0.8" stroke="#6b7280" strokeWidth="1"/>
                    <rect x="610" y="168" width="50" height="232" fill="#94a3b8" opacity="0.8" stroke="#6b7280" strokeWidth="1"/>
                    
                    {/* Reform bars - starting from 0 */}
                    <rect x="220" y="184" width="50" height="216" fill="#3b82f6" opacity="0.9" stroke="#1d4ed8" strokeWidth="1"/>
                    <rect x="370" y="172" width="50" height="228" fill="#3b82f6" opacity="0.9" stroke="#1d4ed8" strokeWidth="1"/>
                    <rect x="520" y="160" width="50" height="240" fill="#3b82f6" opacity="0.9" stroke="#1d4ed8" strokeWidth="1"/>
                    <rect x="670" y="148" width="50" height="252" fill="#3b82f6" opacity="0.9" stroke="#1d4ed8" strokeWidth="1"/>
                  
                    
                    {/* Legend - larger and more prominent */}
                    <rect x="550" y="100" width="140" height="60" fill="rgba(255, 255, 255, 0.9)" stroke="#d1d5db" strokeWidth="1" rx="5"/>
                    <rect x="565" y="115" width="20" height="12" fill="#94a3b8" opacity="0.8" stroke="#6b7280" strokeWidth="1"/>
                    <text x="595" y="125" fontSize="14" fill="#333" fontWeight="bold">Baseline</text>
                    <rect x="565" y="135" width="20" height="12" fill="#3b82f6" opacity="0.9" stroke="#1d4ed8" strokeWidth="1"/>
                    <text x="595" y="145" fontSize="14" fill="#333" fontWeight="bold">Reform</text>
                    
                    {/* Chart title */}
                    <text x="410" y="50" fontSize="18" textAnchor="middle" fill="#333" fontWeight="bold">Annual VAT Revenue Comparison</text>
                  </svg>
                </div>
              </motion.div>

              {/* Winners and Losers Chart */}
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
              >
                <h3>Firm Impact Distribution by Industry in 2026</h3>
                <div style={{ 
                  height: '500px', 
                  backgroundColor: '#f8f9fa', 
                  border: '1px solid #ddd', 
                  borderRadius: '8px',
                  display: 'flex',
                  justifyContent: 'center',
                  alignItems: 'center',
                  position: 'relative'
                }}>
                  <svg width="700" height="450" viewBox="0 0 700 450">
                    {/* Axes */}
                    <line x1="100" y1="50" x2="600" y2="50" stroke="#333" strokeWidth="1"/>
                    <line x1="100" y1="50" x2="100" y2="400" stroke="#333" strokeWidth="1"/>
                    
                    {/* X-axis labels */}
                    <text x="200" y="40" fontSize="12" textAnchor="middle" fill="#666">25%</text>
                    <text x="300" y="40" fontSize="12" textAnchor="middle" fill="#666">50%</text>
                    <text x="400" y="40" fontSize="12" textAnchor="middle" fill="#666">75%</text>
                    <text x="500" y="40" fontSize="12" textAnchor="middle" fill="#666">100%</text>
                    <text x="350" y="25" fontSize="14" textAnchor="middle" fill="#333" fontWeight="bold">Population Share</text>
                    
                    {/* Y-axis labels */}
                    <text x="90" y="70" fontSize="12" textAnchor="end" fill="#666">All</text>
                    <text x="90" y="100" fontSize="12" textAnchor="end" fill="#666">Technology</text>
                    <text x="90" y="130" fontSize="12" textAnchor="end" fill="#666">Professional</text>
                    <text x="90" y="160" fontSize="12" textAnchor="end" fill="#666">Manufacturing</text>
                    <text x="90" y="190" fontSize="12" textAnchor="end" fill="#666">Construction</text>
                    <text x="90" y="220" fontSize="12" textAnchor="end" fill="#666">Retail</text>
                    <text x="90" y="250" fontSize="12" textAnchor="end" fill="#666">Hospitality</text>
                    <text x="90" y="280" fontSize="12" textAnchor="end" fill="#666">Healthcare</text>
                    <text x="90" y="310" fontSize="12" textAnchor="end" fill="#666">Personal Services</text>
                    <text x="90" y="340" fontSize="12" textAnchor="end" fill="#666">Transport</text>
                    <text x="90" y="370" fontSize="12" textAnchor="end" fill="#666">Agriculture</text>
                    
                    {/* Define the data for each decile - percentages that sum to 100 */}
                    {/* All */}
                    <rect x="100" y="55" width="56" height="20" fill="#1e3a8a" opacity="0.9"/> {/* Gain >5%: 14% */}
                    <rect x="156" y="55" width="80" height="20" fill="#3b82f6" opacity="0.7"/> {/* Gain <5%: 20% */}
                    <rect x="236" y="55" width="148" height="20" fill="#d1d5db" opacity="0.8"/> {/* No change: 37% */}
                    <rect x="384" y="55" width="76" height="20" fill="#a3a3a3" opacity="0.7"/> {/* Loss <5%: 19% */}
                    <rect x="460" y="55" width="40" height="20" fill="#525252" opacity="0.9"/> {/* Loss >5%: 10% */}
                    <text x="580" y="68" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Decile 10 (richest) */}
                    <rect x="100" y="85" width="80" height="20" fill="#1e3a8a" opacity="0.9"/> {/* Gain >5%: 20% */}
                    <rect x="180" y="85" width="100" height="20" fill="#3b82f6" opacity="0.7"/> {/* Gain <5%: 25% */}
                    <rect x="280" y="85" width="120" height="20" fill="#d1d5db" opacity="0.8"/> {/* No change: 30% */}
                    <rect x="400" y="85" width="80" height="20" fill="#a3a3a3" opacity="0.7"/> {/* Loss <5%: 20% */}
                    <rect x="480" y="85" width="20" height="20" fill="#525252" opacity="0.9"/> {/* Loss >5%: 5% */}
                    <text x="580" y="98" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Decile 9 */}
                    <rect x="100" y="115" width="72" height="20" fill="#1e3a8a" opacity="0.9"/> {/* 18% */}
                    <rect x="172" y="115" width="88" height="20" fill="#3b82f6" opacity="0.7"/> {/* 22% */}
                    <rect x="260" y="115" width="140" height="20" fill="#d1d5db" opacity="0.8"/> {/* 35% */}
                    <rect x="400" y="115" width="80" height="20" fill="#a3a3a3" opacity="0.7"/> {/* 20% */}
                    <rect x="480" y="115" width="20" height="20" fill="#525252" opacity="0.9"/> {/* 5% */}
                    <text x="580" y="128" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Continue pattern for remaining deciles with varying distributions */}
                    {/* Decile 8 */}
                    <rect x="100" y="145" width="64" height="20" fill="#1e3a8a" opacity="0.9"/>
                    <rect x="164" y="145" width="84" height="20" fill="#3b82f6" opacity="0.7"/>
                    <rect x="248" y="145" width="144" height="20" fill="#d1d5db" opacity="0.8"/>
                    <rect x="392" y="145" width="84" height="20" fill="#a3a3a3" opacity="0.7"/>
                    <rect x="476" y="145" width="24" height="20" fill="#525252" opacity="0.9"/>
                    <text x="580" y="158" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Decile 7 */}
                    <rect x="100" y="175" width="56" height="20" fill="#1e3a8a" opacity="0.9"/>
                    <rect x="156" y="175" width="80" height="20" fill="#3b82f6" opacity="0.7"/>
                    <rect x="236" y="175" width="148" height="20" fill="#d1d5db" opacity="0.8"/>
                    <rect x="384" y="175" width="88" height="20" fill="#a3a3a3" opacity="0.7"/>
                    <rect x="472" y="175" width="28" height="20" fill="#525252" opacity="0.9"/>
                    <text x="580" y="188" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Decile 6 */}
                    <rect x="100" y="205" width="48" height="20" fill="#1e3a8a" opacity="0.9"/>
                    <rect x="148" y="205" width="76" height="20" fill="#3b82f6" opacity="0.7"/>
                    <rect x="224" y="205" width="152" height="20" fill="#d1d5db" opacity="0.8"/>
                    <rect x="376" y="205" width="92" height="20" fill="#a3a3a3" opacity="0.7"/>
                    <rect x="468" y="205" width="32" height="20" fill="#525252" opacity="0.9"/>
                    <text x="580" y="218" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Decile 5 */}
                    <rect x="100" y="235" width="40" height="20" fill="#1e3a8a" opacity="0.9"/>
                    <rect x="140" y="235" width="72" height="20" fill="#3b82f6" opacity="0.7"/>
                    <rect x="212" y="235" width="156" height="20" fill="#d1d5db" opacity="0.8"/>
                    <rect x="368" y="235" width="96" height="20" fill="#a3a3a3" opacity="0.7"/>
                    <rect x="464" y="235" width="36" height="20" fill="#525252" opacity="0.9"/>
                    <text x="580" y="248" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Decile 4 */}
                    <rect x="100" y="265" width="32" height="20" fill="#1e3a8a" opacity="0.9"/>
                    <rect x="132" y="265" width="68" height="20" fill="#3b82f6" opacity="0.7"/>
                    <rect x="200" y="265" width="160" height="20" fill="#d1d5db" opacity="0.8"/>
                    <rect x="360" y="265" width="100" height="20" fill="#a3a3a3" opacity="0.7"/>
                    <rect x="460" y="265" width="40" height="20" fill="#525252" opacity="0.9"/>
                    <text x="580" y="278" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Decile 3 */}
                    <rect x="100" y="295" width="24" height="20" fill="#1e3a8a" opacity="0.9"/>
                    <rect x="124" y="295" width="64" height="20" fill="#3b82f6" opacity="0.7"/>
                    <rect x="188" y="295" width="164" height="20" fill="#d1d5db" opacity="0.8"/>
                    <rect x="352" y="295" width="104" height="20" fill="#a3a3a3" opacity="0.7"/>
                    <rect x="456" y="295" width="44" height="20" fill="#525252" opacity="0.9"/>
                    <text x="580" y="308" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Decile 2 */}
                    <rect x="100" y="325" width="16" height="20" fill="#1e3a8a" opacity="0.9"/>
                    <rect x="116" y="325" width="60" height="20" fill="#3b82f6" opacity="0.7"/>
                    <rect x="176" y="325" width="168" height="20" fill="#d1d5db" opacity="0.8"/>
                    <rect x="344" y="325" width="108" height="20" fill="#a3a3a3" opacity="0.7"/>
                    <rect x="452" y="325" width="48" height="20" fill="#525252" opacity="0.9"/>
                    <text x="580" y="338" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Decile 1 (poorest) */}
                    <rect x="100" y="355" width="8" height="20" fill="#1e3a8a" opacity="0.9"/>
                    <rect x="108" y="355" width="56" height="20" fill="#3b82f6" opacity="0.7"/>
                    <rect x="164" y="355" width="172" height="20" fill="#d1d5db" opacity="0.8"/>
                    <rect x="336" y="355" width="112" height="20" fill="#a3a3a3" opacity="0.7"/>
                    <rect x="448" y="355" width="52" height="20" fill="#525252" opacity="0.9"/>
                    <text x="580" y="368" fontSize="11" textAnchor="start" fill="#666">100%</text>
                    
                    {/* Legend */}
                    <text x="100" y="420" fontSize="14" fontWeight="bold" fill="#333">Legend:</text>
                    <rect x="100" y="425" width="15" height="10" fill="#1e3a8a" opacity="0.9"/>
                    <text x="120" y="434" fontSize="11" fill="#666">Gain &gt;5%</text>
                    <rect x="180" y="425" width="15" height="10" fill="#3b82f6" opacity="0.7"/>
                    <text x="200" y="434" fontSize="11" fill="#666">Gain &lt;5%</text>
                    <rect x="260" y="425" width="15" height="10" fill="#d1d5db" opacity="0.8"/>
                    <text x="280" y="434" fontSize="11" fill="#666">No change</text>
                    <rect x="340" y="425" width="15" height="10" fill="#a3a3a3" opacity="0.7"/>
                    <text x="360" y="434" fontSize="11" fill="#666">Loss &lt;5%</text>
                    <rect x="420" y="425" width="15" height="10" fill="#525252" opacity="0.9"/>
                    <text x="440" y="434" fontSize="11" fill="#666">Loss &gt;5%</text>
                  </svg>
                </div>
              </motion.div>
            </div>
            
            {/* Calibration & Official Statistics Tab */}
            <div>
              <div className="stats">
                <motion.div 
                  className="stat"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.1 }}
                >
                  <h3>VAT Revenue 2024</h3>
                  <p>£164.2bn</p>
                  <div className="helper-text-small" title="Official HMRC VAT receipts for 2024">ⓘ</div>
                </motion.div>
                <motion.div 
                  className="stat"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.2 }}
                >
                  <h3>VAT Registered Businesses</h3>
                  <p>2.3m</p>
                  <div className="helper-text-small" title="Total number of VAT-registered businesses in UK">ⓘ</div>
                </motion.div>
                <motion.div 
                  className="stat"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <h3>Current VAT Rate</h3>
                  <p>20%</p>
                  <div className="helper-text-small" title="Standard VAT rate in UK since 2011">ⓘ</div>
                </motion.div>
                <motion.div 
                  className="stat"
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ duration: 0.5, delay: 0.4 }}
                >
                  <h3>VAT Threshold</h3>
                  <p>£90k</p>
                  <div className="helper-text-small" title="Annual turnover threshold for mandatory VAT registration">ⓘ</div>
                </motion.div>
              </div>

              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                <div style={{lineHeight: '1.6'}}>
                  <p>Our microsimulation model is calibrated against official UK government statistics to ensure accuracy. The charts below show the comparison between our model outputs and official HMRC data.</p>
                  
                  <table style={{ width: '100%', marginTop: '1.5rem', marginBottom: '1.5rem', borderCollapse: 'collapse', border: '1px solid #ddd' }}>
                    <thead>
                      <tr style={{ backgroundColor: '#f8f9fa' }}>
                        <th style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.9rem', fontWeight: 'bold', textAlign: 'left' }}>Metric</th>
                        <th style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.9rem', fontWeight: 'bold', textAlign: 'center' }}>RMSE</th>
                        <th style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.9rem', fontWeight: 'bold', textAlign: 'center' }}>Mean Absolute Error (%)</th>
                        <th style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.9rem', fontWeight: 'bold', textAlign: 'center' }}>R²</th>
                        <th style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.9rem', fontWeight: 'bold', textAlign: 'center' }}>Quantiles Error (25th-75th)</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', fontWeight: '500' }}>VAT Revenue (£bn)</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>2.8</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>2.1%</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>0.987</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>1.4% - 2.9%</td>
                      </tr>
                      <tr style={{ backgroundColor: '#f8f9fa' }}>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', fontWeight: '500' }}>VAT Registered Firms (millions)</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>0.12</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>3.4%</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>0.978</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>2.1% - 4.8%</td>
                      </tr>
                      <tr>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', fontWeight: '500' }}>Sectoral VAT Liability</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>1.6</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>4.2%</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>0.954</td>
                        <td style={{ padding: '0.75rem', border: '1px solid #ddd', fontSize: '0.85rem', textAlign: 'center' }}>2.8% - 6.1%</td>
                      </tr>
                    </tbody>
                  </table>
                  
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '1.5rem' }}>
                    <div>
                      <h4 style={{ marginBottom: '1rem' }}>VAT Revenue Calibration</h4>
                      <div style={{ 
                        height: '400px', 
                        backgroundColor: '#f8f9fa', 
                        border: '1px solid #ddd', 
                        borderRadius: '8px',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'center',
                        alignItems: 'center',
                        position: 'relative'
                      }}>
                        <svg width="360" height="280" viewBox="0 0 360 280">
                          <defs>
                            <linearGradient id="govGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                              <stop offset="0%" stopColor="#4A90E2" stopOpacity="0.8"/>
                              <stop offset="100%" stopColor="#4A90E2" stopOpacity="0.3"/>
                            </linearGradient>
                            <linearGradient id="modelGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                              <stop offset="0%" stopColor="#50C878" stopOpacity="0.8"/>
                              <stop offset="100%" stopColor="#50C878" stopOpacity="0.3"/>
                            </linearGradient>
                          </defs>
                          
                          {/* Axes */}
                          <line x1="50" y1="220" x2="330" y2="220" stroke="#333" strokeWidth="1"/>
                          <line x1="50" y1="30" x2="50" y2="220" stroke="#333" strokeWidth="1"/>
                          
                          {/* Y-axis labels */}
                          <text x="40" y="35" fontSize="12" textAnchor="end" fill="#666">£180bn</text>
                          <text x="40" y="80" fontSize="12" textAnchor="end" fill="#666">£150bn</text>
                          <text x="40" y="125" fontSize="12" textAnchor="end" fill="#666">£120bn</text>
                          <text x="40" y="170" fontSize="12" textAnchor="end" fill="#666">£90bn</text>
                          <text x="40" y="225" fontSize="12" textAnchor="end" fill="#666">£60bn</text>
                          
                          {/* X-axis labels */}
                          <text x="90" y="240" fontSize="12" textAnchor="middle" fill="#666">2019</text>
                          <text x="150" y="240" fontSize="12" textAnchor="middle" fill="#666">2020</text>
                          <text x="210" y="240" fontSize="12" textAnchor="middle" fill="#666">2021</text>
                          <text x="270" y="240" fontSize="12" textAnchor="middle" fill="#666">2022</text>
                          <text x="330" y="240" fontSize="12" textAnchor="middle" fill="#666">2023</text>
                          
                          {/* Government data line */}
                          <polyline
                            fill="none"
                            stroke="#4A90E2"
                            strokeWidth="4"
                            points="90,100 150,130 210,110 270,85 330,70"
                          />
                          
                          {/* Model data line */}
                          <polyline
                            fill="none"
                            stroke="#50C878"
                            strokeWidth="4"
                            strokeDasharray="7,7"
                            points="90,108 150,125 210,115 270,92 330,77"
                          />
                          
                          {/* Data points */}
                          <circle cx="90" cy="100" r="5" fill="#4A90E2"/>
                          <circle cx="150" cy="130" r="5" fill="#4A90E2"/>
                          <circle cx="210" cy="110" r="5" fill="#4A90E2"/>
                          <circle cx="270" cy="85" r="5" fill="#4A90E2"/>
                          <circle cx="330" cy="70" r="5" fill="#4A90E2"/>
                          
                          <circle cx="90" cy="108" r="5" fill="#50C878"/>
                          <circle cx="150" cy="125" r="5" fill="#50C878"/>
                          <circle cx="210" cy="115" r="5" fill="#50C878"/>
                          <circle cx="270" cy="92" r="5" fill="#50C878"/>
                          <circle cx="330" cy="77" r="5" fill="#50C878"/>
                        </svg>
                        
                        <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem', fontSize: '0.85rem' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                            <div style={{ width: '12px', height: '3px', backgroundColor: '#4A90E2' }}></div>
                            <span>HMRC Official</span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                            <div style={{ width: '12px', height: '3px', backgroundColor: '#50C878', backgroundImage: 'repeating-linear-gradient(90deg, transparent, transparent 3px, white 3px, white 6px)' }}></div>
                            <span>Model Output</span>
                          </div>
                        </div>
                      </div>
                      <p style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.5rem', textAlign: 'center' }}>
                        Mean Absolute Error: 2.1% | R² = 0.987
                      </p>
                    </div>

                    <div>
                      <h4 style={{ marginBottom: '1rem' }}>VAT-Registered Firms Calibration</h4>
                      <div style={{ 
                        height: '400px', 
                        backgroundColor: '#f8f9fa', 
                        border: '1px solid #ddd', 
                        borderRadius: '8px',
                        display: 'flex',
                        flexDirection: 'column',
                        justifyContent: 'center',
                        alignItems: 'center',
                        position: 'relative'
                      }}>
                        <svg width="360" height="280" viewBox="0 0 360 280">
                          {/* Axes */}
                          <line x1="50" y1="220" x2="330" y2="220" stroke="#333" strokeWidth="1"/>
                          <line x1="50" y1="30" x2="50" y2="220" stroke="#333" strokeWidth="1"/>
                          
                          {/* Y-axis labels */}
                          <text x="40" y="35" fontSize="12" textAnchor="end" fill="#666">2.4m</text>
                          <text x="40" y="80" fontSize="12" textAnchor="end" fill="#666">2.2m</text>
                          <text x="40" y="125" fontSize="12" textAnchor="end" fill="#666">2.0m</text>
                          <text x="40" y="170" fontSize="12" textAnchor="end" fill="#666">1.8m</text>
                          <text x="40" y="225" fontSize="12" textAnchor="end" fill="#666">1.6m</text>
                          
                          {/* X-axis labels */}
                          <text x="90" y="240" fontSize="12" textAnchor="middle" fill="#666">2019</text>
                          <text x="150" y="240" fontSize="12" textAnchor="middle" fill="#666">2020</text>
                          <text x="210" y="240" fontSize="12" textAnchor="middle" fill="#666">2021</text>
                          <text x="270" y="240" fontSize="12" textAnchor="middle" fill="#666">2022</text>
                          <text x="330" y="240" fontSize="12" textAnchor="middle" fill="#666">2023</text>
                          
                          {/* Government data bars */}
                          <rect x="75" y="130" width="20" height="90" fill="#4A90E2" opacity="0.7"/>
                          <rect x="135" y="135" width="20" height="85" fill="#4A90E2" opacity="0.7"/>
                          <rect x="195" y="115" width="20" height="105" fill="#4A90E2" opacity="0.7"/>
                          <rect x="255" y="100" width="20" height="120" fill="#4A90E2" opacity="0.7"/>
                          <rect x="315" y="85" width="20" height="135" fill="#4A90E2" opacity="0.7"/>
                          
                          {/* Model data bars */}
                          <rect x="95" y="125" width="20" height="95" fill="#50C878" opacity="0.7"/>
                          <rect x="155" y="130" width="20" height="90" fill="#50C878" opacity="0.7"/>
                          <rect x="215" y="110" width="20" height="110" fill="#50C878" opacity="0.7"/>
                          <rect x="275" y="95" width="20" height="125" fill="#50C878" opacity="0.7"/>
                          <rect x="335" y="80" width="20" height="140" fill="#50C878" opacity="0.7"/>
                        </svg>
                        
                        <div style={{ display: 'flex', gap: '1rem', marginTop: '0.5rem', fontSize: '0.85rem' }}>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                            <div style={{ width: '12px', height: '8px', backgroundColor: '#4A90E2', opacity: 0.7 }}></div>
                            <span>ONS Official</span>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                            <div style={{ width: '12px', height: '8px', backgroundColor: '#50C878', opacity: 0.7 }}></div>
                            <span>Model Output</span>
                          </div>
                        </div>
                      </div>
                      <p style={{ fontSize: '0.85rem', color: '#666', marginTop: '0.5rem', textAlign: 'center' }}>
                        Mean Absolute Error: 3.4% | R² = 0.978
                      </p>
                    </div>
                  </div>
                </div>
              </motion.div>
              
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.4 }}
                style={{ width: '100%' }}
              >
                <VATRevenueHistoryChart />
              </motion.div>
              
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.5 }}
                style={{ width: '100%' }}
              >
                <VATRegistrationChart />
              </motion.div>
              
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.6 }}
                style={{ width: '100%' }}
              >
                <VATRateComparisonChart />
              </motion.div>
              
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                <h3>UK VAT Statistics & International Context</h3>
                <div style={{lineHeight: '1.6'}}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginBottom: '1.5rem' }}>
                    <div>
                      <h4 style={{ color: '#2563eb', marginBottom: '0.8rem' }}>Revenue Performance</h4>
                      <ul style={{ margin: 0, paddingLeft: '1.2rem' }}>
                        <li><strong>£164.2bn annual revenue</strong> (18% of total tax receipts)</li>
                        <li><strong>6.8% of GDP</strong> with 7.2% collection gap</li>
                        <li><strong>3.2% annual growth</strong> over past decade</li>
                      </ul>
                    </div>
                    
                    <div>
                      <h4 style={{ color: '#2563eb', marginBottom: '0.8rem' }}>Business Impact</h4>
                      <ul style={{ margin: 0, paddingLeft: '1.2rem' }}>
                        <li><strong>2.3m registered businesses</strong> (41% of all UK firms)</li>
                        <li><strong>£2,800 average compliance cost</strong> per business</li>
                        <li><strong>87% micro/small enterprises</strong> among registered</li>
                      </ul>
                    </div>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem' }}>
                    <div>
                      <h4 style={{ color: '#dc2626', marginBottom: '0.8rem' }}>International Comparison</h4>
                      <ul style={{ margin: 0, paddingLeft: '1.2rem' }}>
                        <li><strong>20% standard rate</strong> (12th highest in EU)</li>
                        <li><strong>£90k threshold</strong> (highest in Europe)</li>
                        <li>Germany €22k, France €85k, Netherlands €20k</li>
                      </ul>
                    </div>
                    
                    <div>
                      <h4 style={{ color: '#dc2626', marginBottom: '0.8rem' }}>Research Evidence</h4>
                      <ul style={{ margin: 0, paddingLeft: '1.2rem' }}>
                        <li><strong>15-20% firms</strong> show growth limitation effects</li>
                        <li><strong>£2.8bn productivity loss</strong> from threshold bunching</li>
                        <li><strong>85% VAT pass-through</strong> to consumer prices</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </motion.div>
            </div>
          
            
            {/* Simulation Guide Tab */}
            <div>
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                <h3>How to Use the VAT Simulation Tool</h3>
                <div style={{lineHeight: '1.6'}}>
                  <h4>Getting Started</h4>
                  <ol>
                    <li><strong>Configure Parameters:</strong> Use the sidebar to set VAT thresholds, rates, and analysis parameters</li>
                    <li><strong>Select Scenarios:</strong> Choose from baseline UK policy or custom reform options</li>
                    <li><strong>Run Analysis:</strong> Click "Analyse VAT Policy" to generate results</li>
                    <li><strong>Review Results:</strong> Switch between tabs to explore impacts and statistics</li>
                  </ol>
                </div>
              </motion.div>

              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.15 }}
              >
                <h3>Key Policy Options</h3>
                <div style={{lineHeight: '1.6'}}>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
                    <div>
                      <h4>Threshold Reforms</h4>
                      <ul>
                        <li><strong>Higher Threshold:</strong> Reduce compliance burden for small businesses</li>
                        <li><strong>Graduated Threshold:</strong> Smooth transition to reduce cliff-edge effects</li>
                      </ul>
                    </div>
                    <div>
                      <h4>Rate Reforms</h4>
                      <ul>
                        <li><strong>Split Rates:</strong> Lower rates for labor-intensive services</li>
                        <li><strong>Standard Rate:</strong> Current 20% across all sectors</li>
                      </ul>
                    </div>
                  </div>
                </div>
              </motion.div>

              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
              >
                <h3>Understanding the Results</h3>
                <div style={{lineHeight: '1.6'}}>
                  <h4>Revenue Impact</h4>
                  <p>Shows projected changes in total VAT receipts, including behavioral responses and sectoral effects.</p>
                  
                  <h4>Firm Distribution</h4>
                  <p>Displays how reforms affect businesses across different size bands and industries, including winners and losers.</p>
                  
                  <h4>Calibration Data</h4>
                  <p>Compares model outputs with official HMRC and ONS statistics to ensure accuracy.</p>
                </div>
              </motion.div>

              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.25 }}
              >
                <h3>Data Sources</h3>
                <div style={{lineHeight: '1.6'}}>
                  <p>The simulation model uses official UK government datasets to ensure accuracy and reliability:</p>
                  
                  <ul>
                    <li><strong>ONS UK Business Statistics:</strong> Firm counts by industry (SIC), turnover bands, and employment size</li>
                    <li><strong>Living Costs and Food Survey (LCFS):</strong> Household spending patterns, income, and demographics</li>
                    <li><strong>HMRC VAT Statistics:</strong> Official VAT receipts by sector and trader registration data</li>
                    <li><strong>ONS Input-Output Tables:</strong> Supply chain linkages for indirect VAT effects</li>
                  </ul>
                </div>
              </motion.div>

              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                <h3>Methodology</h3>
                <div style={{lineHeight: '1.6'}}>
                  <h4>Simulation Approach</h4>
                  <ol>
                    <li><strong>Firm Classification:</strong> Map businesses to VAT registration status based on turnover thresholds</li>
                    <li><strong>Policy Impact:</strong> Calculate direct effects of threshold and rate changes on firm VAT liability</li>
                    <li><strong>Behavioral Response:</strong> Apply research-based elasticities for firm growth and consumer demand</li>
                    <li><strong>Revenue Calculation:</strong> Aggregate impacts across all sectors including supply chain effects</li>
                  </ol>
                  
                  <h4>Key Parameters</h4>
                  <p>The model incorporates price elasticities, VAT pass-through rates, and threshold bunching effects from recent academic research to provide realistic policy impact estimates.</p>
                </div>
              </motion.div>

              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.35 }}
              >
                <h3>Research Evidence</h3>
                <div style={{lineHeight: '1.6'}}>
                  <p>The simulation model incorporates behavioral elasticities from recent academic research on UK VAT policy:</p>
                  
                  <table className="card-table" style={{ width: '100%', marginTop: '1rem', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr style={{ backgroundColor: '#f5f5f5' }}>
                        <th style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.9rem' }}>Study</th>
                        <th style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.9rem' }}>Finding</th>
                        <th style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.9rem' }}>Impact</th>
                      </tr>
                    </thead>
                    <tbody>
                      <tr>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>Liu, Lockwood & Tam (2024)</td>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>Firms slow growth near £90k threshold</td>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>1-2pp annual growth reduction</td>
                      </tr>
                      <tr>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>IMF (Benedek et al., 2015)</td>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>VAT pass-through to consumer prices</td>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>~100% for standard rate</td>
                      </tr>
                      <tr>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>Crossley, Low & Sleeman (2014)</td>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>2008 UK VAT cut effects</td>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>75% price pass-through, 1% sales uplift</td>
                      </tr>
                      <tr>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>Ross & Warwick (2021)</td>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>Firm bunching below threshold</td>
                        <td style={{ padding: '0.5rem', border: '1px solid #ddd', fontSize: '0.85rem' }}>Clear clustering just below £90k</td>
                      </tr>
                    </tbody>
                  </table>
                </div>
              </motion.div>

              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.3 }}
              >
                <h3>Important Notes</h3>
                <div style={{lineHeight: '1.6'}}>
                  <div className="helper-text" style={{ backgroundColor: '#fff3cd', border: '1px solid #ffeaa7', padding: '1rem', borderRadius: '8px', marginBottom: '1rem' }}>
                    <strong>⚠️ Demo Data:</strong> This dashboard uses simulated data for demonstration purposes. Results should not be used for actual policy decisions.
                  </div>
                  
                  <ul>
                    <li>Model incorporates behavioral elasticities from academic research</li>
                    <li>Results include both direct and indirect economic effects</li>
                    <li>Sensitivity analysis available through parameter adjustment</li>
                    <li>All calculations based on UK official statistics and survey data</li>
                  </ul>
                </div>
              </motion.div>
            </div>
            
            {/* Replication Code Tab */}
            <div>
              <motion.div
                className="card"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: 0.1 }}
              >
                <h3>VAT Policy Analysis - Python Replication Code</h3>
                <div style={{lineHeight: '1.6'}}>
                  <p>Use this Python code to replicate the VAT policy analysis shown in this dashboard:</p>
                  
                  <div style={{ 
                    backgroundColor: '#f8f9fa', 
                    border: '1px solid #e9ecef', 
                    borderRadius: '8px', 
                    padding: '1.5rem', 
                    marginTop: '1rem',
                    fontFamily: 'Monaco, Consolas, "Courier New", monospace',
                    fontSize: '0.9rem',
                    lineHeight: '1.4',
                    overflow: 'auto'
                  }}>
                    <pre style={{ margin: 0, whiteSpace: 'pre-wrap' }}>
{`import policyengine_uk as pe
import pandas as pd
import numpy as np

# Load UK VAT policy parameters
uk_policy = pe.country_specific_params.UK()
vat_threshold = 90000  # Current UK VAT threshold

# Simulate VAT reform scenarios
reform_scenarios = pe.vat_simulator.create_scenarios(
    thresholds=[90000, 100000], rates=[0.2, 0.1]
)

# Calculate revenue impacts by industry
results = pe.microsim.run_analysis(reform_scenarios)
print(f"Revenue impact: £{results.total_revenue_change:.1f}bn")`}
                    </pre>
                  </div>
                  
                  <div style={{ marginTop: '1.5rem', padding: '1rem', backgroundColor: '#fff3cd', border: '1px solid #ffeaa7', borderRadius: '8px' }}>
                    <strong>⚠️ Note:</strong> This is demonstration code only. The actual PolicyEngine UK package and methods may differ. 
                    Visit <a href="https://policyengine.org" target="_blank" rel="noopener noreferrer" style={{ color: '#1976d2' }}>policyengine.org</a> for real implementation details.
                  </div>
                </div>
              </motion.div>
            </div>
          </Tabs>
        </div>
      </div>
    </Layout>
  );
}