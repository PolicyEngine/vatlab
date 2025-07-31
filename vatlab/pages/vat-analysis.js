import { useState, useEffect } from 'react';
import Layout from '../components/Layout';
import VATAnalysisSidebar from '../components/VATAnalysisSidebar';
import Loading from '../components/Loading';
import { motion } from 'framer-motion';

export default function VATAnalysis() {
  const [filters, setFilters] = useState({
    threshold: 90000,
    graduatedEndThreshold: '',
    fullRateLaborIntensive: 20,
    fullRateNonLaborIntensive: 20,
    year: 2026,
    elasticity: -0.01
  });
  
  const [analysisResults, setAnalysisResults] = useState(null);
  const [breakdownType, setBreakdownType] = useState('sector');

  // Helper functions
  const isBaselinePolicy = (params) => {
    return params.threshold === 90000 && 
           params.fullRateLaborIntensive === 20 && 
           params.fullRateNonLaborIntensive === 20 &&
           !params.graduatedEndThreshold;
  };

  const calculateReformRevenue = (params) => {
    const baseRevenue = 164200000000;
    const thresholdChange = (params.threshold - 90000) / 90000;
    const avgRateChange = ((params.fullRateLaborIntensive + params.fullRateNonLaborIntensive) / 2 - 20) / 20;
    
    const thresholdImpact = baseRevenue * thresholdChange * -0.02;
    const rateImpact = baseRevenue * avgRateChange * 0.9;
    
    return baseRevenue + thresholdImpact + rateImpact;
  };

  const calculateBusinessImpact = (params) => {
    const thresholdIncrease = params.threshold > 90000;
    const avgRate = (params.fullRateLaborIntensive + params.fullRateNonLaborIntensive) / 2;
    const rateDecrease = avgRate < 20;
    
    let winners, losers;
    if (thresholdIncrease && rateDecrease) {
      winners = 62.3;
      losers = 37.7;
    } else if (thresholdIncrease || rateDecrease) {
      winners = 54.2;
      losers = 45.8;
    } else if (params.threshold < 90000 || avgRate > 20) {
      winners = 28.5;
      losers = 71.5;
    } else {
      winners = 50.0;
      losers = 50.0;
    }
    
    return { winners, losers };
  };
  
  const calculateSectorBreakdown = (params) => {
    const thresholdEffect = (params.threshold - 90000) / 90000;
    const avgRate = (params.fullRateLaborIntensive + params.fullRateNonLaborIntensive) / 2;
    const rateEffect = (avgRate - 20) / 20;
    
    // Base winner percentages by sector (at baseline)
    const baseWinners = {
      'Wholesale & Retail Trade': 50,
      'Accommodation & Food': 50,
      'Professional & Scientific': 50,
      'Construction': 50,
      'Manufacturing': 50
    };
    
    // Adjust based on policy changes
    const adjustments = {
      'Wholesale & Retail Trade': thresholdEffect * 30 - rateEffect * 10, // Benefits from higher threshold
      'Accommodation & Food': thresholdEffect * 25 - rateEffect * 15, // Labor intensive, mixed effect
      'Professional & Scientific': thresholdEffect * 10 - rateEffect * 20, // Less threshold sensitive
      'Construction': thresholdEffect * -10 - rateEffect * 25, // Larger firms, hurt by changes
      'Manufacturing': thresholdEffect * -15 - rateEffect * 30 // Capital intensive, hurt most
    };
    
    const result = {};
    for (const [sector, base] of Object.entries(baseWinners)) {
      const winners = Math.max(10, Math.min(90, base + adjustments[sector]));
      result[sector] = {
        winners: winners,
        losers: 100 - winners
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
    return `${value > 0 ? '+' : ''}${value.toFixed(1)}%`;
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
      )}
      
      <div className="container">
        <VATAnalysisSidebar 
          onFiltersChange={handleFiltersChange} 
          initialFilters={filters} 
        />
        
        <div className="main-content">
          {/* Title without the removed content */}
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            style={{ textAlign: 'center', marginBottom: '2rem' }}
          >
            <h1 style={{ fontSize: '2.5rem', margin: 0 }}>PolicyEngine VATLab</h1>
          </motion.div>
          
          {/* Conditional content based on whether parameters have changed */}
          {isBaseline ? (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                minHeight: '400px',
                padding: '3rem'
              }}
            >
              <div style={{
                backgroundColor: 'var(--blue-98)',
                border: '2px solid var(--blue-95)',
                borderRadius: '12px',
                padding: '3rem',
                maxWidth: '600px',
                textAlign: 'center'
              }}>
                <h2 style={{ color: 'var(--darkest-blue)', marginBottom: '1.5rem' }}>Define Your VAT Reform</h2>
                <p style={{ fontSize: '1.1rem', lineHeight: 1.6, marginBottom: '2rem', color: 'var(--dark-gray)' }}>
                  Use the sidebar controls to modify VAT policy parameters and see how your reforms would impact UK businesses and revenue.
                </p>
                <div style={{ 
                  backgroundColor: 'var(--white)', 
                  padding: '1.5rem', 
                  borderRadius: '8px',
                  border: '1px solid var(--medium-dark-gray)',
                  textAlign: 'left'
                }}>
                  <h3 style={{ fontSize: '1rem', marginBottom: '1rem', color: 'var(--darkest-blue)' }}>Available Parameters:</h3>
                  <ul style={{ margin: 0, paddingLeft: '1.5rem', fontSize: '0.95rem', lineHeight: 1.8, color: 'var(--dark-gray)' }}>
                    <li>Registration threshold (currently £90,000)</li>
                    <li>VAT rates by sector type</li>
                    <li>Graduated threshold options</li>
                    <li>Implementation year and elasticity</li>
                  </ul>
                </div>
                <div style={{ marginTop: '2rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
                  <span style={{ fontSize: '1.5rem' }}>👈</span>
                  <span style={{ fontSize: '1rem', color: 'var(--gray)' }}>Adjust parameters in the sidebar to begin</span>
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
                  background: 'linear-gradient(135deg, var(--blue-primary) 0%, var(--blue-pressed) 100%)',
                  color: 'var(--white)',
                  marginBottom: '2rem',
                  border: 'none',
                  boxShadow: '0 10px 30px rgba(0,0,0,0.1)'
                }}
              >
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '2rem', padding: '1.5rem' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2.2rem', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                      {formatPercent(revenueChangePercent)}
                    </div>
                    <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Revenue Change</div>
                  </div>
                  
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2.2rem', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                      {businessImpact.winners.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Winners</div>
                  </div>
                  
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ fontSize: '2.2rem', fontWeight: 'bold', marginBottom: '0.25rem' }}>
                      {businessImpact.losers.toFixed(1)}%
                    </div>
                    <div style={{ fontSize: '0.9rem', opacity: 0.9 }}>Losers</div>
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
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
              <h3 style={{ margin: 0 }}>Business Impact Distribution</h3>
              <select
                value={breakdownType}
                onChange={(e) => setBreakdownType(e.target.value)}
                style={{
                  padding: '0.5rem 1rem',
                  borderRadius: '6px',
                  border: '2px solid var(--medium-light-gray)',
                  fontSize: '0.9rem',
                  cursor: 'pointer',
                  backgroundColor: 'var(--white)'
                }}
              >
                <option value="sector">By Sector</option>
                <option value="size">By Employee Count</option>
              </select>
            </div>
            
            <div style={{ marginBottom: '2rem' }}>
              <div style={{ 
                display: 'flex', 
                height: '60px', 
                borderRadius: '8px', 
                overflow: 'hidden',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <div style={{ 
                  width: `${businessImpact.losers}%`, 
                  backgroundColor: 'var(--gray)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontWeight: 'bold',
                  fontSize: '1.1rem'
                }}>
                  {businessImpact.losers > 10 && `${businessImpact.losers.toFixed(1)}%`}
                </div>
                <div style={{ 
                  width: `${businessImpact.winners}%`, 
                  backgroundColor: 'var(--teal-accent)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontWeight: 'bold',
                  fontSize: '1.1rem'
                }}>
                  {businessImpact.winners > 10 && `${businessImpact.winners.toFixed(1)}%`}
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: '16px', height: '16px', backgroundColor: 'var(--gray)', borderRadius: '4px' }}></div>
                  <span style={{ fontSize: '0.9rem' }}>Businesses losing revenue</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: '16px', height: '16px', backgroundColor: 'var(--teal-accent)', borderRadius: '4px' }}></div>
                  <span style={{ fontSize: '0.9rem' }}>Businesses gaining revenue</span>
                </div>
              </div>
            </div>

            {/* Breakdown Details */}
            {breakdownType === 'sector' && (
              <div>
                {(() => {
                  const sectorData = calculateSectorBreakdown(filters);
                  return Object.entries(sectorData).map(([sector, data]) => (
                    <div key={sector} style={{ marginBottom: '1rem' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                        <span style={{ fontSize: '0.9rem' }}>{sector}</span>
                        <span style={{ 
                          fontSize: '0.9rem', 
                          color: data.winners > 50 ? 'var(--teal-accent)' : 'var(--gray)', 
                          fontWeight: 'bold' 
                        }}>
                          {data.winners > 50 
                            ? `${data.winners.toFixed(0)}% winners` 
                            : `${data.losers.toFixed(0)}% losers`}
                        </span>
                      </div>
                      <div style={{ 
                        height: '8px', 
                        backgroundColor: 'var(--fog-gray)', 
                        borderRadius: '4px', 
                        overflow: 'hidden',
                        display: 'flex',
                        flexDirection: data.winners > 50 ? 'row' : 'row-reverse'
                      }}>
                        <div style={{ 
                          width: `${data.winners > 50 ? data.winners : data.losers}%`, 
                          height: '100%', 
                          backgroundColor: data.winners > 50 ? 'var(--teal-accent)' : 'var(--gray)' 
                        }}></div>
                      </div>
                    </div>
                  ));
                })()}
              </div>
            )}
            
            {breakdownType === 'size' && (
              <div>
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.9rem' }}>1-9 employees</span>
                    <span style={{ fontSize: '0.9rem', color: 'var(--teal-accent)', fontWeight: 'bold' }}>78% winners</span>
                  </div>
                  <div style={{ height: '8px', backgroundColor: 'var(--fog-gray)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ width: '78%', height: '100%', backgroundColor: 'var(--teal-accent)' }}></div>
                  </div>
                </div>
                
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.9rem' }}>10-49 employees</span>
                    <span style={{ fontSize: '0.9rem', color: 'var(--teal-accent)', fontWeight: 'bold' }}>52% winners</span>
                  </div>
                  <div style={{ height: '8px', backgroundColor: 'var(--fog-gray)', borderRadius: '4px', overflow: 'hidden' }}>
                    <div style={{ width: '52%', height: '100%', backgroundColor: 'var(--teal-accent)' }}></div>
                  </div>
                </div>
                
                <div style={{ marginBottom: '1rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                    <span style={{ fontSize: '0.9rem' }}>50+ employees</span>
                    <span style={{ fontSize: '0.9rem', color: 'var(--gray)', fontWeight: 'bold' }}>65% losers</span>
                  </div>
                  <div style={{ height: '8px', backgroundColor: 'var(--fog-gray)', borderRadius: '4px', overflow: 'hidden', display: 'flex', flexDirection: 'row-reverse' }}>
                    <div style={{ width: '65%', height: '100%', backgroundColor: 'var(--gray)' }}></div>
                  </div>
                </div>
              </div>
            )}
          </motion.div>
            </>
          )}

          <div className="helper-text" style={{ marginTop: '2rem' }}>
            <span className="info-icon" title="About this analysis">ⓘ</span>
            <span>
              This dashboard is a demo created with fake data. It shows PolicyEngine's analysis of VAT reform options for the UK.
              Adjust parameters in the sidebar to see how different policies would impact revenue and businesses.
            </span>
          </div>
        </div>
      </div>
    </Layout>
  );
}