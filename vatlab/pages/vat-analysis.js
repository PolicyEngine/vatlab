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
  const [loading, setLoading] = useState(false);

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
      winners = 48.5;
      losers = 6.2;
    } else if (thresholdIncrease || rateDecrease) {
      winners = 32.8;
      losers = 11.4;
    } else if (params.threshold < 90000 || avgRate > 20) {
      winners = 8.9;
      losers = 37.6;
    } else {
      winners = 15.0;
      losers = 15.0;
    }
    
    return { winners, losers, neutral: 100 - winners - losers };
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
    setLoading(true);
    setTimeout(() => {
      setAnalysisResults(newFilters);
      setFilters(newFilters);
      setLoading(false);
    }, 500);
  };

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
        <VATAnalysisSidebar 
          onFiltersChange={handleFiltersChange} 
          initialFilters={filters} 
          loading={loading} 
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
                <div style={{ textAlign: 'center', padding: '2rem' }}>
                  <h2 style={{ fontSize: '1.5rem', marginBottom: '2rem', opacity: 0.9 }}>
                    Your VAT Reform Impact
                  </h2>
              
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '3rem' }}>
                <div>
                  <div style={{ fontSize: '3rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                    {formatCurrency(currentRevenue)}
                  </div>
                  <div style={{ fontSize: '1rem', opacity: 0.9 }}>Total Revenue</div>
                  {!isBaseline && (
                    <div style={{ fontSize: '1.2rem', marginTop: '0.5rem', color: revenueChange >= 0 ? '#86efac' : '#fca5a5' }}>
                      {formatPercent(revenueChangePercent)}
                    </div>
                  )}
                </div>
                
                <div>
                  <div style={{ fontSize: '3rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                    2.3m
                  </div>
                  <div style={{ fontSize: '1rem', opacity: 0.9 }}>Businesses Affected</div>
                </div>
                
                <div>
                  <div style={{ fontSize: '3rem', fontWeight: 'bold', marginBottom: '0.5rem' }}>
                    {businessImpact.winners.toFixed(1)}%
                  </div>
                  <div style={{ fontSize: '1rem', opacity: 0.9 }}>Winners</div>
                  {!isBaseline && (
                    <div style={{ fontSize: '0.9rem', marginTop: '0.5rem', opacity: 0.8 }}>
                      {businessImpact.losers.toFixed(1)}% losers
                    </div>
                  )}
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
            <h3 style={{ marginTop: 0, marginBottom: '1.5rem' }}>Business Impact Distribution</h3>
            
            <div style={{ marginBottom: '2rem' }}>
              <div style={{ 
                display: 'flex', 
                height: '60px', 
                borderRadius: '8px', 
                overflow: 'hidden',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
              }}>
                <div style={{ 
                  width: `${businessImpact.winners}%`, 
                  backgroundColor: '#22c55e',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontWeight: 'bold',
                  fontSize: '1.1rem'
                }}>
                  {businessImpact.winners > 10 && `${businessImpact.winners.toFixed(1)}%`}
                </div>
                <div style={{ 
                  width: `${businessImpact.neutral}%`, 
                  backgroundColor: '#e5e7eb',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: '#6b7280',
                  fontWeight: 'bold',
                  fontSize: '1.1rem'
                }}>
                  {businessImpact.neutral > 10 && `${businessImpact.neutral.toFixed(1)}%`}
                </div>
                <div style={{ 
                  width: `${businessImpact.losers}%`, 
                  backgroundColor: '#ef4444',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'white',
                  fontWeight: 'bold',
                  fontSize: '1.1rem'
                }}>
                  {businessImpact.losers > 10 && `${businessImpact.losers.toFixed(1)}%`}
                </div>
              </div>
              
              <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '1rem' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: '16px', height: '16px', backgroundColor: '#22c55e', borderRadius: '4px' }}></div>
                  <span style={{ fontSize: '0.9rem' }}>Winners (gain &gt;5%)</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: '16px', height: '16px', backgroundColor: '#e5e7eb', borderRadius: '4px' }}></div>
                  <span style={{ fontSize: '0.9rem' }}>Minimal impact (±5%)</span>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <div style={{ width: '16px', height: '16px', backgroundColor: '#ef4444', borderRadius: '4px' }}></div>
                  <span style={{ fontSize: '0.9rem' }}>Losers (lose &gt;5%)</span>
                </div>
              </div>
            </div>

            {!isBaseline && (
              <div style={{ 
                backgroundColor: '#f3f4f6', 
                padding: '1rem', 
                borderRadius: '8px',
                fontSize: '0.9rem',
                lineHeight: 1.6
              }}>
                <strong>Key Insights:</strong>
                <ul style={{ margin: '0.5rem 0 0 0', paddingLeft: '1.5rem' }}>
                  {filters.threshold > 90000 && (
                    <li>Higher threshold reduces VAT burden on small businesses</li>
                  )}
                  {filters.threshold < 90000 && (
                    <li>Lower threshold brings more businesses into VAT system</li>
                  )}
                  {((filters.fullRateLaborIntensive + filters.fullRateNonLaborIntensive) / 2) > 20 && (
                    <li>Higher rates increase revenue but may impact business growth</li>
                  )}
                  {((filters.fullRateLaborIntensive + filters.fullRateNonLaborIntensive) / 2) < 20 && (
                    <li>Lower rates reduce business costs but decrease revenue</li>
                  )}
                </ul>
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