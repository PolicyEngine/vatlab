import { useState } from 'react';
import { motion } from 'framer-motion';

export default function VATAnalysisSidebar({ onFiltersChange, initialFilters, loading }) {
  const [filters, setFilters] = useState(initialFilters || {
    threshold: 90000,
    graduatedEndThreshold: '',
    fullRateLaborIntensive: 20,
    fullRateNonLaborIntensive: 20,
    year: 2026,
    elasticity: -0.01
  });

  const years = Array.from({ length: 21 }, (_, i) => 2020 + i); // 2020-2040

  const handleFilterChange = (key, value) => {
    const newFilters = { ...filters, [key]: value };
    setFilters(newFilters);
  };

  const handleAnalyse = () => {
    onFiltersChange(filters);
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
      
      <div className="sidebar-card" style={{ overflow: 'auto', maxHeight: 'calc(100vh - 120px)' }}>
        <h3 className="sidebar-title" style={{ textAlign: 'center' }}>VAT Policy Reform</h3>
        <p style={{ marginBottom: '1.5rem', fontSize: '0.95rem', lineHeight: 1.5 }}>
          Configure your VAT reform to compare against the current UK policy (£90k threshold, 20% rate for all sectors).
        </p>
        
        {/* Current UK Policy Info Box */}
        <div style={{ 
          backgroundColor: '#f0f9ff', 
          padding: '1rem', 
          borderRadius: '8px', 
          marginBottom: '1.5rem',
          border: '1px solid #3b82f6',
          fontSize: '0.85rem'
        }}>
          <strong style={{ color: '#1e40af' }}>Baseline: Current UK VAT Policy</strong>
          <div style={{ marginTop: '0.5rem', color: '#374151' }}>
            • £90,000 registration threshold<br/>
            • 20% standard VAT rate for all sectors<br/>
            • No graduated threshold system
          </div>
        </div>

        <div className="form-group">
          <label>Year Applies:</label>
          <div style={{ fontSize: '0.8rem', color: '#666', marginBottom: '0.5rem' }}>
            year when the policy reform takes effect
          </div>
          <select
            value={filters.year}
            onChange={(e) => handleFilterChange('year', parseInt(e.target.value))}
            className="select-dropdown"
          >
            {years.map(year => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>
        </div>

        <div className="form-group">
          <label>Registration Threshold:</label>
          <div style={{ fontSize: '0.8rem', color: '#666', marginBottom: '0.5rem' }}>
            annual turnover above which businesses must register for vat
          </div>
          <input
            type="number"
            value={filters.threshold}
            onChange={(e) => handleFilterChange('threshold', parseInt(e.target.value) || 0)}
            className="select-dropdown"
            placeholder="90000"
            min="0"
            step="1000"
          />
        </div>

        <div className="form-group">
          <label>Graduated End Threshold (optional):</label>
          <div style={{ fontSize: '0.8rem', color: '#666', marginBottom: '0.5rem' }}>
            upper threshold for graduated vat system with smooth transition
          </div>
          <input
            type="number"
            value={filters.graduatedEndThreshold}
            onChange={(e) => handleFilterChange('graduatedEndThreshold', e.target.value ? parseInt(e.target.value) : '')}
            className="select-dropdown"
            placeholder="Leave empty if not applicable"
            min="0"
            step="1000"
          />
        </div>


        <div className="form-group">
          <label>Labor-Intensive Rate (%):</label>
          <div style={{ fontSize: '0.8rem', color: '#666', marginBottom: '0.5rem' }}>
            vat rate applied to labor-intensive services
          </div>
          <input
            type="number"
            value={filters.fullRateLaborIntensive}
            onChange={(e) => handleFilterChange('fullRateLaborIntensive', parseFloat(e.target.value) || 0)}
            className="select-dropdown"
            placeholder="20"
            min="0"
            max="100"
            step="0.1"
          />
        </div>

        <div className="form-group">
          <label>Non-Labor Intensive Rate (%):</label>
          <div style={{ fontSize: '0.8rem', color: '#666', marginBottom: '0.5rem' }}>
            standard vat rate for all other business sectors
          </div>
          <input
            type="number"
            value={filters.fullRateNonLaborIntensive}
            onChange={(e) => handleFilterChange('fullRateNonLaborIntensive', parseFloat(e.target.value) || 0)}
            className="select-dropdown"
            placeholder="20"
            min="0"
            max="100"
            step="0.1"
          />
        </div>

        <div className="form-group">
          <label>Elasticity of firm turnover to VAT threshold:</label>
          <div style={{ fontSize: '0.8rem', color: '#666', marginBottom: '0.5rem' }}>
            how responsive firms are to changes in vat thresholds
          </div>
          <input
            type="number"
            value={filters.elasticity}
            onChange={(e) => handleFilterChange('elasticity', parseFloat(e.target.value) || 0)}
            className="select-dropdown"
            placeholder="-0.01"
            min="-5"
            max="5"
            step="0.1"
          />
        </div>

        <motion.button 
          onClick={handleAnalyse} 
          disabled={loading}
          whileHover={{ scale: 1.03 }}
          whileTap={{ scale: 0.98 }}
          className={!loading ? "pulse" : ""}
          style={{
            width: '100%',
            padding: '1rem 2rem',
            fontSize: '1.1rem',
            fontWeight: '600',
            textAlign: 'center',
            marginTop: '1.5rem'
          }}
        >
          {loading ? 'Loading...' : 'Analyse VAT Policy'}
        </motion.button>
      </div>
    </motion.div>
  );
}