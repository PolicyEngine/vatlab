import { useState } from 'react';
import { motion } from 'framer-motion';

export default function Tabs({ tabs, children }) {
  const [activeTab, setActiveTab] = useState(0);

  return (
    <div className="tab-container">
      <div className="tabs">
        {tabs.map((tab, index) => (
          <div
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            {tab.label}
          </div>
        ))}
      </div>
      
      {tabs[activeTab]?.description && (
        <motion.div 
          className="tab-description"
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          key={`desc-${activeTab}`}
        >
          {tabs[activeTab]?.description}
        </motion.div>
      )}
      
      {children.map((child, index) => (
        <div 
          key={index} 
          className={`tab-content ${activeTab === index ? 'active' : ''}`}
        >
          {child}
        </div>
      ))}
    </div>
  );
}