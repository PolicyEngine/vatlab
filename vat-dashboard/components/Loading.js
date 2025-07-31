import React from 'react';
import { motion } from 'framer-motion';

export default function Loading() {
  return (
    <motion.div 
      className="loading-container"
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      style={{ 
        display: 'flex', 
        flexDirection: 'column', 
        alignItems: 'center', 
        justifyContent: 'center',
        padding: '3rem'
      }}
    >
      <div className="loading">
        <div></div>
        <div></div>
        <div></div>
        <div></div>
      </div>
      <motion.p
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        style={{ 
          marginTop: '1.5rem', 
          color: 'var(--blue-primary)',
          fontWeight: 500
        }}
      >
        Calculating policy impact...
      </motion.p>
    </motion.div>
  );
}