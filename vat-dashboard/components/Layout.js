import Head from 'next/head';
import Link from 'next/link';
import { useRouter } from 'next/router';
import { motion } from 'framer-motion';

export default function Layout({ children }) {
  const router = useRouter();
  
  return (
    <>
      <Head>
        <title></title>
        <meta name="description" content="Analyze VAT policy reforms and their economic impacts using PolicyEngine microsimulation models" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
      
      <nav style={{
        background: '#2C6496',
        padding: '1rem 2rem',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <div style={{
          maxWidth: '1600px',
          margin: '0 auto',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
            <h2 style={{ 
              color: 'white', 
              margin: 0, 
              fontSize: '1.5rem',
              fontWeight: '600'
            }}>
              
            </h2>
          </div>
          <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', cursor: 'pointer' }}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7,10 12,15 17,10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
              </svg>
              <span style={{ color: 'white', fontSize: '0.75rem', marginTop: '0.25rem' }}>Results CSV</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', cursor: 'pointer' }}>
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
              </svg>
              <span style={{ color: 'white', fontSize: '0.75rem', marginTop: '0.25rem' }}>Profile</span>
            </div>
          </div>
        </div>
      </nav>
      
      <motion.main
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.2 }}
        style={{ position: 'relative' }}
      >
        {children}
      </motion.main>
      
      <footer>
        <div style={{ maxWidth: '1600px', margin: '0 auto', padding: '0 2rem', textAlign: 'center' }}>
          <p>Powered by PolicyEngine UK models &copy; {new Date().getFullYear()}</p>
        </div>
      </footer>
    </>
  );
}