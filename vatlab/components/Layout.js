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