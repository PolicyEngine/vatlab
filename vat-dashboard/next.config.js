/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  transpilePackages: ['react-plotly.js', 'plotly.js'],
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      'plotly.js/dist/plotly': 'plotly.js/dist/plotly.min.js',
    };
    return config;
  },
}

module.exports = nextConfig