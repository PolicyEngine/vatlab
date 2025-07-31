/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  transpilePackages: ["react-plotly.js", "plotly.js"],
  output: "export",
  basePath: process.env.NEXT_PUBLIC_BASE_PATH || "",
  images: {
    unoptimized: true,
  },
  webpack: (config) => {
    config.resolve.alias = {
      ...config.resolve.alias,
      "plotly.js/dist/plotly": "plotly.js/dist/plotly.min.js",
    };
    return config;
  },
};

module.exports = nextConfig;
