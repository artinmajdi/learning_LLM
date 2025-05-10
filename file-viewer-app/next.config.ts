import type { NextConfig } from "next";
import path from "path";

const nextConfig: NextConfig = {
  // Enable React Server Components
  reactStrictMode: true,
  
  // Configure webpack to handle external components
  webpack: (config, { isServer }) => {
    // Add support for importing components from the external directory
    config.resolve.alias = {
      ...config.resolve.alias,
      '@/external-components': path.join(__dirname, 'src', 'external-components'),
    };
    
    return config;
  },
  
  // Configure experimental features if needed
  experimental: {
    // Add experimental features here if needed in the future
  },
};

export default nextConfig;
