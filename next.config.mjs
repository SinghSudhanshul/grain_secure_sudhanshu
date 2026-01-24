/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config) => {
    config.resolve.extensions = ['.js', '.jsx', '.ts', '.tsx'];
    return config;
  },
};

export default nextConfig;