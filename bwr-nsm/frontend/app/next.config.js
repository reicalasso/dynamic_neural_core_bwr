/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  experimental: {
    // Disable static optimization to prevent hydration mismatch
    forceSwcTransforms: true,
  },
  compiler: {
    // Remove console.log in production
    removeConsole: process.env.NODE_ENV === 'production',
  },
  // Disable static optimization for pages with dynamic content
  staticPageGenerationTimeout: 1000,
}

module.exports = nextConfig
