/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  experimental: {
    typedRoutes: false,
  },
  // Studio is its own app under monorepo subdir; avoid implicit tracing into
  // the python repo above.
  outputFileTracingRoot: process.cwd(),
};

export default nextConfig;
