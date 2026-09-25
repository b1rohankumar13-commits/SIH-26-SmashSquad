import type { NextConfig } from "next";

// The FastAPI backend (api/main.py, port 8000) serves the prediction exports; proxy it so the
// browser talks to one origin and no CORS setup is needed.
const API_ORIGIN = process.env.BUSTSENTINEL_API ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API_ORIGIN}/api/:path*` }];
  },
};

export default nextConfig;
