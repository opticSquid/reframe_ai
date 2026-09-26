"use client";

import { useEffect, useState } from "react";

export default function Home() {
  const [backendStatus, setBackendStatus] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function checkBackend() {
    setLoading(true);
    try {
      const backendUrl =
        process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:5000";
      const res = await fetch(`${backendUrl}/health`);
      if (res.ok) {
        const data = await res.json();
        setBackendStatus(`Backend reachable: ${JSON.stringify(data)}`);
      } else {
        setBackendStatus(`Backend responded ${res.status}`);
      }
    } catch (err) {
      setBackendStatus(`Cannot reach backend: ${err}`);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-zinc-50 dark:bg-black flex flex-col items-center justify-center p-8">
      <div className="max-w-xl w-full text-center space-y-8">
        <h1 className="text-4xl font-bold text-black dark:text-white">
          ReframeAI
        </h1>
        <p className="text-zinc-600 dark:text-zinc-400">
          Subject-aware media pipeline
        </p>

        <button
          onClick={checkBackend}
          disabled={loading}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Checking..." : "Check Backend Health"}
        </button>

        {backendStatus && (
          <pre className="text-left bg-zinc-100 dark:bg-zinc-900 rounded-lg p-4 text-sm">
            {backendStatus}
          </pre>
        )}
      </div>
    </main>
  );
}
