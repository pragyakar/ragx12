import React, { useEffect, useState, useRef } from 'react';
import Header from './components/Header.jsx';
import X12Form from './components/X12Form.jsx';
import SummaryDisplay from './components/SummaryDisplay.jsx';
import './index.css';

// API base URL is now configurable via Vite env variable.
// Define VITE_API_BASE in a .env (not committed) or .env.local file at project root.
// Falls back to placeholder so the UI still renders without config.
const API_BASE = import.meta.env.VITE_API_BASE || 'https://api.example.com';

export default function App() {
  const [apiStatus, setApiStatus] = useState('...');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const summaryRef = useRef(null);

  // Health check on mount
  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/health`).then(r => r.json()).then(data => {
      if (!active) return;
      setApiStatus(data.status || 'OK');
    }).catch(() => active && setApiStatus('Offline'));
    return () => { active = false; };
  }, []);

  async function handleSummarize(x12) {
    setLoading(true);
    setError('');
    setResult(null);
    try {
      const resp = await fetch(`${API_BASE}/summarize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ x12, top_k: 6, include_prompt: true })
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${resp.status})`);
      }
      const data = await resp.json();
      setResult(data);
    } catch (e) {
      setError(e.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  }

  // Scroll to summary when a new result arrives
  useEffect(() => {
    if (result && summaryRef.current) {
      summaryRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result]);

  return (
    <div className="relative min-h-screen w-full overflow-x-hidden bg-gradient-to-br from-emerald-50 via-teal-50 to-lime-50 text-slate-700">
      {/* Soft ambient blobs / gradient overlays (reduced blue, lighter palette) */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden">
        <div className="absolute -top-40 -left-40 h-96 w-96 rounded-full bg-teal-300/20 blur-3xl" />
        <div className="absolute top-1/3 -right-44 h-[32rem] w-[32rem] rounded-full bg-emerald-300/20 blur-3xl" />
        <div className="absolute bottom-0 left-1/2 -translate-x-1/2 h-80 w-80 rounded-full bg-lime-300/20 blur-3xl" />
        <div className="absolute inset-0 backdrop-brightness-105" />
      </div>
      <div className="relative max-w-5xl mx-auto px-4 py-10 flex flex-col gap-10">
        <Header apiStatus={apiStatus} />
        <main className="flex flex-col gap-10">
          <section className="glass rounded-2xl p-6 shadow-lg shadow-emerald-100/40 ring-1 ring-teal-700/5 backdrop-saturate-150">
            <X12Form onSubmit={handleSummarize} loading={loading} />
          </section>
          <section ref={summaryRef} id="summary" className="scroll-mt-24">
            <SummaryDisplay result={result} loading={loading} error={error} />
          </section>
        </main>
        <footer className="text-center text-xs text-slate-500 py-8">
          <p className="font-medium tracking-wide">Built with ❤️ for healthcare operations.</p>
        </footer>
      </div>
    </div>
  );
}
