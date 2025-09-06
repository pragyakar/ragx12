import React from 'react';

/**
 * Top application header with title and live API status indicator.
 */
export default function Header({ apiStatus }) {
  const statusColor = apiStatus === 'OK' ? 'bg-emerald-500' : apiStatus === '...' ? 'bg-amber-400 animate-pulse' : 'bg-rose-500';
  return (
    <header className="glass rounded-2xl px-6 py-5 flex flex-col gap-3 md:flex-row md:items-center md:justify-between shadow-lg shadow-emerald-100/40 ring-1 ring-emerald-700/10 backdrop-saturate-150">
      <div>
        <h1 className="text-3xl font-semibold tracking-tight flex items-center gap-2 bg-clip-text text-transparent bg-gradient-to-r from-emerald-600 via-teal-600 to-lime-600">
          <span>RagX12</span>
        </h1>
        <p className="text-sm text-emerald-900/70">AI-powered X12 healthcare EDI summary & transparency viewer</p>
      </div>
      <div className="flex items-center gap-2 text-sm text-emerald-900/80">
        <span className={`w-3 h-3 rounded-full ${statusColor} shadow`}></span>
        <span className="font-medium">API Status:</span>
        <span>{apiStatus || '...'}</span>
      </div>
    </header>
  );
}
