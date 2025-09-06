import React, { useState } from 'react';

/**
 * Form component handling raw X12 input and submission to summarization endpoint.
 */
export default function X12Form({ onSubmit, loading }) {
  const [value, setValue] = useState('');
  function handleSubmit(e) { e.preventDefault(); if (!value.trim()) return; onSubmit(value); }
  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-5">
      <label className="flex flex-col gap-2">
        <span className="text-xs font-semibold uppercase tracking-wider text-emerald-800/70">X12 EDI Content</span>
        <textarea
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="Paste raw X12 here (segments separated by ~)"
          rows={14}
          className="w-full resize-y rounded-xl p-4 text-sm leading-relaxed bg-white/60 text-emerald-900 placeholder:text-emerald-700/70 focus:outline-none focus:ring-2 focus:ring-emerald-400/40 border border-emerald-700/10 shadow-inner"
        />
      </label>
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
        <p className="text-xs text-emerald-900/70 leading-snug max-w-prose order-2 md:order-1">
          Generates a human-readable summary plus structured transparency sections for codes, actions & parsed data.
        </p>
        <button
          type="submit"
          disabled={loading}
          className="order-1 md:order-2 px-6 py-3 rounded-xl bg-gradient-to-r from-emerald-500 via-teal-500 to-lime-400 text-white font-semibold shadow-md shadow-emerald-300/40 hover:shadow-emerald-500/40 hover:from-emerald-400 hover:via-teal-400 hover:to-lime-300 active:scale-[.98] transition disabled:opacity-50 disabled:cursor-not-allowed disabled:grayscale self-end md:self-auto"
        >
          {loading ? 'Summarizing...' : 'Summarize'}
        </button>
      </div>
    </form>
  );
}
