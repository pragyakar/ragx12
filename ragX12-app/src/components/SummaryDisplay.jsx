import React, { useState } from 'react';

/**
 * Display component for summarization result plus collapsible technical transparency.
 */
export default function SummaryDisplay({ result, error, loading }) {
  const [open, setOpen] = useState({ parsed: false, retrieval: false, prompt: false, codes: false, actions: true });
  if (error) return <div className="glass rounded-xl p-4 border border-rose-400/50 bg-rose-100/60 text-rose-700 text-sm shadow-lg">{error}</div>;
  if (loading) return <div className="glass rounded-xl p-8 text-center text-emerald-800/70 animate-pulse">Generating summary...</div>;
  if (!result) return <div className="glass rounded-xl p-8 text-center text-xs text-emerald-700/60">No summary yet. Submit X12 to begin.</div>;
  const { summary, data, possible_actions, code_meanings } = result;
  return (
    <div className="flex flex-col gap-8">
      <section className="glass rounded-2xl p-6 shadow-lg shadow-emerald-100/50 ring-1 ring-emerald-700/10">
        <h2 className="text-xl font-semibold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-emerald-600 via-teal-600 to-lime-600">Summary</h2>
        <pre className="whitespace-pre-wrap text-sm leading-relaxed font-medium text-emerald-900/90">
          {summary}
        </pre>
      </section>
      <Collapsible title="Possible Actions" open={open.actions} onToggle={() => setOpen(o => ({ ...o, actions: !o.actions }))}>
        <ul className="list-disc pl-5 text-sm space-y-1 text-emerald-900/90">{(possible_actions || []).map((a,i) => <li key={i}>{a}</li>)}</ul>
      </Collapsible>
      <Collapsible title="Structured Data (parsed summary sections)" open={open.parsed} onToggle={() => setOpen(o => ({ ...o, parsed: !o.parsed }))}>
        <pre className="text-[11px] whitespace-pre-wrap text-emerald-900/80">{JSON.stringify(data, null, 2)}</pre>
      </Collapsible>
      <Collapsible title="Code Meanings" open={open.codes} onToggle={() => setOpen(o => ({ ...o, codes: !o.codes }))}>
        <pre className="text-[11px] whitespace-pre-wrap text-emerald-900/80">{JSON.stringify(code_meanings, null, 2)}</pre>
      </Collapsible>
    </div>
  );
}

function Collapsible({ title, open, onToggle, children }) {
  return (
    <div className="glass rounded-2xl p-5 shadow-md shadow-emerald-100/50 ring-1 ring-emerald-700/10">
      <button type="button" onClick={onToggle} className="flex items-center justify-between w-full text-left">
        <span className="font-semibold text-[11px] tracking-wider uppercase text-emerald-700/70">{title}</span>
        <span className="text-xs text-emerald-600 font-bold w-6 text-right">{open ? '−' : '+'}</span>
      </button>
      {open && <div className="mt-4">{children}</div>}
    </div>
  );
}
