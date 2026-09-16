import React from 'react';
import { RotateCcw, FileText, BarChart2, Satellite, Radio } from 'lucide-react';
import type { SystemStatus } from '../../types';

interface TopNavbarProps {
  systemStatus: SystemStatus | null;
  onReset: () => void;
  onOpenBenchmarks: () => void;
  onOpenExport: () => void;
  isProcessing: boolean;
}

export const TopNavbar: React.FC<TopNavbarProps> = ({
  systemStatus,
  onReset,
  onOpenBenchmarks,
  onOpenExport,
  isProcessing
}) => {
  const isCuda = Boolean(systemStatus && systemStatus.device?.toLowerCase() === 'cuda' && systemStatus.gpu_name);

  return (
    <header className="h-12 bg-sat-bg border-b border-sat-border flex items-center justify-between px-3 z-30 select-none">
      {/* Left Section: Branding & System State */}
      <div className="flex items-center space-x-2.5">
        {/* SatQuery AI Brand Pill */}
        <div className="flex items-center space-x-2 bg-sat-panel px-2.5 py-1 rounded-md border border-sat-border">
          <div className="w-5 h-5 rounded bg-sat-surface flex items-center justify-center text-sat-accent">
            <Satellite className="w-3.5 h-3.5" />
          </div>
          <span className="font-semibold text-xs tracking-tight text-white font-sans">
            SatQuery AI
          </span>
          <span className="text-[11px] text-sat-muted font-mono font-medium pl-1 border-l border-sat-border">
            ISRO SAC 26167
          </span>
        </div>

        {/* Live Status Pill */}
        <div className="flex items-center space-x-2 px-2.5 py-1 bg-sat-panel/80 rounded-md border border-sat-border text-[11px] font-mono">
          <span className="flex items-center space-x-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${isProcessing ? 'bg-sat-warning animate-ping' : 'bg-sat-accent'}`} />
            <span className={isProcessing ? 'text-sat-warning font-semibold' : 'text-sat-accent font-semibold'}>
              {isProcessing ? 'Processing' : 'Online'}
            </span>
          </span>

          <span className="text-sat-borderLight">·</span>

          <span className="text-sat-textSecondary text-[10px]">
            {systemStatus
              ? isCuda
                ? `CUDA GPU [${systemStatus.gpu_name}]`
                : 'CPU MODE (ACTIVE)'
              : 'INITIALIZING...'}
          </span>
        </div>
      </div>

      {/* Right Section: Actions & Telemetry */}
      <div className="flex items-center space-x-2">
        {/* Benchmarks Trigger */}
        <button
          onClick={onOpenBenchmarks}
          className="flex items-center space-x-1.5 px-2.5 py-1 bg-sat-panel hover:bg-sat-surface border border-sat-border hover:border-sat-borderLight rounded-md text-[11px] font-mono text-sat-textSecondary hover:text-white transition"
          title="Hardware benchmarks and latency telemetry"
        >
          <BarChart2 className="w-3.5 h-3.5 text-sat-muted" />
          <span className="hidden md:inline">Benchmarks</span>
        </button>

        {/* Reset Button */}
        <button
          onClick={onReset}
          className="flex items-center space-x-1.5 px-2.5 py-1 bg-sat-panel hover:bg-sat-surface border border-sat-border hover:border-sat-borderLight rounded-md text-[11px] font-mono text-sat-textSecondary hover:text-white transition"
          title="Reset Workspace & Clear Cache"
        >
          <RotateCcw className="w-3 h-3 text-sat-muted" />
          <span>Reset</span>
        </button>

        {/* Export Dossier / Report Button */}
        <button
          onClick={onOpenExport}
          className="flex items-center space-x-1.5 px-3 py-1 bg-white hover:bg-slate-100 text-slate-900 font-semibold rounded-md text-xs font-sans shadow-sm transition active:scale-95"
          title="Export satellite intelligence dossier"
        >
          <FileText className="w-3.5 h-3.5 text-slate-800" />
          <span>Export Dossier</span>
        </button>
      </div>
    </header>
  );
};

