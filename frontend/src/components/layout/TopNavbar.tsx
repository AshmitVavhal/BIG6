import React from 'react';
import { Cpu, HardDrive, RotateCcw, FileText, BarChart2, Satellite, Zap } from 'lucide-react';
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
  const isCuda = systemStatus?.device.toLowerCase() === 'cuda';
  const vramUsed = systemStatus?.vram_used_gb ?? 0;
  const vramTotal = systemStatus?.vram_total_gb ?? 0;
  const vramPct = vramTotal > 0 ? Math.round((vramUsed / vramTotal) * 100) : 0;

  return (
    <header className="h-14 bg-sat-panel border-b border-sat-border flex items-center justify-between px-4 z-30 select-none">
      {/* Left: Branding & Mission Badge */}
      <div className="flex items-center space-x-3">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded bg-sat-accent/15 border border-sat-accent flex items-center justify-center text-sat-accent">
            <Satellite className="w-4 h-4" />
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="font-bold text-sm tracking-wider text-sat-text font-mono">SATQUERY AI</span>
              <span className="text-[10px] bg-sat-surface px-1.5 py-0.5 rounded text-sat-muted border border-sat-border font-mono">
                v1.0.0
              </span>
            </div>
            <div className="text-[11px] text-sat-muted font-mono flex items-center space-x-1.5">
              <span className="text-sat-accent font-semibold">ISRO / SAC 26167</span>
              <span>•</span>
              <span>EARTH OBSERVATION WORKSTATION</span>
            </div>
          </div>
        </div>

        {/* Live Status Indicator */}
        <div className="hidden sm:flex items-center space-x-1.5 px-2.5 py-1 bg-sat-surface rounded-full border border-sat-border text-[11px] font-mono">
          <span className={`w-2 h-2 rounded-full ${isProcessing ? 'bg-sat-warning animate-ping' : 'bg-sat-accent'}`} />
          <span className={isProcessing ? 'text-sat-warning font-semibold' : 'text-sat-accent font-semibold'}>
            {isProcessing ? 'PROCESSING INFERENCE' : 'ONLINE'}
          </span>
          <span className="text-sat-muted">|</span>
          <span className="text-sat-text text-[10px] uppercase font-semibold">
            {isCuda ? `CUDA GPU [${systemStatus?.gpu_name || 'NVIDIA'}]` : 'CPU MODE'}
          </span>
          <span className="text-sat-muted">|</span>
          <span className="text-[10px] font-semibold flex items-center space-x-1 text-sat-accent">
            <span>●</span>
            <span>GEOCHAT VLM</span>
          </span>
          <span className="text-sat-muted">|</span>
          <span className={`text-[10px] font-semibold flex items-center space-x-1 ${
            systemStatus?.gemini_configured ? 'text-sat-cyan' : 'text-sat-muted'
          }`}>
            <span>{systemStatus?.gemini_configured ? '●' : '○'}</span>
            <span>{systemStatus?.gemini_configured ? 'GEMINI REASONING' : 'GEMINI OFFLINE'}</span>
          </span>
        </div>
      </div>

      {/* Center: Quick Benchmark trigger */}
      <div className="hidden lg:flex items-center space-x-2">
        <button
          onClick={onOpenBenchmarks}
          className="flex items-center space-x-1.5 px-3 py-1.5 bg-sat-surface hover:bg-sat-surface/80 border border-sat-border hover:border-sat-borderLight rounded text-xs font-mono text-sat-text transition"
        >
          <BarChart2 className="w-3.5 h-3.5 text-sat-accent" />
          <span>BENCHMARKS</span>
        </button>
      </div>

      {/* Right: Hardware telemetry gauges & actions */}
      <div className="flex items-center space-x-3">
        {/* VRAM / RAM / CPU Telemetry */}
        <div className="hidden md:flex items-center space-x-3 bg-sat-darker px-3 py-1.5 rounded border border-sat-border font-mono text-[11px]">
          {isCuda && vramTotal > 0 && (
            <div className="flex items-center space-x-1.5" title={`VRAM: ${vramUsed} GB / ${vramTotal} GB`}>
              <Zap className="w-3.5 h-3.5 text-sat-cyan" />
              <span className="text-sat-muted">VRAM</span>
              <span className="text-sat-text font-bold">{vramPct}%</span>
            </div>
          )}

          <div className="flex items-center space-x-1.5" title="System RAM Usage">
            <HardDrive className="w-3.5 h-3.5 text-sat-accent" />
            <span className="text-sat-muted">RAM</span>
            <span className="text-sat-text font-bold">{systemStatus?.ram_percent ?? 0}%</span>
          </div>

          <div className="flex items-center space-x-1.5" title="CPU Utilization">
            <Cpu className="w-3.5 h-3.5 text-sat-warning" />
            <span className="text-sat-muted">CPU</span>
            <span className="text-sat-text font-bold">{systemStatus?.cpu_percent ?? 0}%</span>
          </div>
        </div>

        {/* Reset Button */}
        <button
          onClick={onReset}
          className="flex items-center space-x-1 px-2.5 py-1.5 bg-sat-surface hover:bg-sat-surface/80 border border-sat-border hover:border-sat-borderLight rounded text-xs font-mono text-sat-muted hover:text-sat-text transition"
          title="Reset Workspace & Clear Cache"
        >
          <RotateCcw className="w-3.5 h-3.5" />
          <span className="hidden sm:inline">Reset</span>
        </button>

        {/* Export Report */}
        <button
          onClick={onOpenExport}
          className="flex items-center space-x-1.5 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white font-medium rounded text-xs font-mono shadow transition"
          title="Generate and download intelligence PDF report"
        >
          <FileText className="w-3.5 h-3.5" />
          <span>Export Report</span>
        </button>
      </div>
    </header>
  );
};
