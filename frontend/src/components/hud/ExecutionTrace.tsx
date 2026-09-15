import React, { useState } from 'react';
import { Terminal, ChevronDown, ChevronUp, Clock, Cpu } from 'lucide-react';
import { ExecutionStage } from '../../types';
import { sanitizeModelText } from '../../utils/sanitize';

interface ExecutionTraceProps {
  stages: ExecutionStage[];
  totalTimeMs?: number;
}

export const ExecutionTrace: React.FC<ExecutionTraceProps> = ({ stages, totalTimeMs }) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);

  if (!stages || stages.length === 0) return null;

  return (
    <div className="bg-sat-darker border border-sat-border rounded text-xs font-mono overflow-hidden">
      {/* Header Accordion Toggle */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between p-2.5 bg-sat-panel hover:bg-sat-surface transition text-left"
      >
        <div className="flex items-center space-x-2">
          <Terminal className="w-3.5 h-3.5 text-sat-accent" />
          <span className="font-bold text-sat-text">EXECUTION TRACE</span>
          <span className="text-[10px] text-sat-muted">({stages.length} stages profiled)</span>
        </div>

        <div className="flex items-center space-x-3 text-[11px]">
          {totalTimeMs && (
            <span className="text-sat-accent flex items-center space-x-1 font-bold">
              <Clock className="w-3 h-3" />
              <span>{totalTimeMs.toFixed(1)} ms</span>
            </span>
          )}
          {isOpen ? <ChevronUp className="w-4 h-4 text-sat-muted" /> : <ChevronDown className="w-4 h-4 text-sat-muted" />}
        </div>
      </button>

      {/* Expanded Stage List */}
      {isOpen && (
        <div className="p-3 bg-sat-darker border-t border-sat-border space-y-2 max-h-56 overflow-y-auto font-mono text-[11px]">
          {stages.map((stg, idx) => (
            <div key={idx} className="flex items-start space-x-2.5 p-1.5 rounded hover:bg-sat-surface/40 transition">
              <span className="text-sat-muted font-bold min-w-[55px]">[{stg.timestamp}]</span>
              <div className="flex-1 space-y-0.5">
                <div className="flex items-center space-x-2">
                  <span className="text-sat-accent font-bold">{sanitizeModelText(stg.stage)}</span>
                  {stg.duration_ms !== undefined && stg.duration_ms !== null && (
                    <span className="text-[10px] bg-sat-panel px-1.5 py-0.2 rounded border border-sat-border text-sat-muted">
                      +{stg.duration_ms} ms
                    </span>
                  )}
                  {stg.memory_mb !== undefined && stg.memory_mb !== null && (
                    <span className="text-[10px] bg-sat-panel px-1.5 py-0.2 rounded border border-sat-border text-sat-cyan">
                      {stg.memory_mb} MB
                    </span>
                  )}
                </div>
                <p className="text-sat-text/90 text-[11px] font-sans">{sanitizeModelText(stg.message)}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
