import React, { useEffect, useState } from 'react';
import { X, BarChart2, Zap, Cpu, HardDrive, RotateCw } from 'lucide-react';
import { BenchmarkRecord, ModelStatusItem } from '../../types';
import { fetchBenchmarks, fetchModelsStatus } from '../../services/api';

interface BenchmarkModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const BenchmarkModal: React.FC<BenchmarkModalProps> = ({ isOpen, onClose }) => {
  const [benchmarks, setBenchmarks] = useState<{ summary: Record<string, any>; recent_runs: BenchmarkRecord[] }>({
    summary: {},
    recent_runs: []
  });
  const [models, setModels] = useState<ModelStatusItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);

  const loadData = async () => {
    setLoading(true);
    try {
      const [bData, mData] = await Promise.all([fetchBenchmarks(), fetchModelsStatus()]);
      setBenchmarks(bData);
      setModels(mData.models);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      loadData();
    }
  }, [isOpen]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 z-50 select-none">
      <div className="bg-sat-panel border border-sat-border rounded-lg shadow-2xl max-w-3xl w-full max-h-[85vh] flex flex-col overflow-hidden font-mono">
        {/* Modal Header */}
        <div className="p-4 border-b border-sat-border flex items-center justify-between bg-sat-darker">
          <div className="flex items-center space-x-2.5">
            <BarChart2 className="w-5 h-5 text-sat-accent" />
            <div>
              <h2 className="text-sm font-bold text-sat-text tracking-wide">MODEL BENCHMARKS & HARDWARE TELEMETRY</h2>
              <span className="text-[11px] text-sat-muted">ISRO SAC-26167 In-Memory Performance Profiler</span>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={loadData}
              className="p-1.5 hover:bg-sat-surface rounded text-sat-muted hover:text-sat-text transition"
              title="Refresh Benchmarks"
            >
              <RotateCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 hover:bg-sat-surface rounded text-sat-muted hover:text-sat-text transition"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* Modal Scrollable Body */}
        <div className="p-4 overflow-y-auto space-y-4">
          {/* Active Model Status Summary */}
          <div>
            <h3 className="text-xs font-bold text-sat-muted mb-2 flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-sat-accent" />
              <span>ORCHESTRATED MODEL STATUS & VRAM FOOTPRINT</span>
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {models.map((m) => (
                <div key={m.key} className="bg-sat-darker p-2.5 rounded border border-sat-border space-y-1 text-xs">
                  <div className="flex items-center justify-between">
                    <span className="font-bold text-sat-text">{m.name}</span>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
                      m.loaded ? 'bg-sat-accent/20 text-sat-accent border border-sat-accent/40' : 'bg-sat-surface text-sat-muted'
                    }`}>
                      {m.loaded ? '● LOADED' : '○ IDLE'}
                    </span>
                  </div>
                  <p className="text-[11px] text-sat-muted font-sans leading-tight">{m.description}</p>
                  <div className="flex items-center justify-between text-[10px] text-sat-muted pt-1 border-t border-sat-border">
                    <span>DEVICE: <span className="text-sat-text uppercase font-bold">{m.device}</span></span>
                    <span>MEMORY: <span className="text-sat-cyan font-bold">{m.vram_mb} MB</span></span>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Model Latency Summary */}
          <div>
            <h3 className="text-xs font-bold text-sat-muted mb-2 flex items-center space-x-1.5">
              <Zap className="w-3.5 h-3.5 text-sat-cyan" />
              <span>AGGREGATE INFERENCE LATENCY BENCHMARKS</span>
            </h3>
            {Object.keys(benchmarks.summary).length > 0 ? (
              <div className="bg-sat-darker rounded border border-sat-border overflow-hidden text-xs">
                <table className="w-full text-left text-[11px]">
                  <thead className="bg-sat-panel text-sat-muted text-[10px]">
                    <tr>
                      <th className="p-2">MODEL</th>
                      <th className="p-2">TASK</th>
                      <th className="p-2">AVG LATENCY</th>
                      <th className="p-2">EXECUTIONS</th>
                      <th className="p-2">COMPUTE DEVICE</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-sat-border">
                    {Object.entries(benchmarks.summary).map(([modelName, s]: [string, any]) => (
                      <tr key={modelName} className="hover:bg-sat-surface text-sat-text">
                        <td className="p-2 font-bold">{modelName}</td>
                        <td className="p-2">{s.task}</td>
                        <td className="p-2 text-sat-accent font-bold">{s.avg_ms} ms</td>
                        <td className="p-2">{s.count} runs</td>
                        <td className="p-2 uppercase">{s.last_device}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-xs text-sat-muted italic bg-sat-darker p-3 rounded border border-sat-border">
                No benchmarks recorded yet. Run any analysis workflow to measure real-time hardware inference latency.
              </p>
            )}
          </div>

          {/* Recent Runs Log */}
          {benchmarks.recent_runs.length > 0 && (
            <div>
              <h3 className="text-xs font-bold text-sat-muted mb-2">RECENT EXECUTION LOGS</h3>
              <div className="bg-sat-darker rounded border border-sat-border overflow-hidden text-[11px] max-h-40 overflow-y-auto">
                <table className="w-full text-left">
                  <thead className="bg-sat-panel text-sat-muted text-[10px] sticky top-0">
                    <tr>
                      <th className="p-1.5">TIMESTAMP</th>
                      <th className="p-1.5">MODEL</th>
                      <th className="p-1.5">TASK</th>
                      <th className="p-1.5">LATENCY</th>
                      <th className="p-1.5">SIZE</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-sat-border">
                    {benchmarks.recent_runs.map((r, i) => (
                      <tr key={i} className="hover:bg-sat-surface text-sat-text">
                        <td className="p-1.5 text-sat-muted">{r.timestamp}</td>
                        <td className="p-1.5 font-semibold">{r.model}</td>
                        <td className="p-1.5">{r.task}</td>
                        <td className="p-1.5 text-sat-accent">{r.inference_time_ms} ms</td>
                        <td className="p-1.5 text-sat-muted">{r.image_size}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 bg-sat-darker border-t border-sat-border flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-1.5 bg-sat-surface hover:bg-sat-surface/80 border border-sat-border text-xs rounded text-sat-text font-mono transition"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
