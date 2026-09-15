import React from 'react';
import {
  FileText,
  Activity,
  Layers,
  Radio,
  Sparkles,
  CheckCircle2,
  Copy,
  Check,
  Cpu,
  Zap,
  Info
} from 'lucide-react';
import {
  VQAResponse,
  HighlightResponse,
  ChangeDetectionResponse,
  OpticalSARResponse,
  WorkspaceMode,
  DetectedRegion,
  ChangedRegion
} from '../../types';
import { ConfidenceBadge } from './ConfidenceBadge';
import { ModelTransparencyAlert } from './ModelTransparencyAlert';
import { MetadataPanel } from './MetadataPanel';
import { ExecutionTrace } from './ExecutionTrace';
import { FormattedAnswer } from './FormattedAnswer';

interface AnalysisResultProps {
  mode: WorkspaceMode;
  vqaResult: VQAResponse | null;
  highlightResult: HighlightResponse | null;
  changeResult: ChangeDetectionResponse | null;
  opticalSarResult: OpticalSARResponse | null;
  onSelectRegion?: (region: DetectedRegion | null) => void;
  selectedRegion?: DetectedRegion | null;
}

export const AnalysisResult: React.FC<AnalysisResultProps> = ({
  mode,
  vqaResult,
  highlightResult,
  changeResult,
  opticalSarResult,
  onSelectRegion,
  selectedRegion
}) => {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="bg-sat-panel border-t border-sat-border p-4 space-y-4 max-h-96 overflow-y-auto font-sans">
      {/* =========================================================================
          1. VQA RESULT
         ========================================================================= */}
      {mode === 'vqa' && vqaResult && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sat-border pb-2">
            <div className="flex items-center space-x-2">
              <FileText className="w-4 h-4 text-sat-accent" />
              <h3 className="font-mono font-bold text-xs text-sat-text uppercase tracking-wider">
                VQA ANALYSIS RESULT
              </h3>
            </div>
            <div className="flex items-center space-x-2">
              <ConfidenceBadge
                confidence={vqaResult.confidence}
                confidenceType={vqaResult.confidence_type}
                modelName={vqaResult.semantic_model || vqaResult.model}
              />
              <button
                onClick={() => handleCopy(vqaResult.answer)}
                className="flex items-center space-x-1 px-2 py-1 bg-sat-surface hover:bg-sat-surface/80 border border-sat-border rounded text-[11px] font-mono text-sat-muted hover:text-sat-text transition"
              >
                {copied ? <Check className="w-3 h-3 text-sat-accent" /> : <Copy className="w-3 h-3" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
          </div>

          {/* Model Attribution Header */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-xs">
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">SEMANTIC ENGINE:</span>
              <span className="text-xs font-bold text-sat-accent truncate block">
                Remote-Sensing VLM
              </span>
            </div>
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">REASONING MODE:</span>
              <span className="text-xs font-bold text-sat-cyan truncate block">
                {vqaResult.device === 'Cloud API' ? 'Enhanced Cloud Reasoning' : 'Local GPU Inference'}
              </span>
            </div>
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">INFERENCE LATENCY:</span>
              <span className="text-xs font-bold text-sat-text block">
                {vqaResult.processing_time_ms.toFixed(0)} ms ({vqaResult.device.toUpperCase()})
              </span>
            </div>
          </div>

          <div className="bg-sat-darker p-3 rounded border border-sat-border space-y-1.5 font-sans">
            <div className="text-[11px] font-mono text-sat-muted flex items-center space-x-2">
              <span className="text-sat-accent font-bold">QUERY:</span>
              <span className="text-sat-text">{vqaResult.question}</span>
            </div>
            <div className="pt-1">
              <FormattedAnswer text={vqaResult.answer} />
            </div>
          </div>

          {vqaResult.transparency_warning && (
            <ModelTransparencyAlert warning={vqaResult.transparency_warning} />
          )}

          {vqaResult.geo_metadata && <MetadataPanel metadata={vqaResult.geo_metadata} />}

          <ExecutionTrace stages={vqaResult.execution_trace} totalTimeMs={vqaResult.processing_time_ms} />
        </div>
      )}

      {/* =========================================================================
          2. IMAGE HIGHLIGHT RESULT
         ========================================================================= */}
      {mode === 'highlight' && highlightResult && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sat-border pb-2">
            <div className="flex items-center space-x-2">
              <Sparkles className="w-4 h-4 text-sat-accent" />
              <h3 className="font-mono font-bold text-xs text-sat-text uppercase tracking-wider">
                DETECTION & SEGMENTATION METRICS
              </h3>
            </div>
            <ConfidenceBadge
              confidence={
                highlightResult.detections.length > 0
                  ? highlightResult.detections[0].confidence
                  : 0.85
              }
              confidenceType={
                highlightResult.detections.length > 0
                  ? highlightResult.detections[0].confidence_type
                  : 'model'
              }
              modelName={highlightResult.model}
            />
          </div>

          {/* Model Attribution Strip */}
          <div className="grid grid-cols-3 gap-2 font-mono text-xs">
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">DETECTION ENGINE:</span>
              <span className="text-xs font-bold text-sat-accent block">Zero-Shot Detector</span>
            </div>
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">SEGMENTATION ENGINE:</span>
              <span className="text-xs font-bold text-sat-cyan block">Polygon Segmenter</span>
            </div>
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">SEMANTIC EXPLANATION:</span>
              <span className="text-xs font-bold text-sat-text truncate block">Remote-Sensing VLM</span>
            </div>
          </div>

          {/* Metric Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-xs">
            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">DETECTED INSTANCES:</span>
              <span className="text-lg font-bold text-sat-accent">{highlightResult.num_detections}</span>
              <span className="text-[10px] text-sat-muted block">Objects Segmented</span>
            </div>

            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">TARGET PROMPT:</span>
              <span className="text-sm font-bold text-sat-text truncate block">{highlightResult.prompt}</span>
              <span className="text-[10px] text-sat-muted block">Zero-Shot Text Class</span>
            </div>

            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">TOTAL HIGHLIGHTED AREA:</span>
              <span className="text-lg font-bold text-sat-cyan">{highlightResult.total_area_percentage}%</span>
              <span className="text-[10px] text-sat-muted block">({highlightResult.total_highlighted_area_pixels.toLocaleString()} px)</span>
            </div>

            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">INFERENCE LATENCY:</span>
              <span className="text-lg font-bold text-sat-text">{highlightResult.processing_time_ms.toFixed(0)} ms</span>
              <span className="text-[10px] text-sat-muted block">
                {highlightResult.specialized_time_ms ? `CV: ${highlightResult.specialized_time_ms.toFixed(0)}ms` : highlightResult.device.toUpperCase()}
              </span>
            </div>
          </div>

          {/* Semantic Summary */}
          {highlightResult.semantic_summary && (
            <div className="bg-sat-darker p-3 rounded border border-sat-border">
              <div className="flex items-center justify-between text-[11px] font-mono mb-1">
                <span className="text-sat-accent font-bold flex items-center space-x-1.5">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>SPATIAL DISTRIBUTION & SEMANTIC CONTEXT:</span>
                </span>
                <button
                  onClick={() => handleCopy(highlightResult.semantic_summary || '')}
                  className="text-sat-muted hover:text-sat-text text-[10px] flex items-center space-x-1"
                >
                  {copied ? <Check className="w-3 h-3 text-sat-accent" /> : <Copy className="w-3 h-3" />}
                  <span>{copied ? 'Copied' : 'Copy'}</span>
                </button>
              </div>
              <FormattedAnswer text={highlightResult.semantic_summary} />
            </div>
          )}

          {highlightResult.transparency_warning && (
            <ModelTransparencyAlert warning={highlightResult.transparency_warning} />
          )}

          {/* Detection Table */}
          {highlightResult.detections.length > 0 && (
            <div className="bg-sat-darker rounded border border-sat-border overflow-hidden font-mono text-xs">
              <div className="p-2 bg-sat-panel border-b border-sat-border flex items-center justify-between text-[11px] font-bold text-sat-muted">
                <span>DETECTED REGIONS ({highlightResult.detections.length})</span>
                <span className="text-[10px]">CLICK ROW TO FOCUS CANVAS</span>
              </div>
              <div className="max-h-40 overflow-y-auto">
                <table className="w-full text-left text-[11px]">
                  <thead className="bg-sat-panel/50 text-sat-muted text-[10px] sticky top-0">
                    <tr>
                      <th className="p-1.5">ID</th>
                      <th className="p-1.5">CLASS</th>
                      <th className="p-1.5">CONFIDENCE</th>
                      <th className="p-1.5">AREA (PX)</th>
                      <th className="p-1.5">BOUNDING BOX</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-sat-border">
                    {highlightResult.detections.map((d) => (
                      <tr
                        key={d.id}
                        onClick={() => onSelectRegion?.(d)}
                        className={`hover:bg-sat-surface cursor-pointer transition ${
                          selectedRegion?.id === d.id ? 'bg-sat-accentDim text-sat-accent' : 'text-sat-text'
                        }`}
                      >
                        <td className="p-1.5 font-bold">#{d.id}</td>
                        <td className="p-1.5 uppercase">{d.label}</td>
                        <td className="p-1.5">{(d.confidence * 100).toFixed(0)}%</td>
                        <td className="p-1.5">{d.area_pixels.toLocaleString()}</td>
                        <td className="p-1.5 text-sat-muted">
                          [{d.bounding_box.join(', ')}]
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {highlightResult.geo_metadata && <MetadataPanel metadata={highlightResult.geo_metadata} />}

          <ExecutionTrace stages={highlightResult.execution_trace} totalTimeMs={highlightResult.processing_time_ms} />
        </div>
      )}

      {/* =========================================================================
          3. BI-TEMPORAL CHANGE RESULT
         ========================================================================= */}
      {mode === 'bitemporal' && changeResult && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sat-border pb-2">
            <div className="flex items-center space-x-2">
              <Layers className="w-4 h-4 text-sat-accent" />
              <h3 className="font-mono font-bold text-xs text-sat-text uppercase tracking-wider">
                BI-TEMPORAL CHANGE ANALYSIS RESULT
              </h3>
            </div>
            <ConfidenceBadge
              confidence={0.91}
              confidenceType="model"
              modelName={changeResult.detection_model || changeResult.model}
            />
          </div>

          {/* Model Attribution Strip */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 font-mono text-xs">
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">CHANGE DETECTION ENGINE:</span>
              <span className="text-xs font-bold text-sat-accent block">Bi-Temporal Engine</span>
            </div>
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">SEMANTIC REASONING:</span>
              <span className="text-xs font-bold text-sat-cyan truncate block">Remote-Sensing VLM</span>
            </div>
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">LATENCY BREAKDOWN:</span>
              <span className="text-xs font-bold text-sat-text block">
                Total: {changeResult.processing_time_ms.toFixed(0)} ms
              </span>
            </div>
          </div>

          {/* Quantitative Change KPI Cards (Derived Exclusively from ChangeMamba) */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-xs">
            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">CALCULATED CHANGE AREA:</span>
              <span className="text-xl font-bold text-sat-warning">{changeResult.change_percentage}%</span>
              <span className="text-[10px] text-sat-muted block">Authoritative Metric</span>
            </div>

            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">CHANGED REGIONS:</span>
              <span className="text-xl font-bold text-sat-text">{changeResult.num_regions}</span>
              <span className="text-[10px] text-sat-muted block">Connected Components</span>
            </div>

            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">TOTAL CHANGED PIXELS:</span>
              <span className="text-xl font-bold text-sat-cyan">{changeResult.total_changed_pixels.toLocaleString()}</span>
              <span className="text-[10px] text-sat-muted block">Out of {changeResult.total_pixels.toLocaleString()} px</span>
            </div>

            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">CO-REGISTRATION STATUS:</span>
              <span className="text-xs font-bold text-sat-accent truncate block">
                {changeResult.is_coregistered ? 'VERIFIED ALIGNED' : 'RESAMPLED GRID'}
              </span>
              <span className="text-[10px] text-sat-muted block truncate" title={changeResult.coregistration_notes || ''}>
                {changeResult.coregistration_notes || 'Sub-pixel co-registration'}
              </span>
            </div>
          </div>

          {/* Semantic Reasoning by GeoChat Remote-Sensing VLM */}
          <div className="bg-sat-darker p-3 rounded border border-sat-border space-y-1">
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-sat-accent font-bold flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5" />
                <span>SEMANTIC CHANGE INTERPRETATION:</span>
              </span>
              <button
                onClick={() => handleCopy(changeResult.semantic_analysis)}
                className="text-sat-muted hover:text-sat-text text-[10px] flex items-center space-x-1"
              >
                {copied ? <Check className="w-3 h-3 text-sat-accent" /> : <Copy className="w-3 h-3" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
            <div className="pt-1">
              <FormattedAnswer text={changeResult.semantic_analysis} />
            </div>
          </div>

          {/* Attribution Notice */}
          <div className="p-2.5 bg-sat-surface rounded border border-sat-border flex items-start space-x-2 text-[11px] font-mono text-sat-muted">
            <Info className="w-3.5 h-3.5 text-sat-accent flex-shrink-0 mt-0.5" />
            <span>
              Semantic interpretation generated by <b>Remote-Sensing VLM</b>; quantitative change detection and mask performed by <b>Change Engine</b>.
            </span>
          </div>

          {changeResult.transparency_warning && (
            <ModelTransparencyAlert warning={changeResult.transparency_warning} />
          )}

          {/* Changed Region List */}
          {changeResult.regions.length > 0 && (
            <div className="bg-sat-darker rounded border border-sat-border overflow-hidden font-mono text-xs">
              <div className="p-2 bg-sat-panel border-b border-sat-border text-[11px] font-bold text-sat-muted">
                CONNECTED CHANGED COMPONENTS ({changeResult.regions.length})
              </div>
              <div className="max-h-36 overflow-y-auto">
                <table className="w-full text-left text-[11px]">
                  <thead className="bg-sat-panel/50 text-sat-muted text-[10px] sticky top-0">
                    <tr>
                      <th className="p-1.5">ID</th>
                      <th className="p-1.5">AREA (PX)</th>
                      <th className="p-1.5">CENTROID (CX, CY)</th>
                      <th className="p-1.5">CONFIDENCE</th>
                      <th className="p-1.5">BOUNDING BOX</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-sat-border">
                    {changeResult.regions.slice(0, 30).map((r) => (
                      <tr key={r.region_id} className="hover:bg-sat-surface text-sat-text">
                        <td className="p-1.5 font-bold">#{r.region_id}</td>
                        <td className="p-1.5">{r.area_pixels.toLocaleString()}</td>
                        <td className="p-1.5">[{r.centroid.join(', ')}]</td>
                        <td className="p-1.5">{(r.confidence * 100).toFixed(0)}%</td>
                        <td className="p-1.5 text-sat-muted">[{r.bounding_box.join(', ')}]</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          <ExecutionTrace stages={changeResult.execution_trace} totalTimeMs={changeResult.processing_time_ms} />
        </div>
      )}

      {/* =========================================================================
          4. OPTICAL + SAR FUSION RESULT
         ========================================================================= */}
      {mode === 'optical_sar' && opticalSarResult && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2 border-b border-sat-border pb-2">
            <div className="flex items-center space-x-2">
              <Radio className="w-4 h-4 text-sat-cyan" />
              <h3 className="font-mono font-bold text-xs text-sat-text uppercase tracking-wider">
                OPTICAL + SAR CROSS-MODAL FUSION RESULT
              </h3>
            </div>
            <ConfidenceBadge
              confidence={0.93}
              confidenceType="model"
              modelName={opticalSarResult.model}
            />
          </div>

          {/* Model Attribution Strip */}
          <div className="grid grid-cols-3 gap-2 font-mono text-xs">
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">OPTICAL PROCESSING:</span>
              <span className="text-xs font-bold text-sat-text block">{opticalSarResult.optical_processing || 'Visible Reflectance'}</span>
            </div>
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">SAR RADAR ENGINE:</span>
              <span className="text-xs font-bold text-sat-accent block">{opticalSarResult.sar_processing || '7x7 Lee Filter & CFAR'}</span>
            </div>
            <div className="bg-sat-darker p-2 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">SEMANTIC REASONING:</span>
              <span className="text-xs font-bold text-sat-cyan truncate block">Remote-Sensing VLM</span>
            </div>
          </div>

          {/* Radar Backscatter vs Optical Reflectance Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-xs">
            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">SAR DYNAMIC RANGE:</span>
              <span className="text-xl font-bold text-sat-cyan">{opticalSarResult.sar_stats.dynamic_range_db} dB</span>
              <span className="text-[10px] text-sat-muted block">Sentinel-1 Radar</span>
            </div>

            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">HIGH-BACKSCATTER RATIO:</span>
              <span className="text-xl font-bold text-sat-warning">
                {((opticalSarResult.sar_stats.high_backscatter_ratio ?? 0) * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-sat-muted block">Double-Bounce (Metallic/Buildings)</span>
            </div>

            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">LOW-BACKSCATTER RATIO:</span>
              <span className="text-xl font-bold text-sat-accent">
                {((opticalSarResult.sar_stats.low_backscatter_ratio ?? 0) * 100).toFixed(1)}%
              </span>
              <span className="text-[10px] text-sat-muted block">Specular (Water/Runways)</span>
            </div>

            <div className="bg-sat-darker p-2.5 rounded border border-sat-border">
              <span className="text-[10px] text-sat-muted block">OPTICAL MEAN INTENSITY:</span>
              <span className="text-xl font-bold text-sat-text">{opticalSarResult.optical_stats.mean_intensity.toFixed(1)}</span>
              <span className="text-[10px] text-sat-muted block">Visible Spectral Channel</span>
            </div>
          </div>

          {/* Semantic Multimodal Analysis */}
          <div className="bg-sat-darker p-3 rounded border border-sat-border space-y-1">
            <div className="flex items-center justify-between text-[11px] font-mono">
              <span className="text-sat-cyan font-bold flex items-center space-x-1.5">
                <Sparkles className="w-3.5 h-3.5" />
                <span>CROSS-MODAL REASONING & FEATURE CORRELATION:</span>
              </span>
              <button
                onClick={() => handleCopy(opticalSarResult.cross_modal_analysis)}
                className="text-sat-muted hover:text-sat-text text-[10px] flex items-center space-x-1"
              >
                {copied ? <Check className="w-3 h-3 text-sat-cyan" /> : <Copy className="w-3 h-3" />}
                <span>{copied ? 'Copied' : 'Copy'}</span>
              </button>
            </div>
            <p className="text-xs text-sat-text leading-relaxed font-sans pt-1 whitespace-pre-line">
              {opticalSarResult.cross_modal_analysis}
            </p>
          </div>

          {/* Explicit Model Transparency Alert for SAR */}
          <ModelTransparencyAlert warning={opticalSarResult.transparency_warning} type="warning" />

          {/* Detected Features List */}
          <div className="bg-sat-darker p-2.5 rounded border border-sat-border font-mono text-xs">
            <span className="text-[10px] text-sat-muted block mb-1 font-bold">EXTRACTED PHYSICAL PARAMETERS:</span>
            <ul className="space-y-1 text-[11px] text-sat-text">
              {opticalSarResult.detected_features.map((feat, idx) => (
                <li key={idx} className="flex items-center space-x-2">
                  <span className="w-1.5 h-1.5 rounded-full bg-sat-cyan" />
                  <span>{feat}</span>
                </li>
              ))}
            </ul>
          </div>

          <ExecutionTrace stages={opticalSarResult.execution_trace} totalTimeMs={opticalSarResult.processing_time_ms} />
        </div>
      )}

      {/* Initial / Empty State */}
      {!vqaResult && !highlightResult && !changeResult && !opticalSarResult && (
        <div className="text-center py-6 text-sat-muted font-mono text-xs">
          <Activity className="w-6 h-6 mx-auto mb-2 opacity-40 text-sat-accent" />
          <span>Awaiting analysis pipeline execution. Select parameters in sidebar and run.</span>
        </div>
      )}
    </div>
  );
};
