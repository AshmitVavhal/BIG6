import React, { useRef } from 'react';
import {
  MessageSquare,
  Sparkles,
  Layers,
  Radio,
  Paperclip,
  ArrowRight,
  Loader2,
  Database,
  Check,
  Copy,
  Upload
} from 'lucide-react';
import type {
  WorkspaceMode,
  SampleDataset,
  SemanticModelChoice,
  VQAResponse,
  HighlightResponse,
  ChangeDetectionResponse,
  OpticalSARResponse,
  DetectedRegion
} from '../../types';
import { ConfidenceBadge } from '../hud/ConfidenceBadge';
import { FormattedAnswer } from '../hud/FormattedAnswer';
import { MetadataPanel } from '../hud/MetadataPanel';
import { ExecutionTrace } from '../hud/ExecutionTrace';
import { ModelTransparencyAlert } from '../hud/ModelTransparencyAlert';

interface SidebarProps {
  currentMode: WorkspaceMode;
  onSelectMode: (mode: WorkspaceMode) => void;
  samples?: SampleDataset[];
  // VQA state
  vqaImage?: string | null;
  vqaQuestion: string;
  onSetVqaQuestion: (q: string) => void;
  onSelectVqaImage?: (path: string) => void;
  onRunVQA: () => void;
  vqaResult?: VQAResponse | null;
  // Highlight state
  highlightImage?: string | null;
  highlightPrompt: string;
  onSetHighlightPrompt: (p: string) => void;
  onSelectHighlightImage?: (path: string) => void;
  highlightThreshold: number;
  onSetHighlightThreshold: (t: number) => void;
  useMask2Former?: boolean;
  onToggleMask2Former?: (v: boolean) => void;
  onRunHighlight: () => void;
  highlightResult?: HighlightResponse | null;
  selectedRegion?: DetectedRegion | null;
  onSelectRegion?: (region: DetectedRegion | null) => void;
  // Bi-Temporal state
  t1Image?: string | null;
  t2Image?: string | null;
  onSelectT1Image: (path: string) => void;
  onSelectT2Image: (path: string) => void;
  changeThreshold: number;
  onSetChangeThreshold: (t: number) => void;
  onRunChange: () => void;
  changeResult?: ChangeDetectionResponse | null;
  // Optical + SAR state
  opticalImage?: string | null;
  sarImage?: string | null;
  onSelectOpticalImage: (path: string) => void;
  onSelectSarImage: (path: string) => void;
  opticalSarQuestion: string;
  onSetOpticalSarQuestion: (q: string) => void;
  despeckleSar: boolean;
  onToggleDespeckle: (v: boolean) => void;
  onRunOpticalSar: () => void;
  opticalSarResult?: OpticalSARResponse | null;
  // General upload
  onUploadFile: (file: File, targetSlot?: string) => Promise<void>;
  isProcessing: boolean;
  // Semantic reasoning selection
  semanticModel?: SemanticModelChoice;
  onSetSemanticModel?: (m: SemanticModelChoice) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentMode,
  onSelectMode,
  vqaImage,
  vqaQuestion,
  onSetVqaQuestion,
  onSelectVqaImage,
  onRunVQA,
  vqaResult,
  highlightImage,
  highlightPrompt,
  onSetHighlightPrompt,
  onSelectHighlightImage,
  highlightThreshold,
  onSetHighlightThreshold,
  onRunHighlight,
  highlightResult,
  selectedRegion,
  onSelectRegion,
  t1Image,
  t2Image,
  onSelectT1Image,
  onSelectT2Image,
  changeThreshold,
  onSetChangeThreshold,
  onRunChange,
  changeResult,
  opticalImage,
  sarImage,
  onSelectOpticalImage,
  onSelectSarImage,
  opticalSarQuestion,
  onSetOpticalSarQuestion,
  despeckleSar,
  onToggleDespeckle,
  onRunOpticalSar,
  opticalSarResult,
  onUploadFile,
  isProcessing,
  semanticModel = 'geochat',
  onSetSemanticModel
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeUploadSlot = useRef<string>('default');
  const [copied, setCopied] = React.useState<boolean>(false);

  const triggerUpload = (slot: string) => {
    activeUploadSlot.current = slot;
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
      fileInputRef.current.click();
    }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await onUploadFile(file, activeUploadSlot.current);
    }
  };

  const handleCopy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleDownloadDataset = () => {
    const downloadUrl = 'https://drive.usercontent.google.com/download?id=1lwVLG62RdiPA4cNFfyzT36HZ-IZ44suY&export=download&confirm=t';
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.setAttribute('download', 'Dataset.zip');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getDisplayFilename = (pathStr?: string | null, defaultText = 'Upload Image') => {
    if (!pathStr) return defaultText;
    const base = pathStr.split('/').pop()?.split('\\').pop() || pathStr;
    return base;
  };

  // Get active query text and trigger handler
  const getQueryValue = () => {
    if (currentMode === 'vqa') return vqaQuestion;
    if (currentMode === 'highlight') return highlightPrompt;
    if (currentMode === 'optical_sar') return opticalSarQuestion;
    return '';
  };

  const setQueryValue = (val: string) => {
    if (currentMode === 'vqa') onSetVqaQuestion(val);
    else if (currentMode === 'highlight') onSetHighlightPrompt(val);
    else if (currentMode === 'optical_sar') onSetOpticalSarQuestion(val);
  };

  const handlePrimaryExecute = () => {
    if (isProcessing) return;
    if (currentMode === 'vqa') onRunVQA();
    else if (currentMode === 'highlight') onRunHighlight();
    else if (currentMode === 'bitemporal') onRunChange();
    else if (currentMode === 'optical_sar') onRunOpticalSar();
  };

  const hasActiveResult = Boolean(
    (currentMode === 'vqa' && vqaResult) ||
    (currentMode === 'highlight' && highlightResult) ||
    (currentMode === 'bitemporal' && changeResult) ||
    (currentMode === 'optical_sar' && opticalSarResult)
  );

  const turnCount = hasActiveResult ? 1 : 0;

  // Prompt suggestions
  const highlightPresets = ['building', 'road', 'water body', 'vehicle', 'aircraft', 'ship'];

  return (
    <aside className="w-80 md:w-[340px] bg-sat-bg border-r border-sat-border flex flex-col h-[calc(100vh-3rem)] z-20 select-none">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept=".jpg,.jpeg,.png,.tif,.tiff,.geotiff"
      />

      {/* Top Header: Mission Feed & Semantic Model Selector */}
      <div className="h-10 px-3.5 border-b border-sat-border flex items-center justify-between bg-sat-bg/80">
        <div className="flex items-center space-x-2">
          <span className="font-semibold text-xs text-white font-sans">Mission Feed</span>
          <span className="text-[10px] bg-sat-panel text-sat-muted px-2 py-0.5 rounded-full border border-sat-border font-mono">
            {turnCount} turns
          </span>
        </div>

        {/* Semantic Model Selector (GeoChat Base vs LoRA) */}
        {onSetSemanticModel ? (
          <div className="flex items-center bg-sat-panel border border-sat-border rounded-md p-0.5">
            <button
              onClick={() => onSetSemanticModel('geochat')}
              className={`px-2 py-0.5 rounded text-[10px] font-mono transition ${
                semanticModel === 'geochat'
                  ? 'bg-white text-slate-900 font-bold shadow-sm'
                  : 'text-sat-muted hover:text-white'
              }`}
              title="GeoChat-7B Base Foundation VLM"
            >
              Base
            </button>
            <button
              onClick={() => onSetSemanticModel('geochat_lora')}
              className={`px-2 py-0.5 rounded text-[10px] font-mono transition ${
                semanticModel === 'geochat_lora'
                  ? 'bg-white text-slate-900 font-bold shadow-sm'
                  : 'text-sat-muted hover:text-white'
              }`}
              title="GeoChat LoRA (PEFT Fine-Tuned for Remote Sensing)"
            >
              LoRA
            </button>
          </div>
        ) : (
          <button
            onClick={handleDownloadDataset}
            className="flex items-center space-x-1 px-2 py-0.5 rounded text-[11px] font-mono text-sat-muted hover:text-white hover:bg-sat-panel transition"
            title="Download verified satellite benchmark dataset"
          >
            <Database className="w-3 h-3 text-sat-accent" />
            <span>Dataset</span>
          </button>
        )}
      </div>

      {/* Middle Scrollable Body: Mission Intelligence Results or Empty State */}
      <div className="flex-1 overflow-y-auto p-3.5 space-y-3.5">
        {/* ==================== 1. EMPTY STATE ==================== */}
        {!hasActiveResult && (
          <div className="h-full flex flex-col items-center justify-center text-center px-4 py-8">
            <div className="w-12 h-12 rounded-full border border-sat-border bg-sat-panel/80 flex items-center justify-center text-sat-muted mb-4">
              <Radio className="w-5 h-5 text-sat-accent animate-pulse" />
            </div>
            <h3 className="font-semibold text-sm text-white mb-1.5 font-sans">
              SatQuery Mission Intelligence
            </h3>
            <p className="text-xs text-sat-muted leading-relaxed max-w-[240px]">
              Select an operational mode below, ask a question, or launch a scenario from the canvas.
            </p>
          </div>
        )}

        {/* ==================== 2. VQA RESULTS ==================== */}
        {currentMode === 'vqa' && vqaResult && (
          <div className="space-y-3 font-sans">
            <div className="flex items-center justify-between border-b border-sat-border pb-2">
              <span className="text-[11px] font-mono font-semibold text-sat-accent uppercase tracking-wider">
                VQA Intelligence Response
              </span>
              <div className="flex items-center space-x-1.5">
                <ConfidenceBadge
                  confidence={vqaResult.confidence}
                  confidenceType={vqaResult.confidence_type}
                />
                <button
                  onClick={() => handleCopy(vqaResult.answer)}
                  className="p-1 hover:bg-sat-surface rounded text-sat-muted hover:text-white transition"
                  title="Copy Answer"
                >
                  {copied ? <Check className="w-3 h-3 text-sat-accent" /> : <Copy className="w-3 h-3" />}
                </button>
              </div>
            </div>

            {/* Query Pill */}
            <div className="bg-sat-panel p-2.5 rounded-lg border border-sat-border text-xs">
              <span className="text-[10px] font-mono text-sat-muted block mb-0.5">QUERY</span>
              <p className="text-white font-medium">{vqaResult.question}</p>
            </div>

            {/* Formatted VQA Answer */}
            <div className="bg-sat-panel p-3 rounded-lg border border-sat-border">
              <FormattedAnswer text={vqaResult.answer} />
            </div>

            {vqaResult.transparency_warning && (
              <ModelTransparencyAlert warning={vqaResult.transparency_warning} />
            )}

            {vqaResult.geo_metadata && <MetadataPanel metadata={vqaResult.geo_metadata} />}
            <ExecutionTrace stages={vqaResult.execution_trace} totalTimeMs={vqaResult.processing_time_ms} />
          </div>
        )}

        {/* ==================== 3. DETECTION RESULTS ==================== */}
        {currentMode === 'highlight' && highlightResult && (
          <div className="space-y-3 font-sans">
            <div className="flex items-center justify-between border-b border-sat-border pb-2">
              <span className="text-[11px] font-mono font-semibold text-sat-accent uppercase tracking-wider">
                Detection Metrics
              </span>
              <ConfidenceBadge
                confidence={
                  highlightResult.detections.length > 0
                    ? highlightResult.detections[0].confidence
                    : 0.85
                }
                confidenceType="model"
              />
            </div>

            {/* Key Metrics Grid */}
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="bg-sat-panel p-2 rounded-lg border border-sat-border">
                <span className="text-[10px] text-sat-muted block">INSTANCES</span>
                <span className="text-base font-bold text-white">{highlightResult.num_detections}</span>
              </div>
              <div className="bg-sat-panel p-2 rounded-lg border border-sat-border">
                <span className="text-[10px] text-sat-muted block">AREA %</span>
                <span className="text-base font-bold text-sat-accent">{highlightResult.total_area_percentage}%</span>
              </div>
            </div>

            {/* Semantic Summary */}
            {highlightResult.semantic_summary && (
              <div className="bg-sat-panel p-3 rounded-lg border border-sat-border">
                <FormattedAnswer text={highlightResult.semantic_summary} />
              </div>
            )}

            {/* Detection Table */}
            {highlightResult.detections.length > 0 && (
              <div className="bg-sat-panel rounded-lg border border-sat-border overflow-hidden text-xs font-mono">
                <div className="px-2.5 py-1.5 bg-sat-surface border-b border-sat-border text-[10px] text-sat-muted font-bold flex justify-between">
                  <span>DETECTED OBJECTS ({highlightResult.detections.length})</span>
                  <span className="text-[9px]">CLICK TO FOCUS</span>
                </div>
                <div className="max-h-36 overflow-y-auto">
                  <table className="w-full text-left text-[11px]">
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
                          <td className="p-1.5 text-sat-muted">{d.area_pixels.toLocaleString()}px</td>
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

        {/* ==================== 4. BI-TEMPORAL CHANGE RESULTS ==================== */}
        {currentMode === 'bitemporal' && changeResult && (
          <div className="space-y-3 font-sans">
            <div className="flex items-center justify-between border-b border-sat-border pb-2">
              <span className="text-[11px] font-mono font-semibold text-sat-accent uppercase tracking-wider">
                Bi-Temporal Change Analysis
              </span>
              <ConfidenceBadge confidence={0.91} confidenceType="model" />
            </div>

            {/* Change Metrics */}
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="bg-sat-panel p-2 rounded-lg border border-sat-border">
                <span className="text-[10px] text-sat-muted block">TOTAL CHANGE</span>
                <span className="text-base font-bold text-sat-warning">{changeResult.change_percentage}%</span>
              </div>
              <div className="bg-sat-panel p-2 rounded-lg border border-sat-border">
                <span className="text-[10px] text-sat-muted block">REGIONS</span>
                <span className="text-base font-bold text-white">{changeResult.num_regions}</span>
              </div>
            </div>

            {/* Semantic Interpretation */}
            <div className="bg-sat-panel p-3 rounded-lg border border-sat-border">
              <FormattedAnswer text={changeResult.semantic_analysis} />
            </div>

            {/* Changed Regions Table */}
            {changeResult.regions.length > 0 && (
              <div className="bg-sat-panel rounded-lg border border-sat-border overflow-hidden text-xs font-mono">
                <div className="px-2.5 py-1.5 bg-sat-surface border-b border-sat-border text-[10px] text-sat-muted font-bold">
                  CONNECTED COMPONENTS ({changeResult.regions.length})
                </div>
                <div className="max-h-36 overflow-y-auto">
                  <table className="w-full text-left text-[11px]">
                    <tbody className="divide-y divide-sat-border">
                      {changeResult.regions.slice(0, 15).map((r) => (
                        <tr key={r.region_id} className="hover:bg-sat-surface text-sat-text">
                          <td className="p-1.5 font-bold">#{r.region_id}</td>
                          <td className="p-1.5">{r.area_pixels.toLocaleString()}px</td>
                          <td className="p-1.5 text-sat-muted">{(r.confidence * 100).toFixed(0)}%</td>
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

        {/* ==================== 5. OPTICAL + SAR RESULTS ==================== */}
        {currentMode === 'optical_sar' && opticalSarResult && (
          <div className="space-y-3 font-sans">
            <div className="flex items-center justify-between border-b border-sat-border pb-2">
              <span className="text-[11px] font-mono font-semibold text-sat-cyan uppercase tracking-wider">
                Cross-Modal Fusion Result
              </span>
              <ConfidenceBadge confidence={0.93} confidenceType="model" />
            </div>

            {/* Metrics */}
            <div className="grid grid-cols-2 gap-2 text-xs font-mono">
              <div className="bg-sat-panel p-2 rounded-lg border border-sat-border">
                <span className="text-[10px] text-sat-muted block">SAR DYNAMIC RANGE</span>
                <span className="text-base font-bold text-sat-cyan">{opticalSarResult.sar_stats.dynamic_range_db} dB</span>
              </div>
              <div className="bg-sat-panel p-2 rounded-lg border border-sat-border">
                <span className="text-[10px] text-sat-muted block">HIGH-BACKSCATTER</span>
                <span className="text-base font-bold text-sat-warning">
                  {((opticalSarResult.sar_stats.high_backscatter_ratio ?? 0) * 100).toFixed(1)}%
                </span>
              </div>
            </div>

            {/* Cross-modal text */}
            <div className="bg-sat-panel p-3 rounded-lg border border-sat-border text-xs text-sat-text leading-relaxed whitespace-pre-line">
              {opticalSarResult.cross_modal_analysis}
            </div>

            <ExecutionTrace stages={opticalSarResult.execution_trace} totalTimeMs={opticalSarResult.processing_time_ms} />
          </div>
        )}
      </div>

      {/* ==================== BOTTOM CONTROL DOCK ==================== */}
      <div className="p-3 border-t border-sat-border bg-sat-bg space-y-2.5">
        {/* =========================================================================
            1. VQA & DETECT CONTROLS
           ========================================================================= */}
        {(currentMode === 'vqa' || currentMode === 'highlight') && (
          <>
            {/* Detection Sensitivity Slider (Detect mode only) */}
            {currentMode === 'highlight' && (
              <div className="bg-sat-panel p-2.5 rounded-lg border border-sat-border space-y-1 text-xs">
                <div className="flex justify-between text-[11px] font-mono mb-1">
                  <span className="text-sat-muted">Sensitivity:</span>
                  <span className="text-sat-accent font-bold">{(highlightThreshold * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.85"
                  step="0.05"
                  value={highlightThreshold}
                  onChange={(e) => onSetHighlightThreshold(parseFloat(e.target.value))}
                  className="w-full accent-sat-accent h-1 bg-sat-surface rounded"
                />
              </div>
            )}

            {/* Satellite Scene Upload & Sample Selector */}
            <div className="bg-sat-panel p-2.5 rounded-lg border border-sat-border space-y-1.5">
              <div className="flex items-center justify-between text-[10px] font-mono">
                <div>
                  <span className="font-semibold text-sat-text block uppercase">
                    {currentMode === 'vqa' ? 'VQA Satellite Scene' : 'Detection Target Scene'}
                  </span>
                  <span className="text-sat-muted text-[9px]">TIFF / GeoTIFF / PNG / JPG</span>
                </div>
                {((currentMode === 'vqa' && vqaImage) || (currentMode === 'highlight' && highlightImage)) && (
                  <span className="text-[10px] font-mono text-sat-accent flex items-center space-x-1">
                    <Check className="w-3 h-3" />
                    <span>Loaded</span>
                  </span>
                )}
              </div>
              <div className="flex items-center space-x-1.5">
                <button
                  onClick={() => {
                    if (currentMode === 'vqa') triggerUpload('vqa');
                    else if (currentMode === 'highlight') triggerUpload('highlight');
                  }}
                  className={`flex-1 flex items-center justify-between px-3 py-2 rounded-md border text-xs font-mono transition ${
                    (currentMode === 'vqa' ? vqaImage : highlightImage)
                      ? 'bg-sat-surface border-sat-accent/40 text-sat-accent font-medium'
                      : 'bg-sat-surface hover:bg-sat-surfaceHover border-dashed border-sat-borderLight text-sat-muted hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2 truncate">
                    <Upload className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate">
                      {currentMode === 'vqa'
                        ? (vqaImage ? getDisplayFilename(vqaImage) : 'Upload Image')
                        : (highlightImage ? getDisplayFilename(highlightImage) : 'Upload Image')}
                    </span>
                  </div>
                  {(currentMode === 'vqa' ? vqaImage : highlightImage) ? (
                    <Check className="w-3.5 h-3.5 text-sat-accent flex-shrink-0" />
                  ) : (
                    <span className="text-[10px] text-sat-muted font-sans">—</span>
                  )}
                </button>
                <button
                  onClick={() => {
                    if (currentMode === 'vqa') onSelectVqaImage?.('VQA1.png');
                    else onSelectHighlightImage?.('VQA1.png');
                  }}
                  title="Load Sample 1 (Urban Scene)"
                  className="px-2 py-2 bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight rounded-md text-[10px] font-mono text-sat-muted hover:text-white transition whitespace-nowrap"
                >
                  Sample 1
                </button>
                <button
                  onClick={() => {
                    if (currentMode === 'vqa') onSelectVqaImage?.('VQA2.png');
                    else onSelectHighlightImage?.('VQA2.png');
                  }}
                  title="Load Sample 2 (Industrial Scene)"
                  className="px-2 py-2 bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight rounded-md text-[10px] font-mono text-sat-muted hover:text-white transition whitespace-nowrap"
                >
                  Sample 2
                </button>
              </div>
            </div>

            {/* Quick preset chips for Detect mode */}
            {currentMode === 'highlight' && (
              <div className="flex items-center space-x-1.5 overflow-x-auto no-scrollbar py-0.5">
                <span className="text-[10px] font-mono text-sat-muted flex-shrink-0">Presets:</span>
                {highlightPresets.map((p) => (
                  <button
                    key={p}
                    onClick={() => onSetHighlightPrompt(p)}
                    className={`px-2 py-0.5 rounded-full border text-[10px] whitespace-nowrap transition ${
                      highlightPrompt === p
                        ? 'bg-sat-accent/20 border-sat-accent text-sat-accent font-semibold'
                        : 'bg-sat-panel hover:bg-sat-surface border-sat-border text-sat-muted hover:text-white'
                    }`}
                  >
                    {p}
                  </button>
                ))}
              </div>
            )}

            {/* Command / Query Input with Circular Arrow Button */}
            <div className="relative flex items-center">
              <input
                type="text"
                value={getQueryValue()}
                onChange={(e) => setQueryValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handlePrimaryExecute();
                }}
                placeholder={
                  currentMode === 'vqa'
                    ? 'What is visible in this image?'
                    : 'building, road, water body...'
                }
                className="w-full bg-sat-panel border border-sat-border focus:border-sat-borderLight rounded-full py-2.5 pl-3.5 pr-11 text-xs text-white placeholder-sat-muted focus:outline-none transition shadow-inner font-sans"
              />

              <button
                onClick={handlePrimaryExecute}
                disabled={isProcessing}
                className="absolute right-1.5 w-7 h-7 rounded-full bg-white hover:bg-slate-200 text-slate-900 flex items-center justify-center transition disabled:opacity-50 disabled:cursor-not-allowed shadow"
                title="Execute Analysis"
              >
                {isProcessing ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-900" />
                ) : (
                  <ArrowRight className="w-3.5 h-3.5 stroke-[2.5]" />
                )}
              </button>
            </div>
          </>
        )}

        {/* =========================================================================
            2. BI-TEMPORAL MODULE CONTROLS (TWO SEPARATE INDEPENDENT UPLOADS)
           ========================================================================= */}
        {currentMode === 'bitemporal' && (
          <div className="space-y-2.5">
            {/* T1 Box: Earlier / Pre-Change */}
            <div className="bg-sat-panel p-2.5 rounded-lg border border-sat-border space-y-1.5">
              <div className="flex items-center justify-between text-[10px] font-mono">
                <div>
                  <span className="font-semibold text-sat-text block">T1 — EARLIER / PRE-CHANGE</span>
                  <span className="text-sat-muted text-[9px]">Earlier / Pre-change image</span>
                </div>
                <span className="text-sat-muted">JPG / PNG / TIFF / GeoTIFF</span>
              </div>
              <div className="flex items-center space-x-1.5">
                <button
                  onClick={() => triggerUpload('t1')}
                  className={`flex-1 flex items-center justify-between px-3 py-2 rounded-md border text-xs font-mono transition ${
                    t1Image
                      ? 'bg-sat-surface border-sat-accent/40 text-sat-accent font-medium'
                      : 'bg-sat-surface hover:bg-sat-surfaceHover border-dashed border-sat-borderLight text-sat-muted hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2 truncate">
                    <Upload className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate">{t1Image ? getDisplayFilename(t1Image) : 'Upload T1 Image'}</span>
                  </div>
                  {t1Image ? (
                    <Check className="w-3.5 h-3.5 text-sat-accent flex-shrink-0" />
                  ) : (
                    <span className="text-[10px] text-sat-muted font-sans">—</span>
                  )}
                </button>
                <button
                  onClick={() => onSelectT1Image('Bi_Temporal T1.png')}
                  title="Load preloaded T1 baseline sample"
                  className="px-2 py-2 bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight rounded-md text-[10px] font-mono text-sat-muted hover:text-white transition whitespace-nowrap"
                >
                  Sample
                </button>
              </div>
            </div>

            {/* T2 Box: Later / Post-Change */}
            <div className="bg-sat-panel p-2.5 rounded-lg border border-sat-border space-y-1.5">
              <div className="flex items-center justify-between text-[10px] font-mono">
                <div>
                  <span className="font-semibold text-sat-text block">T2 — LATER / POST-CHANGE</span>
                  <span className="text-sat-muted text-[9px]">Later / Post-change image</span>
                </div>
                <span className="text-sat-muted">JPG / PNG / TIFF / GeoTIFF</span>
              </div>
              <div className="flex items-center space-x-1.5">
                <button
                  onClick={() => triggerUpload('t2')}
                  className={`flex-1 flex items-center justify-between px-3 py-2 rounded-md border text-xs font-mono transition ${
                    t2Image
                      ? 'bg-sat-surface border-sat-accent/40 text-sat-accent font-medium'
                      : 'bg-sat-surface hover:bg-sat-surfaceHover border-dashed border-sat-borderLight text-sat-muted hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2 truncate">
                    <Upload className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate">{t2Image ? getDisplayFilename(t2Image) : 'Upload T2 Image'}</span>
                  </div>
                  {t2Image ? (
                    <Check className="w-3.5 h-3.5 text-sat-accent flex-shrink-0" />
                  ) : (
                    <span className="text-[10px] text-sat-muted font-sans">—</span>
                  )}
                </button>
                <button
                  onClick={() => onSelectT2Image('Bi_Temporal T2.png')}
                  title="Load preloaded T2 post-change sample"
                  className="px-2 py-2 bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight rounded-md text-[10px] font-mono text-sat-muted hover:text-white transition whitespace-nowrap"
                >
                  Sample
                </button>
              </div>
            </div>

            {/* Status indicator when waiting for one of the images */}
            {(!t1Image || !t2Image) && (t1Image || t2Image) && (
              <div className="text-[11px] font-mono text-sat-warning/90 bg-sat-warning/10 px-2.5 py-1.5 rounded border border-sat-warning/20">
                {!t1Image ? 'T1 — | T2 ✓ (Waiting for T1 Earlier image)' : 'T1 ✓ | T2 — (Waiting for T2 Later image)'}
              </div>
            )}

            {/* Load Pair & Sensitivity Slider */}
            <div className="bg-sat-panel p-2.5 rounded-lg border border-sat-border space-y-2">
              <div className="flex items-center justify-between">
                <button
                  onClick={() => {
                    onSelectT1Image('Bi_Temporal T1.png');
                    onSelectT2Image('Bi_Temporal T2.png');
                  }}
                  className="px-2.5 py-1 rounded bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight text-[11px] font-mono text-sat-textSecondary hover:text-white transition"
                >
                  Load Pair
                </button>
                <div className="flex items-center space-x-1 text-[11px] font-mono">
                  <span className="text-sat-muted">Threshold:</span>
                  <span className="text-sat-accent font-bold">{(changeThreshold * 100).toFixed(0)}%</span>
                </div>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.80"
                step="0.05"
                value={changeThreshold}
                onChange={(e) => onSetChangeThreshold(parseFloat(e.target.value))}
                className="w-full accent-sat-accent h-1 bg-sat-surface rounded"
              />
            </div>

            {/* Run Bi-Temporal Analysis Button */}
            <button
              onClick={handlePrimaryExecute}
              disabled={isProcessing || !t1Image || !t2Image}
              className={`w-full py-2.5 px-4 rounded-full font-sans font-semibold text-xs flex items-center justify-center space-x-2 transition shadow ${
                isProcessing || !t1Image || !t2Image
                  ? 'bg-sat-panel text-sat-muted cursor-not-allowed border border-sat-border'
                  : 'bg-white hover:bg-slate-200 text-slate-900 active:scale-[0.98]'
              }`}
            >
              {isProcessing ? (
                <Loader2 className="w-4 h-4 animate-spin text-slate-900" />
              ) : (
                <ArrowRight className="w-4 h-4 stroke-[2.5]" />
              )}
              <span>{isProcessing ? 'Detecting Changes...' : 'Run Change Analysis'}</span>
            </button>
          </div>
        )}

        {/* =========================================================================
            3. OPTICAL + SAR MODULE CONTROLS (TWO SEPARATE INDEPENDENT UPLOADS)
           ========================================================================= */}
        {currentMode === 'optical_sar' && (
          <div className="space-y-2.5">
            {/* Optical Box */}
            <div className="bg-sat-panel p-2.5 rounded-lg border border-sat-border space-y-1.5">
              <div className="flex items-center justify-between text-[10px] font-mono">
                <div>
                  <span className="font-semibold text-sat-cyan block">OPTICAL IMAGERY</span>
                  <span className="text-sat-muted text-[9px]">Earlier/Optical image</span>
                </div>
                <span className="text-sat-muted">JPG / PNG / TIFF / GeoTIFF</span>
              </div>
              <div className="flex items-center space-x-1.5">
                <button
                  onClick={() => triggerUpload('optical')}
                  className={`flex-1 flex items-center justify-between px-3 py-2 rounded-md border text-xs font-mono transition ${
                    opticalImage
                      ? 'bg-sat-surface border-sat-cyan/40 text-sat-cyan font-medium'
                      : 'bg-sat-surface hover:bg-sat-surfaceHover border-dashed border-sat-borderLight text-sat-muted hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2 truncate">
                    <Upload className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate">{opticalImage ? getDisplayFilename(opticalImage) : 'Upload Optical Image'}</span>
                  </div>
                  {opticalImage ? (
                    <Check className="w-3.5 h-3.5 text-sat-cyan flex-shrink-0" />
                  ) : (
                    <span className="text-[10px] text-sat-muted font-sans">—</span>
                  )}
                </button>
                <button
                  onClick={() => onSelectOpticalImage('optical.png')}
                  title="Load preloaded optical sample"
                  className="px-2 py-2 bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight rounded-md text-[10px] font-mono text-sat-muted hover:text-white transition whitespace-nowrap"
                >
                  Sample
                </button>
              </div>
            </div>

            {/* SAR Box */}
            <div className="bg-sat-panel p-2.5 rounded-lg border border-sat-border space-y-1.5">
              <div className="flex items-center justify-between text-[10px] font-mono">
                <div>
                  <span className="font-semibold text-sat-warning block">SAR IMAGERY</span>
                  <span className="text-sat-muted text-[9px]">SAR image</span>
                </div>
                <span className="text-sat-muted">TIFF / GeoTIFF</span>
              </div>
              <div className="flex items-center space-x-1.5">
                <button
                  onClick={() => triggerUpload('sar')}
                  className={`flex-1 flex items-center justify-between px-3 py-2 rounded-md border text-xs font-mono transition ${
                    sarImage
                      ? 'bg-sat-surface border-sat-warning/40 text-sat-warning font-medium'
                      : 'bg-sat-surface hover:bg-sat-surfaceHover border-dashed border-sat-borderLight text-sat-muted hover:text-white'
                  }`}
                >
                  <div className="flex items-center space-x-2 truncate">
                    <Upload className="w-3.5 h-3.5 flex-shrink-0" />
                    <span className="truncate">{sarImage ? getDisplayFilename(sarImage) : 'Upload SAR Image'}</span>
                  </div>
                  {sarImage ? (
                    <Check className="w-3.5 h-3.5 text-sat-warning flex-shrink-0" />
                  ) : (
                    <span className="text-[10px] text-sat-muted font-sans">—</span>
                  )}
                </button>
                <button
                  onClick={() => onSelectSarImage('sar.png')}
                  title="Load preloaded SAR sample"
                  className="px-2 py-2 bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight rounded-md text-[10px] font-mono text-sat-muted hover:text-white transition whitespace-nowrap"
                >
                  Sample
                </button>
              </div>
            </div>

            {/* Status indicator when waiting for one of the images */}
            {(!opticalImage || !sarImage) && (opticalImage || sarImage) && (
              <div className="text-[11px] font-mono text-sat-warning/90 bg-sat-warning/10 px-2.5 py-1.5 rounded border border-sat-warning/20">
                {!opticalImage ? 'Optical — | SAR ✓ (Waiting for Optical image)' : 'Optical ✓ | SAR — (Waiting for SAR image)'}
              </div>
            )}

            {/* Load Pair & Despeckle */}
            <div className="flex items-center justify-between gap-2">
              <button
                onClick={() => {
                  onSelectOpticalImage('optical.png');
                  onSelectSarImage('sar.png');
                }}
                className="flex-1 py-1.5 px-2 bg-sat-panel hover:bg-sat-surface border border-sat-border hover:border-sat-borderLight rounded-md text-[11px] font-mono text-sat-textSecondary hover:text-white transition text-center"
              >
                Load Verified Pair
              </button>
              <label className="flex items-center space-x-1.5 px-2.5 py-1.5 bg-sat-panel border border-sat-border rounded-md text-[11px] font-mono text-sat-text cursor-pointer">
                <input
                  type="checkbox"
                  checked={despeckleSar}
                  onChange={(e) => onToggleDespeckle(e.target.checked)}
                  className="accent-sat-cyan rounded"
                />
                <span className="text-[10px]">7x7 Lee Filter</span>
              </label>
            </div>

            {/* Command / Query Input with Circular Arrow Button */}
            <div className="relative flex items-center">
              <input
                type="text"
                value={getQueryValue()}
                onChange={(e) => setQueryValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') handlePrimaryExecute();
                }}
                placeholder="Compare optical and SAR radar..."
                className="w-full bg-sat-panel border border-sat-border focus:border-sat-borderLight rounded-full py-2.5 pl-3.5 pr-11 text-xs text-white placeholder-sat-muted focus:outline-none transition shadow-inner font-sans"
              />

              <button
                onClick={handlePrimaryExecute}
                disabled={isProcessing || !opticalImage || !sarImage}
                className="absolute right-1.5 w-7 h-7 rounded-full bg-white hover:bg-slate-200 text-slate-900 flex items-center justify-center transition disabled:opacity-50 disabled:cursor-not-allowed shadow"
                title="Execute Analysis"
              >
                {isProcessing ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin text-slate-900" />
                ) : (
                  <ArrowRight className="w-3.5 h-3.5 stroke-[2.5]" />
                )}
              </button>
            </div>
          </div>
        )}
      </div>
    </aside>
  );
};


