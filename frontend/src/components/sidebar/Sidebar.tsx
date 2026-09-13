import React, { useRef } from 'react';
import {
  MessageSquare,
  Sparkles,
  Layers,
  Radio,
  Upload,
  Play,
  Cpu,
  Zap,
  AlertCircle
} from 'lucide-react';
import type { WorkspaceMode, SampleDataset, SemanticModelChoice } from '../../types';

interface SidebarProps {
  currentMode: WorkspaceMode;
  onSelectMode: (mode: WorkspaceMode) => void;
  samples: SampleDataset[];
  // VQA state
  vqaImage: string | null;
  vqaQuestion: string;
  onSetVqaQuestion: (q: string) => void;
  onSelectVqaImage: (path: string) => void;
  onRunVQA: () => void;
  // Highlight state
  highlightImage: string | null;
  highlightPrompt: string;
  onSetHighlightPrompt: (p: string) => void;
  onSelectHighlightImage: (path: string) => void;
  highlightThreshold: number;
  onSetHighlightThreshold: (t: number) => void;
  useMask2Former: boolean;
  onToggleMask2Former: (v: boolean) => void;
  onRunHighlight: () => void;
  // Bi-Temporal state
  t1Image: string | null;
  t2Image: string | null;
  onSelectT1Image: (path: string) => void;
  onSelectT2Image: (path: string) => void;
  changeThreshold: number;
  onSetChangeThreshold: (t: number) => void;
  onRunChange: () => void;
  // Optical + SAR state
  opticalImage: string | null;
  sarImage: string | null;
  onSelectOpticalImage: (path: string) => void;
  onSelectSarImage: (path: string) => void;
  opticalSarQuestion: string;
  onSetOpticalSarQuestion: (q: string) => void;
  despeckleSar: boolean;
  onToggleDespeckle: (v: boolean) => void;
  onRunOpticalSar: () => void;
  // General upload
  onUploadFile: (file: File, targetSlot?: string) => Promise<void>;
  isProcessing: boolean;
  // Semantic reasoning selection
  semanticModel: SemanticModelChoice;
  onSetSemanticModel: (m: SemanticModelChoice) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  currentMode,
  onSelectMode,
  samples,
  vqaImage,
  vqaQuestion,
  onSetVqaQuestion,
  onSelectVqaImage,
  onRunVQA,
  highlightImage,
  highlightPrompt,
  onSetHighlightPrompt,
  onSelectHighlightImage,
  highlightThreshold,
  onSetHighlightThreshold,
  useMask2Former,
  onToggleMask2Former,
  onRunHighlight,
  t1Image,
  t2Image,
  onSelectT1Image,
  onSelectT2Image,
  changeThreshold,
  onSetChangeThreshold,
  onRunChange,
  opticalImage,
  sarImage,
  onSelectOpticalImage,
  onSelectSarImage,
  opticalSarQuestion,
  onSetOpticalSarQuestion,
  despeckleSar,
  onToggleDespeckle,
  onRunOpticalSar,
  onUploadFile,
  isProcessing,
  semanticModel,
  onSetSemanticModel
}) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeUploadSlot = useRef<string>('default');

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

  const vqaSuggestions = [
    'What is visible in this image?',
    'Describe the major land-cover types.',
    'Are there any buildings or structures?',
    'Is there water or a drainage corridor?',
    'Are there signs of construction?',
    'Describe the vegetation cover.'
  ];

  const highlightSuggestions = [
    'building',
    'road',
    'water body',
    'vehicle',
    'aircraft',
    'ship',
    'forest',
    'solar panel'
  ];

  return (
    <aside className="w-80 md:w-96 bg-sat-panel border-r border-sat-border flex flex-col h-[calc(100vh-3.5rem)] z-20 select-none">
      <input
        type="file"
        ref={fileInputRef}
        onChange={handleFileChange}
        className="hidden"
        accept=".jpg,.jpeg,.png,.tif,.tiff,.geotiff"
      />

      {/* Mode Navigation Tabs */}
      <div className="grid grid-cols-2 p-2 gap-1.5 border-b border-sat-border bg-sat-darker">
        <button
          onClick={() => onSelectMode('vqa')}
          className={`flex items-center space-x-1.5 px-3 py-2 rounded text-xs font-mono font-medium transition ${
            currentMode === 'vqa'
              ? 'bg-sat-accent text-sat-darker font-bold shadow'
              : 'bg-sat-surface text-sat-muted hover:text-sat-text hover:bg-sat-surface/80 border border-sat-border'
          }`}
        >
          <MessageSquare className="w-3.5 h-3.5" />
          <span>VQA</span>
        </button>

        <button
          onClick={() => onSelectMode('highlight')}
          className={`flex items-center space-x-1.5 px-3 py-2 rounded text-xs font-mono font-medium transition ${
            currentMode === 'highlight'
              ? 'bg-sat-accent text-sat-darker font-bold shadow'
              : 'bg-sat-surface text-sat-muted hover:text-sat-text hover:bg-sat-surface/80 border border-sat-border'
          }`}
        >
          <Sparkles className="w-3.5 h-3.5" />
          <span>HIGHLIGHT</span>
        </button>

        <button
          onClick={() => onSelectMode('bitemporal')}
          className={`flex items-center space-x-1.5 px-3 py-2 rounded text-xs font-mono font-medium transition ${
            currentMode === 'bitemporal'
              ? 'bg-sat-accent text-sat-darker font-bold shadow'
              : 'bg-sat-surface text-sat-muted hover:text-sat-text hover:bg-sat-surface/80 border border-sat-border'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>BI-TEMPORAL</span>
        </button>

        <button
          onClick={() => onSelectMode('optical_sar')}
          className={`flex items-center space-x-1.5 px-3 py-2 rounded text-xs font-mono font-medium transition ${
            currentMode === 'optical_sar'
              ? 'bg-sat-accent text-sat-darker font-bold shadow'
              : 'bg-sat-surface text-sat-muted hover:text-sat-text hover:bg-sat-surface/80 border border-sat-border'
          }`}
        >
          <Radio className="w-3.5 h-3.5" />
          <span>OPTICAL+SAR</span>
        </button>
      </div>

      {/* Mode Content Scrollable Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {/* AI Reasoning Selector */}
        <div className="bg-sat-darker p-2.5 rounded border border-sat-border space-y-2 font-mono">
          <div className="flex items-center justify-between text-[11px]">
            <span className="font-bold text-sat-muted flex items-center space-x-1.5">
              <Cpu className="w-3.5 h-3.5 text-sat-accent" />
              <span>HYBRID REASONING PIPELINE</span>
            </span>
            <span className="text-[10px] text-sat-cyan font-bold flex items-center space-x-1">
              <Sparkles className="w-3 h-3" />
              <span>GEOCHAT + GEMINI</span>
            </span>
          </div>
          <div className="p-2 rounded bg-sat-surface/60 border border-sat-border text-[11px] space-y-1.5">
            <div className="flex items-center justify-between text-sat-text text-[10px]">
              <span className="text-sat-muted">VLM Specialist:</span>
              <span className="text-sat-accent font-bold">GeoChat-7B</span>
            </div>
            <div className="flex items-center justify-between text-sat-text text-[10px]">
              <span className="text-sat-muted">Multimodal Reasoning:</span>
              <span className="text-sat-cyan font-bold">Google Gemini</span>
            </div>
            <div className="flex items-center justify-between text-sat-text text-[10px]">
              <span className="text-sat-muted">Quantitative CV:</span>
              <span className="text-sat-warning font-bold">LAE-DINO • Mask2Former • ChangeMamba</span>
            </div>
          </div>
        </div>

        {/* =========================================================================
            1. VQA WORKSPACE SIDEBAR
           ========================================================================= */}
        {currentMode === 'vqa' && (
          <div className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-mono font-semibold text-sat-muted flex items-center justify-between">
                <span>INPUT SATELLITE IMAGE</span>
                <span className="text-[10px] text-sat-accent">JPG, PNG, GeoTIFF</span>
              </label>

              <div className="flex space-x-2">
                <button
                  onClick={() => triggerUpload('vqa')}
                  className="flex-1 flex items-center justify-center space-x-1.5 py-2.5 px-3 bg-sat-surface hover:bg-sat-surface/80 border border-dashed border-sat-borderLight hover:border-sat-accent rounded text-xs font-mono text-sat-text transition"
                >
                  <Upload className="w-3.5 h-3.5 text-sat-accent" />
                  <span>{vqaImage ? vqaImage.split('_').slice(-1)[0] : 'Upload Satellite File'}</span>
                </button>
              </div>

              {/* Sample Selector */}
              {samples.length > 0 && (
                <div className="mt-2">
                  <span className="text-[10px] font-mono text-sat-muted block mb-1">PRELOADED SAMPLES:</span>
                  <div className="flex flex-wrap gap-1.5">
                    {samples.map((s) => (
                      <button
                        key={s.filename}
                        onClick={() => onSelectVqaImage(s.filename)}
                        className={`text-[11px] font-mono px-2 py-1 rounded border transition ${
                          vqaImage === s.filename
                            ? 'bg-sat-accent/20 border-sat-accent text-sat-accent'
                            : 'bg-sat-darker border-sat-border text-sat-muted hover:text-sat-text'
                        }`}
                      >
                        {s.filename.replace('.png', '').replace('.tif', '')}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Query Input */}
            <div className="space-y-1.5">
              <label className="text-xs font-mono font-semibold text-sat-muted">QUESTION / QUERY</label>
              <textarea
                value={vqaQuestion}
                onChange={(e) => onSetVqaQuestion(e.target.value)}
                rows={3}
                placeholder="Ask about terrain, land-cover, buildings, water, roads..."
                className="w-full bg-sat-darker border border-sat-border focus:border-sat-accent rounded p-2.5 text-xs font-sans text-sat-text placeholder-sat-muted/50 focus:outline-none resize-none"
              />

              {/* Suggested Questions */}
              <div className="space-y-1 pt-1">
                <span className="text-[10px] font-mono text-sat-muted">SUGGESTED QUERIES:</span>
                <div className="space-y-1">
                  {vqaSuggestions.map((q, idx) => (
                    <button
                      key={idx}
                      onClick={() => onSetVqaQuestion(q)}
                      className="w-full text-left text-[11px] p-1.5 rounded bg-sat-surface hover:bg-sat-surface/80 border border-sat-border text-sat-text truncate transition"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Run Button */}
            <button
              onClick={onRunVQA}
              disabled={isProcessing || !vqaImage}
              className={`w-full py-2.5 px-4 rounded font-mono font-bold text-xs flex items-center justify-center space-x-2 transition shadow ${
                isProcessing || !vqaImage
                  ? 'bg-sat-surface text-sat-muted cursor-not-allowed border border-sat-border'
                  : 'bg-sat-accent hover:bg-sat-accentHover text-sat-darker'
              }`}
            >
              <Play className="w-4 h-4 fill-current" />
              <span>{isProcessing ? 'RUNNING GEOCHAT INFERENCE...' : 'RUN VQA REASONING'}</span>
            </button>
          </div>
        )}

        {/* =========================================================================
            2. IMAGE HIGHLIGHT WORKSPACE SIDEBAR
           ========================================================================= */}
        {currentMode === 'highlight' && (
          <div className="space-y-4">
            <div className="space-y-1.5">
              <label className="text-xs font-mono font-semibold text-sat-muted flex items-center justify-between">
                <span>TARGET SATELLITE IMAGE</span>
                <span className="text-[10px] text-sat-accent">LAE-DINO + Mask2Former</span>
              </label>

              <button
                onClick={() => triggerUpload('highlight')}
                className="w-full flex items-center justify-center space-x-1.5 py-2.5 px-3 bg-sat-surface hover:bg-sat-surface/80 border border-dashed border-sat-borderLight hover:border-sat-accent rounded text-xs font-mono text-sat-text transition"
              >
                <Upload className="w-3.5 h-3.5 text-sat-accent" />
                <span>{highlightImage ? highlightImage.split('_').slice(-1)[0] : 'Upload Satellite File'}</span>
              </button>

              {/* Sample Selector */}
              {samples.length > 0 && (
                <div className="mt-2">
                  <span className="text-[10px] font-mono text-sat-muted block mb-1">PRELOADED SAMPLES:</span>
                  <div className="flex flex-wrap gap-1.5">
                    {samples.map((s) => (
                      <button
                        key={s.filename}
                        onClick={() => onSelectHighlightImage(s.filename)}
                        className={`text-[11px] font-mono px-2 py-1 rounded border transition ${
                          highlightImage === s.filename
                            ? 'bg-sat-accent/20 border-sat-accent text-sat-accent'
                            : 'bg-sat-darker border-sat-border text-sat-muted hover:text-sat-text'
                        }`}
                      >
                        {s.filename.replace('.png', '').replace('.tif', '')}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Object Prompt */}
            <div className="space-y-1.5">
              <label className="text-xs font-mono font-semibold text-sat-muted">OBJECT DETECTION PROMPT</label>
              <input
                type="text"
                value={highlightPrompt}
                onChange={(e) => onSetHighlightPrompt(e.target.value)}
                placeholder="e.g. building, road, water, vehicle..."
                className="w-full bg-sat-darker border border-sat-border focus:border-sat-accent rounded p-2.5 text-xs font-sans text-sat-text focus:outline-none"
              />

              {/* Suggested Prompts */}
              <div className="pt-1">
                <span className="text-[10px] font-mono text-sat-muted block mb-1">PROMPT PRESETS:</span>
                <div className="flex flex-wrap gap-1">
                  {highlightSuggestions.map((p, idx) => (
                    <button
                      key={idx}
                      onClick={() => onSetHighlightPrompt(p)}
                      className="text-[10px] font-mono px-2 py-0.5 rounded bg-sat-surface hover:bg-sat-surface/80 border border-sat-border text-sat-text transition"
                    >
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            {/* Detection Threshold & Mask2Former Toggle */}
            <div className="space-y-3 bg-sat-darker p-3 rounded border border-sat-border">
              <div className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-sat-muted">Detection Confidence Threshold:</span>
                  <span className="text-sat-accent font-bold">{(highlightThreshold * 100).toFixed(0)}%</span>
                </div>
                <input
                  type="range"
                  min="0.10"
                  max="0.85"
                  step="0.05"
                  value={highlightThreshold}
                  onChange={(e) => onSetHighlightThreshold(parseFloat(e.target.value))}
                  className="w-full accent-sat-accent"
                />
              </div>

              <label className="flex items-center space-x-2 text-xs font-mono text-sat-text cursor-pointer">
                <input
                  type="checkbox"
                  checked={useMask2Former}
                  onChange={(e) => onToggleMask2Former(e.target.checked)}
                  className="accent-sat-accent"
                />
                <span>Enable Mask2Former Polygon Refinement</span>
              </label>
            </div>

            {/* Run Button */}
            <button
              onClick={onRunHighlight}
              disabled={isProcessing || !highlightImage || !highlightPrompt}
              className={`w-full py-2.5 px-4 rounded font-mono font-bold text-xs flex items-center justify-center space-x-2 transition shadow ${
                isProcessing || !highlightImage || !highlightPrompt
                  ? 'bg-sat-surface text-sat-muted cursor-not-allowed border border-sat-border'
                  : 'bg-sat-accent hover:bg-sat-accentHover text-sat-darker'
              }`}
            >
              <Play className="w-4 h-4 fill-current" />
              <span>{isProcessing ? 'DETECTING & SEGMENTING...' : 'RUN HIGHLIGHT PIPELINE'}</span>
            </button>
          </div>
        )}

        {/* =========================================================================
            3. BI-TEMPORAL CHANGE SIDEBAR
           ========================================================================= */}
        {currentMode === 'bitemporal' && (
          <div className="space-y-4">
            {/* T1 Before Upload */}
            <div className="space-y-1.5">
              <label className="text-xs font-mono font-semibold text-sat-muted flex items-center justify-between">
                <span>T1 / PRE-CHANGE BASELINE</span>
                <span className="text-[10px] text-sat-accent">Earlier Date</span>
              </label>
              <button
                onClick={() => triggerUpload('t1')}
                className="w-full flex items-center justify-center space-x-1.5 py-2.5 px-3 bg-sat-surface hover:bg-sat-surface/80 border border-dashed border-sat-borderLight hover:border-sat-accent rounded text-xs font-mono text-sat-text transition"
              >
                <Upload className="w-3.5 h-3.5 text-sat-accent" />
                <span>{t1Image ? t1Image.split('_').slice(-1)[0] : 'Upload T1 (Before)'}</span>
              </button>
            </div>

            {/* T2 After Upload */}
            <div className="space-y-1.5">
              <label className="text-xs font-mono font-semibold text-sat-muted flex items-center justify-between">
                <span>T2 / POST-CHANGE TARGET</span>
                <span className="text-[10px] text-sat-accent">Later Date</span>
              </label>
              <button
                onClick={() => triggerUpload('t2')}
                className="w-full flex items-center justify-center space-x-1.5 py-2.5 px-3 bg-sat-surface hover:bg-sat-surface/80 border border-dashed border-sat-borderLight hover:border-sat-accent rounded text-xs font-mono text-sat-text transition"
              >
                <Upload className="w-3.5 h-3.5 text-sat-accent" />
                <span>{t2Image ? t2Image.split('_').slice(-1)[0] : 'Upload T2 (After)'}</span>
              </button>
            </div>

            {/* Quick 1-Click Bi-Temporal Pair loader */}
            <div className="bg-sat-darker p-2.5 rounded border border-sat-border space-y-1.5">
              <span className="text-[10px] font-mono text-sat-muted block">PRESET BI-TEMPORAL SCENES:</span>
              <button
                onClick={() => {
                  onSelectT1Image('bitemporal_t1_2024.png');
                  onSelectT2Image('bitemporal_t2_2026.png');
                }}
                className="w-full text-left text-xs font-mono p-2 rounded bg-sat-surface hover:bg-sat-surface/80 border border-sat-border text-sat-text transition flex items-center justify-between"
              >
                <span>Urban Expansion (2024 vs 2026)</span>
                <span className="text-sat-accent text-[10px] font-bold">LOAD PAIR</span>
              </button>
            </div>

            {/* Change Sensitivity Threshold */}
            <div className="space-y-1 bg-sat-darker p-3 rounded border border-sat-border">
              <div className="flex justify-between text-xs font-mono">
                <span className="text-sat-muted">Change Decision Threshold:</span>
                <span className="text-sat-accent font-bold">{(changeThreshold * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.20"
                max="0.80"
                step="0.05"
                value={changeThreshold}
                onChange={(e) => onSetChangeThreshold(parseFloat(e.target.value))}
                className="w-full accent-sat-accent"
              />
            </div>

            {/* Run Button */}
            <button
              onClick={onRunChange}
              disabled={isProcessing || !t1Image || !t2Image}
              className={`w-full py-2.5 px-4 rounded font-mono font-bold text-xs flex items-center justify-center space-x-2 transition shadow ${
                isProcessing || !t1Image || !t2Image
                  ? 'bg-sat-surface text-sat-muted cursor-not-allowed border border-sat-border'
                  : 'bg-sat-accent hover:bg-sat-accentHover text-sat-darker'
              }`}
            >
              <Play className="w-4 h-4 fill-current" />
              <span>{isProcessing ? 'RUNNING CHANGEMAMBA...' : 'RUN CHANGE DETECTION'}</span>
            </button>
          </div>
        )}

        {/* =========================================================================
            4. OPTICAL + SAR FUSION SIDEBAR
           ========================================================================= */}
        {currentMode === 'optical_sar' && (
          <div className="space-y-4">
            {/* Optical Upload */}
            <div className="space-y-1.5">
              <label className="text-xs font-mono font-semibold text-sat-muted flex items-center justify-between">
                <span>OPTICAL SATELLITE IMAGE</span>
                <span className="text-[10px] text-sat-cyan">Visible RGB</span>
              </label>
              <button
                onClick={() => triggerUpload('optical')}
                className="w-full flex items-center justify-center space-x-1.5 py-2.5 px-3 bg-sat-surface hover:bg-sat-surface/80 border border-dashed border-sat-borderLight hover:border-sat-cyan rounded text-xs font-mono text-sat-text transition"
              >
                <Upload className="w-3.5 h-3.5 text-sat-cyan" />
                <span>{opticalImage ? opticalImage.split('_').slice(-1)[0] : 'Upload Optical RGB'}</span>
              </button>
            </div>

            {/* SAR Upload */}
            <div className="space-y-1.5">
              <label className="text-xs font-mono font-semibold text-sat-muted flex items-center justify-between">
                <span>SAR SATELLITE IMAGE</span>
                <span className="text-[10px] text-sat-warning">Sentinel-1 Radar</span>
              </label>
              <button
                onClick={() => triggerUpload('sar')}
                className="w-full flex items-center justify-center space-x-1.5 py-2.5 px-3 bg-sat-surface hover:bg-sat-surface/80 border border-dashed border-sat-borderLight hover:border-sat-warning rounded text-xs font-mono text-sat-text transition"
              >
                <Upload className="w-3.5 h-3.5 text-sat-warning" />
                <span>{sarImage ? sarImage.split('_').slice(-1)[0] : 'Upload SAR Image'}</span>
              </button>
            </div>

            {/* Quick 1-Click Optical+SAR Pair loader */}
            <div className="bg-sat-darker p-2.5 rounded border border-sat-border space-y-1.5">
              <span className="text-[10px] font-mono text-sat-muted block">PRESET MULTIMODAL PAIR:</span>
              <button
                onClick={() => {
                  onSelectOpticalImage('optical_multispectral.png');
                  onSelectSarImage('sar_sentinel1.png');
                }}
                className="w-full text-left text-xs font-mono p-2 rounded bg-sat-surface hover:bg-sat-surface/80 border border-sat-border text-sat-text transition flex items-center justify-between"
              >
                <span>Port & Coastal Runway (Opt+SAR)</span>
                <span className="text-sat-cyan text-[10px] font-bold">LOAD PAIR</span>
              </button>
            </div>

            {/* Despeckle Filter Toggle */}
            <div className="bg-sat-darker p-3 rounded border border-sat-border space-y-2">
              <label className="flex items-center space-x-2 text-xs font-mono text-sat-text cursor-pointer">
                <input
                  type="checkbox"
                  checked={despeckleSar}
                  onChange={(e) => onToggleDespeckle(e.target.checked)}
                  className="accent-sat-cyan"
                />
                <span>Apply 7x7 Lee Despeckle Filter</span>
              </label>
            </div>

            {/* Optional Cross-modal Question */}
            <div className="space-y-1.5">
              <label className="text-xs font-mono font-semibold text-sat-muted">CROSS-MODAL QUERY</label>
              <input
                type="text"
                value={opticalSarQuestion}
                onChange={(e) => onSetOpticalSarQuestion(e.target.value)}
                placeholder="What differences are visible between optical and SAR?"
                className="w-full bg-sat-darker border border-sat-border focus:border-sat-cyan rounded p-2 text-xs font-sans text-sat-text focus:outline-none"
              />
            </div>

            {/* Run Button */}
            <button
              onClick={onRunOpticalSar}
              disabled={isProcessing || !opticalImage || !sarImage}
              className={`w-full py-2.5 px-4 rounded font-mono font-bold text-xs flex items-center justify-center space-x-2 transition shadow ${
                isProcessing || !opticalImage || !sarImage
                  ? 'bg-sat-surface text-sat-muted cursor-not-allowed border border-sat-border'
                  : 'bg-sat-cyan hover:bg-sat-cyan/80 text-sat-darker font-bold'
              }`}
            >
              <Play className="w-4 h-4 fill-current" />
              <span>{isProcessing ? 'FUSING MULTIMODAL DATA...' : 'RUN FUSION ANALYSIS'}</span>
            </button>
          </div>
        )}
      </div>
    </aside>
  );
};
