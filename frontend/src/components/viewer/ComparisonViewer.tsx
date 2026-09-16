import React, { useState, useEffect, useRef } from 'react';
import {
  Columns,
  Split,
  Eye,
  Sliders,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Play,
  Pause,
  Maximize2,
  Minimize2,
  Crosshair,
  Layers
} from 'lucide-react';
import type { ComparisonMode } from '../../types';

interface ComparisonViewerProps {
  imageAUrl: string | null;
  imageBUrl: string | null;
  labelA?: string;
  labelB?: string;
  title?: string;
  onUploadFile?: (file: File) => Promise<void>;
  onSelectScenario?: (scenario: 'flood' | 'urban' | 'sar' | 'change') => void;
  scenarioType?: 'change' | 'sar';
}

export const ComparisonViewer: React.FC<ComparisonViewerProps> = ({
  imageAUrl,
  imageBUrl,
  labelA = 'T1 (PRE-CHANGE)',
  labelB = 'T2 / CHANGE OVERLAY',
  title = 'Bi-Temporal Change Suite • WGS84 / UTM 2D GIS',
  onUploadFile,
  onSelectScenario,
  scenarioType = 'change'
}) => {
  const [mode, setMode] = useState<ComparisonMode>('split');
  const [splitPos, setSplitPos] = useState<number>(50); // percentage
  const [blendAlpha, setBlendAlpha] = useState<number>(50); // percentage
  const [isBlinking, setIsBlinking] = useState<boolean>(false);
  const [blinkSpeedHz, setBlinkSpeedHz] = useState<number>(2); // Hz
  const [blinkActiveImage, setBlinkActiveImage] = useState<'A' | 'B'>('A');

  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number }>({ x: 1002, y: 243 });
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement>(null);
  const splitDragging = useRef<boolean>(false);

  const [containerSize, setContainerSize] = useState<{ width: number; height: number }>({
    width: 1000,
    height: 700
  });

  const [imgA, setImgA] = useState<{ width: number; height: number }>({ width: 1024, height: 1024 });
  const [imgB, setImgB] = useState<{ width: number; height: number }>({ width: 1024, height: 1024 });

  // Track container dimensions with ResizeObserver
  useEffect(() => {
    if (!containerRef.current) return;
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        if (rect.width > 0 && rect.height > 0) {
          setContainerSize({ width: rect.width, height: rect.height });
        }
      }
    };
    updateSize();

    const ro = new ResizeObserver(() => {
      updateSize();
    });
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Track natural image dimensions
  useEffect(() => {
    if (!imageAUrl) return;
    const img = new Image();
    img.onload = () => {
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        setImgA({ width: img.naturalWidth, height: img.naturalHeight });
      }
    };
    img.src = imageAUrl;
    if (img.complete && img.naturalWidth > 0) {
      setImgA({ width: img.naturalWidth, height: img.naturalHeight });
    }
  }, [imageAUrl]);

  useEffect(() => {
    if (!imageBUrl) return;
    const img = new Image();
    img.onload = () => {
      if (img.naturalWidth > 0 && img.naturalHeight > 0) {
        setImgB({ width: img.naturalWidth, height: img.naturalHeight });
      }
    };
    img.src = imageBUrl;
    if (img.complete && img.naturalWidth > 0) {
      setImgB({ width: img.naturalWidth, height: img.naturalHeight });
    }
  }, [imageBUrl]);

  // Mode change handler
  const handleModeChange = (newMode: ComparisonMode) => {
    setMode(newMode);
    if (newMode === 'blink') {
      setIsBlinking(true);
    }
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  // Blink timer
  useEffect(() => {
    let interval: any = null;
    if (mode === 'blink' && isBlinking) {
      const intervalMs = Math.round(1000 / blinkSpeedHz);
      interval = setInterval(() => {
        setBlinkActiveImage((prev) => (prev === 'A' ? 'B' : 'A'));
      }, intervalMs);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [mode, isBlinking, blinkSpeedHz]);

  // Viewport and Dimension Calculations
  const paddingH = 32; // 16px left + 16px right margin
  const paddingV = 96; // 48px top toolbar + 48px bottom HUD
  const gap = 16; // divider gap between T1 and T2

  const availWidth = Math.max(100, containerSize.width - paddingH);
  const availHeight = Math.max(100, containerSize.height - paddingV);

  // 1. Dual Mode Calculations (Side-by-Side: divide available width)
  const halfWidth = Math.max(50, (availWidth - gap) / 2);
  const naturalWA = imgA.width || 1024;
  const naturalHA = imgA.height || 1024;
  const naturalWB = imgB.width || 1024;
  const naturalHB = imgB.height || 1024;

  const scaleDualA = Math.min(halfWidth / naturalWA, availHeight / naturalHA);
  const scaleDualB = Math.min(halfWidth / naturalWB, availHeight / naturalHB);
  const dualScale = Math.min(scaleDualA, scaleDualB);

  const dualWidthA = Math.round(naturalWA * dualScale);
  const dualHeightA = Math.round(naturalHA * dualScale);
  const dualWidthB = Math.round(naturalWB * dualScale);
  const dualHeightB = Math.round(naturalHB * dualScale);

  // 2. Single View Calculations (Split / Blink / Blend)
  const scaleSingle = Math.min(availWidth / naturalWA, availHeight / naturalHA);
  const singleWidth = Math.round(naturalWA * scaleSingle);
  const singleHeight = Math.round(naturalHA * scaleSingle);

  // Zoom & Pan Handlers
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    setZoom((prev) => Math.min(8.0, Math.max(0.2, prev * factor)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0 && !splitDragging.current) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const rawX = Math.round(e.clientX - rect.left);
      const rawY = Math.round(e.clientY - rect.top);
      setCursorPos({ x: rawX, y: rawY });
    }

    if (splitDragging.current && containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const x = Math.max(0, Math.min(rect.width, e.clientX - rect.left));
      setSplitPos(Math.round((x / rect.width) * 100));
      return;
    }

    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    splitDragging.current = false;
  };

  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setSplitPos(50);
    setBlendAlpha(50);
  };

  const handleFit = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;
    if (!document.fullscreenElement) {
      containerRef.current.requestFullscreen().catch(() => {});
      setIsFullscreen(true);
    } else {
      document.exitFullscreen().catch(() => {});
      setIsFullscreen(false);
    }
  };

  const hasBothImages = Boolean(imageAUrl && imageBUrl);

  return (
    <div
      ref={containerRef}
      className="relative flex-1 bg-sat-bg gis-grid-bg flex flex-col overflow-hidden select-none"
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* ================= TOP FLOATING BADGE & MODE SELECTOR ================= */}
      <div className="absolute top-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2 pointer-events-none z-20">
        {/* Title */}
        <div className="bg-sat-panel/90 backdrop-blur border border-sat-border px-3 py-1.5 rounded-full flex items-center space-x-2 pointer-events-auto shadow-md">
          <span className="w-1.5 h-1.5 rounded-full bg-sat-accent" />
          <span className="text-[11px] font-mono text-sat-textSecondary font-medium">{title}</span>
        </div>

        {/* Comparison Mode Selector Buttons */}
        <div className="bg-sat-panel/90 backdrop-blur border border-sat-border p-1 rounded-full flex items-center space-x-1 pointer-events-auto shadow-md">
          <button
            onClick={() => handleModeChange('split')}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded-full text-xs font-mono transition ${
              mode === 'split' ? 'bg-white text-slate-900 font-bold' : 'text-sat-muted hover:text-white'
            }`}
          >
            <Split className="w-3.5 h-3.5" />
            <span>Split</span>
          </button>

          <button
            onClick={() => handleModeChange('dual')}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded-full text-xs font-mono transition ${
              mode === 'dual' ? 'bg-white text-slate-900 font-bold' : 'text-sat-muted hover:text-white'
            }`}
          >
            <Columns className="w-3.5 h-3.5" />
            <span>Dual</span>
          </button>

          <button
            onClick={() => handleModeChange('blink')}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded-full text-xs font-mono transition ${
              mode === 'blink' ? 'bg-white text-slate-900 font-bold' : 'text-sat-muted hover:text-white'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>Blink</span>
          </button>

          <button
            onClick={() => handleModeChange('blend')}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded-full text-xs font-mono transition ${
              mode === 'blend' ? 'bg-white text-slate-900 font-bold' : 'text-sat-muted hover:text-white'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>Blend</span>
          </button>
        </div>
      </div>

      {/* Mode Specific Dynamic Toolbar Bar */}
      {mode === 'blink' && (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 bg-sat-panel/95 backdrop-blur border border-sat-border px-3.5 py-1.5 rounded-full flex items-center space-x-3 text-xs font-mono z-20 shadow-lg">
          <button
            onClick={() => setIsBlinking(!isBlinking)}
            className="flex items-center space-x-1 px-2.5 py-0.5 bg-sat-accent text-sat-darker rounded-full font-bold"
          >
            {isBlinking ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
            <span>{isBlinking ? 'Pause' : 'Blink'}</span>
          </button>
          <span className="text-sat-muted text-[11px]">RATE:</span>
          {[1, 2, 3, 5].map((hz) => (
            <button
              key={hz}
              onClick={() => setBlinkSpeedHz(hz)}
              className={`px-2 py-0.5 rounded-full text-[11px] ${
                blinkSpeedHz === hz ? 'bg-sat-surface text-sat-accent font-bold border border-sat-accent/40' : 'text-sat-muted hover:text-white'
              }`}
            >
              {hz}Hz
            </button>
          ))}
          <span className="text-sat-accent font-bold pl-2 text-[11px]">
            ACTIVE: {blinkActiveImage === 'A' ? labelA : labelB}
          </span>
        </div>
      )}

      {mode === 'blend' && (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 bg-sat-panel/95 backdrop-blur border border-sat-border px-4 py-1.5 rounded-full flex items-center space-x-3 text-xs font-mono z-20 shadow-lg min-w-[320px]">
          <span className="text-sat-muted text-[11px]">{labelA}</span>
          <input
            type="range"
            min="0"
            max="100"
            value={blendAlpha}
            onChange={(e) => setBlendAlpha(parseInt(e.target.value))}
            className="flex-1 accent-sat-accent h-1 bg-sat-surface rounded"
          />
          <span className="text-sat-muted text-[11px]">{labelB}</span>
          <span className="text-sat-accent font-bold">{blendAlpha}%</span>
        </div>
      )}

      {/* Main Imagery Comparison Viewport */}
      <div className="flex-1 flex items-center justify-center overflow-hidden cursor-grab active:cursor-grabbing">
        {!hasBothImages ? (
          <div className="max-w-md w-full mx-4 bg-sat-panel/90 backdrop-blur border border-sat-border rounded-2xl p-6 text-center shadow-2xl">
            <div className="w-10 h-10 rounded-full bg-sat-surface border border-sat-border flex items-center justify-center text-sat-muted mx-auto mb-3">
              <Layers className="w-5 h-5 text-sat-accent" />
            </div>
            <h3 className="text-sm font-semibold text-white mb-1">
              {scenarioType === 'sar' ? 'Optical + SAR Image Pair' : 'Multi-Temporal / Bi-Temporal Pair'}
            </h3>
            <p className="text-xs text-sat-muted mb-4 font-sans">
              Load both scenes from the sidebar or click below to launch the verified pair.
            </p>
            <button
              onClick={() => onSelectScenario?.(scenarioType)}
              className="px-4 py-2 bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border rounded-lg text-xs font-mono text-sat-accent transition"
            >
              {scenarioType === 'sar' ? 'Load Verified Pair' : 'Load Verified Scenario Pair'}
            </button>
          </div>
        ) : (
          <div
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transformOrigin: 'center center',
              transition: isDragging ? 'none' : 'transform 0.08s ease-out'
            }}
            className="relative select-none flex items-center justify-center"
          >
            {/* 1. SPLIT VIEW */}
            {mode === 'split' && (
              <div
                style={{ width: singleWidth, height: singleHeight }}
                className="relative border border-sat-border shadow-2xl rounded-lg overflow-hidden flex-shrink-0"
              >
                {/* Image A (Base) */}
                <img
                  src={imageAUrl!}
                  alt={labelA}
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  className="pointer-events-none block select-none"
                  draggable={false}
                  onLoad={(e) => {
                    const img = e.currentTarget;
                    if (img.naturalWidth && img.naturalHeight) {
                      setImgA({ width: img.naturalWidth, height: img.naturalHeight });
                    }
                  }}
                />

                {/* Image B (Clipped by splitPos) */}
                <div
                  style={{ width: `${splitPos}%` }}
                  className="absolute inset-0 overflow-hidden border-r-2 border-sat-accent pointer-events-none"
                >
                  <img
                    src={imageBUrl!}
                    alt={labelB}
                    style={{
                      width: singleWidth,
                      height: singleHeight,
                      maxWidth: 'none',
                      objectFit: 'contain'
                    }}
                    className="pointer-events-none block select-none"
                    draggable={false}
                    onLoad={(e) => {
                      const img = e.currentTarget;
                      if (img.naturalWidth && img.naturalHeight) {
                        setImgB({ width: img.naturalWidth, height: img.naturalHeight });
                      }
                    }}
                  />
                  {/* Label B Pill */}
                  <div className="absolute top-2.5 left-2.5 bg-sat-darker/90 backdrop-blur px-2.5 py-1 rounded-md text-[10px] font-mono text-sat-accent border border-sat-accent/40 font-bold shadow-md">
                    {labelB}
                  </div>
                </div>

                {/* Label A Pill */}
                <div className="absolute top-2.5 right-2.5 bg-sat-darker/90 backdrop-blur px-2.5 py-1 rounded-md text-[10px] font-mono text-sat-muted border border-sat-border font-bold shadow-md">
                  {labelA}
                </div>

                {/* Draggable Divider Handle */}
                <div
                  style={{ left: `calc(${splitPos}% - 12px)` }}
                  onMouseDown={(e) => {
                    e.stopPropagation();
                    splitDragging.current = true;
                  }}
                  className="absolute top-1/2 -translate-y-1/2 w-6 h-12 bg-sat-accent text-sat-darker rounded-full flex items-center justify-center cursor-ew-resize z-30 shadow-lg pointer-events-auto"
                >
                  <Split className="w-3.5 h-3.5" />
                </div>
              </div>
            )}

            {/* 2. DUAL VIEW (Side by Side) */}
            {mode === 'dual' && (
              <div
                style={{
                  width: dualWidthA + dualWidthB + gap,
                  height: Math.max(dualHeightA, dualHeightB)
                }}
                className="flex items-center justify-center space-x-4 select-none"
              >
                {/* T1 Box: Left Half */}
                <div
                  style={{
                    width: dualWidthA,
                    height: dualHeightA
                  }}
                  className="relative rounded-lg overflow-hidden border border-sat-border/80 bg-sat-panel shadow-2xl flex items-center justify-center flex-shrink-0"
                >
                  <div className="absolute top-2.5 left-2.5 bg-sat-darker/90 backdrop-blur px-2.5 py-1 rounded text-[10px] font-mono text-sat-text border border-sat-border font-bold z-10 shadow-md">
                    {labelA}
                  </div>
                  <img
                    src={imageAUrl!}
                    alt={labelA}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    className="pointer-events-none block select-none"
                    draggable={false}
                    onLoad={(e) => {
                      const img = e.currentTarget;
                      if (img.naturalWidth && img.naturalHeight) {
                        setImgA({ width: img.naturalWidth, height: img.naturalHeight });
                      }
                    }}
                  />
                </div>

                {/* Center Divider */}
                <div className="w-[2px] self-stretch bg-sat-borderLight/60 rounded-full flex-shrink-0 my-2" />

                {/* T2 Box: Right Half */}
                <div
                  style={{
                    width: dualWidthB,
                    height: dualHeightB
                  }}
                  className="relative rounded-lg overflow-hidden border border-sat-border/80 bg-sat-panel shadow-2xl flex items-center justify-center flex-shrink-0"
                >
                  <div className="absolute top-2.5 left-2.5 bg-sat-darker/90 backdrop-blur px-2.5 py-1 rounded text-[10px] font-mono text-sat-accent border border-sat-accent/40 font-bold z-10 shadow-md">
                    {labelB}
                  </div>
                  <img
                    src={imageBUrl!}
                    alt={labelB}
                    style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                    className="pointer-events-none block select-none"
                    draggable={false}
                    onLoad={(e) => {
                      const img = e.currentTarget;
                      if (img.naturalWidth && img.naturalHeight) {
                        setImgB({ width: img.naturalWidth, height: img.naturalHeight });
                      }
                    }}
                  />
                </div>
              </div>
            )}

            {/* 3. BLINK VIEW (Stroboscopic) */}
            {mode === 'blink' && (
              <div
                style={{ width: singleWidth, height: singleHeight }}
                className="relative border border-sat-border shadow-2xl rounded-lg overflow-hidden flex-shrink-0"
              >
                <img
                  src={blinkActiveImage === 'A' ? imageAUrl! : imageBUrl!}
                  alt={blinkActiveImage === 'A' ? labelA : labelB}
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  className="pointer-events-none block select-none"
                  draggable={false}
                  onLoad={(e) => {
                    const img = e.currentTarget;
                    if (img.naturalWidth && img.naturalHeight) {
                      if (blinkActiveImage === 'A') {
                        setImgA({ width: img.naturalWidth, height: img.naturalHeight });
                      } else {
                        setImgB({ width: img.naturalWidth, height: img.naturalHeight });
                      }
                    }
                  }}
                />
                <div className="absolute bottom-3 left-3 bg-sat-darker/90 backdrop-blur px-3 py-1 rounded-md text-xs font-mono text-sat-accent border border-sat-accent/40 font-bold shadow-md">
                  {blinkActiveImage === 'A' ? labelA : labelB}
                </div>
              </div>
            )}

            {/* 4. BLEND VIEW (Alpha Blended) */}
            {mode === 'blend' && (
              <div
                style={{ width: singleWidth, height: singleHeight }}
                className="relative border border-sat-border shadow-2xl rounded-lg overflow-hidden flex-shrink-0"
              >
                <img
                  src={imageAUrl!}
                  alt={labelA}
                  style={{ width: '100%', height: '100%', objectFit: 'contain' }}
                  className="pointer-events-none block select-none"
                  draggable={false}
                />
                <img
                  src={imageBUrl!}
                  alt={labelB}
                  style={{
                    opacity: blendAlpha / 100,
                    width: '100%',
                    height: '100%',
                    objectFit: 'contain'
                  }}
                  className="pointer-events-none absolute inset-0 transition-opacity select-none"
                  draggable={false}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* ================= BOTTOM FLOATING HUD CONTROLS ================= */}
      <div className="absolute bottom-3 left-3 bg-sat-panel/90 backdrop-blur border border-sat-border px-3 py-1.5 rounded-full flex items-center space-x-2.5 text-[11px] font-mono text-sat-muted pointer-events-auto shadow-md z-20">
        <Crosshair className="w-3 h-3 text-sat-muted" />
        <span>
          X: <span className="text-sat-text font-medium">{cursorPos.x}</span> Y:{' '}
          <span className="text-sat-text font-medium">{cursorPos.y}</span>
        </span>
        <span className="text-sat-borderLight">|</span>
        <span>
          ZOOM: <span className="text-sat-text font-bold">{(zoom * 100).toFixed(0)}%</span>
        </span>
      </div>

      <div className="absolute bottom-3 right-3 bg-sat-panel/90 backdrop-blur border border-sat-border p-1 rounded-full flex items-center space-x-1 pointer-events-auto shadow-md z-20">
        <button
          onClick={() => setZoom((prev) => Math.min(8.0, prev * 1.25))}
          className="p-1.5 text-sat-muted hover:text-white hover:bg-sat-surface rounded-full transition"
          title="Zoom In"
        >
          <ZoomIn className="w-3.5 h-3.5" />
        </button>

        <button
          onClick={handleFit}
          className="px-2 py-1 text-[10px] font-mono font-semibold text-sat-muted hover:text-white hover:bg-sat-surface rounded-full transition"
          title="Fit to Canvas"
        >
          Fit
        </button>

        <button
          onClick={handleReset}
          className="p-1.5 text-sat-muted hover:text-white hover:bg-sat-surface rounded-full transition"
          title="Reset Pan & Zoom"
        >
          <RotateCcw className="w-3 h-3" />
        </button>

        <button
          onClick={toggleFullscreen}
          className="p-1.5 text-sat-muted hover:text-white hover:bg-sat-surface rounded-full transition"
          title="Toggle Fullscreen"
        >
          {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
        </button>
      </div>
    </div>
  );
};
