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
  Download
} from 'lucide-react';
import { ComparisonMode } from '../../types';

interface ComparisonViewerProps {
  imageAUrl: string | null;
  imageBUrl: string | null;
  labelA?: string;
  labelB?: string;
  title?: string;
}

export const ComparisonViewer: React.FC<ComparisonViewerProps> = ({
  imageAUrl,
  imageBUrl,
  labelA = 'T1 (BEFORE)',
  labelB = 'T2 / CHANGE MASK',
  title = 'BI-TEMPORAL MULTI-VIEW COMPARISON'
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

  const containerRef = useRef<HTMLDivElement>(null);
  const splitDragging = useRef<boolean>(false);

  // Stroboscopic Blink Timer
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

  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const factor = e.deltaY < 0 ? 1.15 : 0.85;
    setZoom((prev) => Math.min(8.0, Math.max(0.4, prev * factor)));
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0 && !splitDragging.current) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
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

  return (
    <div
      ref={containerRef}
      className="relative flex-1 bg-sat-darker gis-grid-bg flex flex-col overflow-hidden border-b border-sat-border select-none"
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
    >
      {/* Top Header & Mode Controls */}
      <div className="absolute top-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2 pointer-events-none z-20">
        {/* Title */}
        <div className="bg-sat-panel/90 backdrop-blur border border-sat-border px-3 py-1.5 rounded flex items-center space-x-2 pointer-events-auto">
          <span className="text-xs font-mono font-bold text-sat-text">{title}</span>
        </div>

        {/* Comparison Mode Selector Buttons */}
        <div className="bg-sat-panel/90 backdrop-blur border border-sat-border p-1 rounded flex items-center space-x-1 pointer-events-auto">
          {/* Split Mode */}
          <button
            onClick={() => setMode('split')}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-mono transition ${
              mode === 'split' ? 'bg-sat-accent text-sat-darker font-bold' : 'text-sat-muted hover:text-sat-text'
            }`}
          >
            <Split className="w-3.5 h-3.5" />
            <span>SPLIT</span>
          </button>

          {/* Dual Mode */}
          <button
            onClick={() => setMode('dual')}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-mono transition ${
              mode === 'dual' ? 'bg-sat-accent text-sat-darker font-bold' : 'text-sat-muted hover:text-sat-text'
            }`}
          >
            <Columns className="w-3.5 h-3.5" />
            <span>DUAL</span>
          </button>

          {/* Blink Mode */}
          <button
            onClick={() => {
              setMode('blink');
              setIsBlinking(true);
            }}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-mono transition ${
              mode === 'blink' ? 'bg-sat-accent text-sat-darker font-bold' : 'text-sat-muted hover:text-sat-text'
            }`}
          >
            <Eye className="w-3.5 h-3.5" />
            <span>BLINK</span>
          </button>

          {/* Blend Mode */}
          <button
            onClick={() => setMode('blend')}
            className={`flex items-center space-x-1 px-2.5 py-1 rounded text-xs font-mono transition ${
              mode === 'blend' ? 'bg-sat-accent text-sat-darker font-bold' : 'text-sat-muted hover:text-sat-text'
            }`}
          >
            <Sliders className="w-3.5 h-3.5" />
            <span>BLEND</span>
          </button>
        </div>

        {/* Zoom & Navigation toolbar */}
        <div className="bg-sat-panel/90 backdrop-blur border border-sat-border p-1 rounded flex items-center space-x-1 pointer-events-auto">
          <button
            onClick={() => setZoom((prev) => Math.min(8.0, prev * 1.25))}
            className="p-1.5 text-sat-muted hover:text-sat-text rounded transition"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={() => setZoom((prev) => Math.max(0.4, prev / 1.25))}
            className="p-1.5 text-sat-muted hover:text-sat-text rounded transition"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={handleReset}
            className="p-1.5 text-sat-muted hover:text-sat-text rounded transition"
            title="Reset Pan/Zoom"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Mode Specific Dynamic Toolbar Bar */}
      {mode === 'blink' && (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 bg-sat-panel/95 border border-sat-accent/40 px-3 py-1.5 rounded-full flex items-center space-x-3 text-xs font-mono z-20 shadow-lg">
          <button
            onClick={() => setIsBlinking(!isBlinking)}
            className="flex items-center space-x-1 px-2 py-0.5 bg-sat-accent text-sat-darker rounded font-bold"
          >
            {isBlinking ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
            <span>{isBlinking ? 'PAUSE' : 'BLINK'}</span>
          </button>
          <span className="text-sat-muted">FREQUENCY:</span>
          {[1, 2, 3, 5].map((hz) => (
            <button
              key={hz}
              onClick={() => setBlinkSpeedHz(hz)}
              className={`px-2 py-0.5 rounded ${
                blinkSpeedHz === hz ? 'bg-sat-surface text-sat-accent font-bold border border-sat-accent' : 'text-sat-muted'
              }`}
            >
              {hz}Hz
            </button>
          ))}
          <span className="text-sat-accent font-bold pl-2">ACTIVE: {blinkActiveImage === 'A' ? labelA : labelB}</span>
        </div>
      )}

      {mode === 'blend' && (
        <div className="absolute top-14 left-1/2 -translate-x-1/2 bg-sat-panel/95 border border-sat-accent/40 px-4 py-1.5 rounded-full flex items-center space-x-3 text-xs font-mono z-20 shadow-lg min-w-[320px]">
          <span className="text-sat-muted text-[11px]">{labelA}</span>
          <input
            type="range"
            min="0"
            max="100"
            value={blendAlpha}
            onChange={(e) => setBlendAlpha(parseInt(e.target.value))}
            className="flex-1 accent-sat-accent"
          />
          <span className="text-sat-muted text-[11px]">{labelB}</span>
          <span className="text-sat-accent font-bold">{blendAlpha}%</span>
        </div>
      )}

      {/* Main Imagery Comparison Viewport */}
      <div className="flex-1 flex items-center justify-center overflow-hidden cursor-grab active:cursor-grabbing">
        {!imageAUrl || !imageBUrl ? (
          <div className="text-center p-8 border border-dashed border-sat-border rounded-lg bg-sat-panel/50 max-w-md">
            <Columns className="w-10 h-10 text-sat-muted mx-auto mb-3 opacity-50" />
            <h3 className="text-sm font-mono font-bold text-sat-text mb-1">REQUIRES DUAL SATELLITE INPUTS</h3>
            <p className="text-xs text-sat-muted">
              Upload both T1/T2 scenes or Optical/SAR pairs from the sidebar to activate the interactive comparison suite.
            </p>
          </div>
        ) : (
          <div
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transition: isDragging ? 'none' : 'transform 0.1s ease-out'
            }}
            className="relative select-none"
          >
            {/* 1. SPLIT VIEW */}
            {mode === 'split' && (
              <div className="relative border border-sat-border shadow-2xl rounded overflow-hidden">
                {/* Image A (Base) */}
                <img src={imageAUrl} alt={labelA} className="max-w-none pointer-events-none" draggable={false} />

                {/* Image B (Clipped by splitPos) */}
                <div
                  style={{ width: `${splitPos}%` }}
                  className="absolute inset-0 overflow-hidden border-r-2 border-sat-accent pointer-events-none"
                >
                  <img
                    src={imageBUrl}
                    alt={labelB}
                    className="max-w-none pointer-events-none"
                    style={{ width: '100%', height: '100%', objectFit: 'none', objectPosition: 'left' }}
                    draggable={false}
                  />
                  {/* Label B Pill */}
                  <div className="absolute top-3 left-3 bg-sat-darker/90 px-2.5 py-1 rounded text-[11px] font-mono text-sat-accent border border-sat-accent font-bold">
                    {labelB}
                  </div>
                </div>

                {/* Label A Pill */}
                <div className="absolute top-3 right-3 bg-sat-darker/90 px-2.5 py-1 rounded text-[11px] font-mono text-sat-muted border border-sat-border font-bold">
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
              <div className="flex items-center space-x-4 border border-sat-border p-2 bg-sat-panel rounded shadow-2xl">
                <div className="relative">
                  <div className="absolute top-2 left-2 bg-sat-darker/90 px-2 py-0.5 rounded text-[10px] font-mono text-sat-text border border-sat-border font-bold">
                    {labelA}
                  </div>
                  <img src={imageAUrl} alt={labelA} className="max-w-none pointer-events-none rounded" draggable={false} />
                </div>

                <div className="relative">
                  <div className="absolute top-2 left-2 bg-sat-darker/90 px-2 py-0.5 rounded text-[10px] font-mono text-sat-accent border border-sat-accent font-bold">
                    {labelB}
                  </div>
                  <img src={imageBUrl} alt={labelB} className="max-w-none pointer-events-none rounded" draggable={false} />
                </div>
              </div>
            )}

            {/* 3. BLINK VIEW (Stroboscopic) */}
            {mode === 'blink' && (
              <div className="relative border border-sat-border shadow-2xl rounded overflow-hidden">
                <img
                  src={blinkActiveImage === 'A' ? imageAUrl : imageBUrl}
                  alt={blinkActiveImage === 'A' ? labelA : labelB}
                  className="max-w-none pointer-events-none"
                  draggable={false}
                />
                <div className="absolute bottom-3 left-3 bg-sat-darker/90 px-3 py-1 rounded text-xs font-mono text-sat-accent border border-sat-accent font-bold">
                  {blinkActiveImage === 'A' ? labelA : labelB}
                </div>
              </div>
            )}

            {/* 4. BLEND VIEW (Alpha Blended) */}
            {mode === 'blend' && (
              <div className="relative border border-sat-border shadow-2xl rounded overflow-hidden">
                <img src={imageAUrl} alt={labelA} className="max-w-none pointer-events-none" draggable={false} />
                <img
                  src={imageBUrl}
                  alt={labelB}
                  style={{ opacity: blendAlpha / 100 }}
                  className="max-w-none pointer-events-none absolute inset-0 transition-opacity"
                  draggable={false}
                />
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer Bar */}
      <div className="h-6 bg-sat-panel border-t border-sat-border flex items-center justify-between px-3 text-[10px] font-mono text-sat-muted z-10">
        <div className="flex items-center space-x-3">
          <span>MODE: {mode.toUpperCase()} COMPARISON</span>
          <span>•</span>
          <span>CO-REGISTRATION: PHASE-CORRELATED</span>
        </div>
      </div>
    </div>
  );
};
