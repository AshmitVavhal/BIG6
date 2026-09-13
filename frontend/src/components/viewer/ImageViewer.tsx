import React, { useState, useRef, useEffect } from 'react';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  RotateCcw,
  Download,
  Crosshair,
  Sliders,
  Eye,
  EyeOff
} from 'lucide-react';
import { DetectedRegion } from '../../types';

interface ImageViewerProps {
  imageUrl: string | null;
  annotatedImageUrl?: string | null;
  maskImageUrl?: string | null;
  detections?: DetectedRegion[];
  title?: string;
  onSelectRegion?: (region: DetectedRegion | null) => void;
  selectedRegion?: DetectedRegion | null;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  imageUrl,
  annotatedImageUrl,
  maskImageUrl,
  detections = [],
  title = 'SATELLITE SCENE VIEWER',
  onSelectRegion,
  selectedRegion
}) => {
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number } | null>(null);
  const [showAnnotations, setShowAnnotations] = useState<boolean>(true);
  const [showMasks, setShowMasks] = useState<boolean>(true);
  const [maskOpacity, setMaskOpacity] = useState<number>(0.65);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);

  const containerRef = useRef<HTMLDivElement>(null);

  // Zoom with scroll wheel
  const handleWheel = (e: React.WheelEvent) => {
    e.preventDefault();
    const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
    setZoom((prev) => Math.min(8.0, Math.max(0.4, prev * zoomFactor)));
  };

  // Pan with drag
  const handleMouseDown = (e: React.MouseEvent) => {
    if (e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (containerRef.current) {
      const rect = containerRef.current.getBoundingClientRect();
      const rawX = e.clientX - rect.left;
      const rawY = e.clientY - rect.top;
      // Convert to normalized / coordinate space
      setCursorPos({ x: Math.round(rawX), y: Math.round(rawY) });
    }

    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y
      });
    }
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleReset = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleFit = () => {
    setZoom(0.95);
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

  const handleDownload = () => {
    const targetUrl = showAnnotations && annotatedImageUrl ? annotatedImageUrl : imageUrl;
    if (!targetUrl) return;
    const a = document.createElement('a');
    a.href = targetUrl;
    a.download = `satquery_scene_${Date.now()}.png`;
    a.click();
  };

  const displayImage = showAnnotations && annotatedImageUrl ? annotatedImageUrl : imageUrl;

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
      {/* Top Controls Toolbar */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-10">
        {/* Left: Title & Coordinates */}
        <div className="bg-sat-panel/90 backdrop-blur border border-sat-border px-3 py-1.5 rounded flex items-center space-x-3 pointer-events-auto">
          <span className="text-xs font-mono font-bold text-sat-text tracking-wide">{title}</span>
          {cursorPos && (
            <div className="flex items-center space-x-1.5 text-[11px] font-mono text-sat-muted border-l border-sat-border pl-3">
              <Crosshair className="w-3 h-3 text-sat-accent" />
              <span>
                X: <span className="text-sat-text">{cursorPos.x}</span> Y: <span className="text-sat-text">{cursorPos.y}</span>
              </span>
              <span className="text-sat-borderLight">|</span>
              <span>
                ZOOM: <span className="text-sat-accent font-bold">{(zoom * 100).toFixed(0)}%</span>
              </span>
            </div>
          )}
        </div>

        {/* Right: Navigation & Layer Tools */}
        <div className="bg-sat-panel/90 backdrop-blur border border-sat-border p-1 rounded flex items-center space-x-1 pointer-events-auto">
          {/* Layer toggles if annotations exist */}
          {annotatedImageUrl && (
            <div className="flex items-center space-x-1 border-r border-sat-border pr-1 mr-1">
              <button
                onClick={() => setShowAnnotations(!showAnnotations)}
                className={`p-1.5 rounded text-xs transition ${
                  showAnnotations ? 'bg-sat-accent/20 text-sat-accent border border-sat-accent/40' : 'text-sat-muted hover:text-sat-text'
                }`}
                title={showAnnotations ? 'Hide Annotation Overlays' : 'Show Annotation Overlays'}
              >
                {showAnnotations ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
              </button>
            </div>
          )}

          {/* Zoom In */}
          <button
            onClick={() => setZoom((prev) => Math.min(8.0, prev * 1.25))}
            className="p-1.5 text-sat-muted hover:text-sat-text hover:bg-sat-surface rounded transition"
            title="Zoom In"
          >
            <ZoomIn className="w-3.5 h-3.5" />
          </button>

          {/* Zoom Out */}
          <button
            onClick={() => setZoom((prev) => Math.max(0.4, prev / 1.25))}
            className="p-1.5 text-sat-muted hover:text-sat-text hover:bg-sat-surface rounded transition"
            title="Zoom Out"
          >
            <ZoomOut className="w-3.5 h-3.5" />
          </button>

          {/* Reset View */}
          <button
            onClick={handleReset}
            className="p-1.5 text-sat-muted hover:text-sat-text hover:bg-sat-surface rounded transition"
            title="Reset Pan & Zoom (1:1)"
          >
            <RotateCcw className="w-3.5 h-3.5" />
          </button>

          {/* Fit to Screen */}
          <button
            onClick={handleFit}
            className="px-2 py-1 text-[11px] font-mono text-sat-muted hover:text-sat-text hover:bg-sat-surface rounded transition"
            title="Fit to Screen"
          >
            FIT
          </button>

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            className="p-1.5 text-sat-muted hover:text-sat-text hover:bg-sat-surface rounded transition"
            title="Toggle Fullscreen"
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>

          {/* Download */}
          <button
            onClick={handleDownload}
            disabled={!displayImage}
            className="p-1.5 text-sat-muted hover:text-sat-accent hover:bg-sat-surface rounded transition disabled:opacity-40"
            title="Download Image"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Main Imagery Canvas Area */}
      <div className="flex-1 flex items-center justify-center cursor-grab active:cursor-grabbing overflow-hidden">
        {displayImage ? (
          <div
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transition: isDragging ? 'none' : 'transform 0.1s ease-out'
            }}
            className="relative select-none shadow-2xl border border-sat-border"
          >
            <img
              src={displayImage}
              alt="Satellite Scene"
              className="max-w-none pointer-events-none rounded"
              draggable={false}
            />

            {/* Optional Interactive Bounding Box Polygon layer */}
            {showAnnotations && detections.length > 0 && (
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                {detections.map((det) => {
                  const [x1, y1, x2, y2] = det.bounding_box;
                  const isSelected = selectedRegion?.id === det.id;
                  return (
                    <g key={det.id}>
                      {det.mask_polygon && det.mask_polygon.length > 2 && (
                        <polygon
                          points={det.mask_polygon.map((pt) => pt.join(',')).join(' ')}
                          fill={isSelected ? 'rgba(6, 182, 212, 0.4)' : 'rgba(16, 185, 129, 0.25)'}
                          stroke={isSelected ? '#06b6d4' : '#10b981'}
                          strokeWidth={isSelected ? '2.5' : '1.5'}
                        />
                      )}
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
        ) : (
          <div className="text-center p-8 border border-dashed border-sat-border rounded-lg bg-sat-panel/50 max-w-md">
            <Crosshair className="w-10 h-10 text-sat-muted mx-auto mb-3 opacity-50" />
            <h3 className="text-sm font-mono font-bold text-sat-text mb-1">NO SATELLITE SCENE LOADED</h3>
            <p className="text-xs text-sat-muted">
              Select a preloaded satellite sample from the sidebar or upload a GeoTIFF/PNG/JPG image to start analysis.
            </p>
          </div>
        )}
      </div>

      {/* Bottom Status Reticle Coordinates bar */}
      <div className="h-6 bg-sat-panel border-t border-sat-border flex items-center justify-between px-3 text-[10px] font-mono text-sat-muted z-10">
        <div className="flex items-center space-x-3">
          <span>CANVAS: 2D GIS VIEWPORT</span>
          <span>•</span>
          <span>PROJECTION: ORTHORECTIFIED WGS84 / UTM</span>
        </div>
        <div className="flex items-center space-x-3">
          <span>INTERACTION: PAN (DRAG) | ZOOM (WHEEL)</span>
        </div>
      </div>
    </div>
  );
};
