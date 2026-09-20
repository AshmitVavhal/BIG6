import React, { useState, useRef } from 'react';
import {
  ZoomIn,
  ZoomOut,
  Maximize2,
  Minimize2,
  RotateCcw,
  Download,
  Crosshair,
  Eye,
  EyeOff,
  Droplets,
  Building2,
  Radio,
  Layers,
  CloudUpload,
  Paperclip
} from 'lucide-react';
import type { DetectedRegion } from '../../types';

interface ImageViewerProps {
  imageUrl: string | null;
  annotatedImageUrl?: string | null;
  maskImageUrl?: string | null;
  detections?: DetectedRegion[];
  title?: string;
  onSelectRegion?: (region: DetectedRegion | null) => void;
  selectedRegion?: DetectedRegion | null;
  onUploadFile?: (file: File) => Promise<void>;
  onSelectScenario?: (scenario: 'flood' | 'urban' | 'sar' | 'change') => void;
  onDownloadDataset?: () => void;
}

export const ImageViewer: React.FC<ImageViewerProps> = ({
  imageUrl,
  annotatedImageUrl,
  maskImageUrl,
  detections = [],
  title = 'Target Detection • "building" - WGS84 / UTM 2D GIS',
  onSelectRegion,
  selectedRegion,
  onUploadFile,
  onSelectScenario,
  onDownloadDataset
}) => {
  const [zoom, setZoom] = useState<number>(1);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [dragStart, setDragStart] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [cursorPos, setCursorPos] = useState<{ x: number; y: number }>({ x: 1002, y: 243 });
  const [showAnnotations, setShowAnnotations] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [isDragOver, setIsDragOver] = useState<boolean>(false);

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
      const rawX = Math.round(e.clientX - rect.left);
      const rawY = Math.round(e.clientY - rect.top);
      setCursorPos({ x: rawX, y: rawY });
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

  // Drag & drop file upload on canvas
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file && onUploadFile) {
      await onUploadFile(file);
    }
  };

  const displayImage = showAnnotations && annotatedImageUrl ? annotatedImageUrl : imageUrl;

  return (
    <div
      ref={containerRef}
      className={`relative flex-1 bg-sat-bg gis-grid-bg flex flex-col overflow-hidden select-none transition-colors ${
        isDragOver ? 'ring-2 ring-sat-accent/50 bg-sat-panel/20' : ''
      }`}
      onWheel={handleWheel}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
      {/* ================= TOP FLOATING BADGE & ACTIONS ================= */}
      <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-20">
        {/* Left: Mode / Scene Status Pill */}
        <div className="bg-sat-panel/90 backdrop-blur border border-sat-border px-3 py-1.5 rounded-full flex items-center space-x-2 pointer-events-auto shadow-md">
          <span className="w-1.5 h-1.5 rounded-full bg-sat-accent" />
          <span className="text-[11px] font-mono text-sat-textSecondary font-medium">
            {title}
          </span>
        </div>

        {/* Right: Quick Action / Dataset Download */}
        <div className="bg-sat-panel/90 backdrop-blur border border-sat-border p-1 rounded-full flex items-center space-x-1 pointer-events-auto shadow-md">
          {annotatedImageUrl && (
            <button
              onClick={() => setShowAnnotations(!showAnnotations)}
              className={`p-1.5 rounded-full text-xs transition ${
                showAnnotations
                  ? 'bg-sat-accent/20 text-sat-accent'
                  : 'text-sat-muted hover:text-white'
              }`}
              title={showAnnotations ? 'Hide Annotation Overlays' : 'Show Annotation Overlays'}
            >
              {showAnnotations ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />}
            </button>
          )}

          <button
            onClick={onDownloadDataset || handleDownload}
            className="p-1.5 text-sat-muted hover:text-white hover:bg-sat-surface rounded-full transition"
            title="Download Scene / Dataset"
          >
            <Download className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* ================= MAIN IMAGERY CANVAS / EMPTY STATE ================= */}
      <div className="flex-1 flex items-center justify-center cursor-grab active:cursor-grabbing overflow-hidden">
        {displayImage ? (
          <div
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
              transition: isDragging ? 'none' : 'transform 0.1s ease-out'
            }}
            className="relative select-none shadow-2xl rounded overflow-hidden"
          >
            <img
              src={displayImage}
              alt="Satellite Scene"
              className="max-w-none pointer-events-none rounded"
              draggable={false}
            />

            {/* Polygon / Detection Overlay SVG */}
            {showAnnotations && detections.length > 0 && (
              <svg className="absolute inset-0 w-full h-full pointer-events-none">
                {detections.map((det) => {
                  const isSelected = selectedRegion?.id === det.id;
                  const bx = det.bounding_box;
                  return (
                    <g key={det.id}>
                      {/* Bounding Box Rectangle */}
                      {bx && bx.length === 4 && (
                        <rect
                          x={bx[0]}
                          y={bx[1]}
                          width={Math.max(1, bx[2] - bx[0])}
                          height={Math.max(1, bx[3] - bx[1])}
                          fill="none"
                          stroke={isSelected ? '#38bdf8' : '#ffffff'}
                          strokeWidth={isSelected ? '2.5' : '1.5'}
                          strokeDasharray={isSelected ? 'none' : '5 3'}
                        />
                      )}
                      {/* Polygon Segmentation Mask */}
                      {det.mask_polygon && det.mask_polygon.length > 2 && (
                        <polygon
                          points={det.mask_polygon.map((pt) => pt.join(',')).join(' ')}
                          fill={isSelected ? 'rgba(56, 189, 248, 0.40)' : 'rgba(255, 255, 255, 0.25)'}
                          stroke={isSelected ? '#38bdf8' : '#ffffff'}
                          strokeWidth={isSelected ? '2' : '1'}
                        />
                      )}
                    </g>
                  );
                })}
              </svg>
            )}
          </div>
        ) : (
          /* ================= EMPTY STATE: EARTH OBSERVATION STUDIO ================= */
          <div className="max-w-xl w-full mx-4 bg-sat-panel/90 backdrop-blur border border-sat-border rounded-2xl p-6 sm:p-7 shadow-2xl flex flex-col items-center text-center select-none pointer-events-auto">
            {/* Top Icon */}
            <div className="w-9 h-9 rounded-full bg-sat-surface border border-sat-border flex items-center justify-center text-sat-muted mb-3.5">
              <Paperclip className="w-4 h-4 text-sat-muted" />
            </div>

            {/* Title & Subtitle */}
            <h2 className="text-base sm:text-lg font-semibold text-white font-sans mb-1.5">
              Earth Observation Studio
            </h2>
            <p className="text-xs text-sat-muted leading-relaxed max-w-md mb-6 font-sans">
              Drop high-resolution GeoTIFF, Cartosat, or Sentinel imagery directly onto this canvas, or launch a verified ISRO scenario benchmark below.
            </p>

            {/* 2x2 Grid of Scenario Benchmark Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 w-full mb-6">
              {/* Card 1: Flood Inundation */}
              <button
                onClick={() => onSelectScenario?.('flood')}
                className="bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight p-3.5 rounded-xl text-left flex items-center justify-between group transition active:scale-[0.98]"
              >
                <div>
                  <h4 className="text-xs font-semibold text-white group-hover:text-sat-cyan transition">
                    Flood Inundation
                  </h4>
                  <span className="text-[11px] text-sat-muted">
                    Sentinel-2 NDWI waterlogging
                  </span>
                </div>
                <div className="w-6 h-6 rounded-md bg-sat-cyan/10 flex items-center justify-center text-sat-cyan flex-shrink-0 ml-2">
                  <Droplets className="w-3.5 h-3.5" />
                </div>
              </button>

              {/* Card 2: Urban Structures */}
              <button
                onClick={() => onSelectScenario?.('urban')}
                className="bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight p-3.5 rounded-xl text-left flex items-center justify-between group transition active:scale-[0.98]"
              >
                <div>
                  <h4 className="text-xs font-semibold text-white group-hover:text-sat-warning transition">
                    Urban Structures
                  </h4>
                  <span className="text-[11px] text-sat-muted">
                    Cartosat-2S building footprints
                  </span>
                </div>
                <div className="w-6 h-6 rounded-md bg-sat-warning/10 flex items-center justify-center text-sat-warning flex-shrink-0 ml-2">
                  <Building2 className="w-3.5 h-3.5" />
                </div>
              </button>

              {/* Card 3: Maritime Port SAR */}
              <button
                onClick={() => onSelectScenario?.('sar')}
                className="bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight p-3.5 rounded-xl text-left flex items-center justify-between group transition active:scale-[0.98]"
              >
                <div>
                  <h4 className="text-xs font-semibold text-white group-hover:text-sat-accent transition">
                    Maritime Port SAR
                  </h4>
                  <span className="text-[11px] text-sat-muted">
                    RISAT-1 C-band radar backscatter
                  </span>
                </div>
                <div className="w-6 h-6 rounded-md bg-sat-accent/10 flex items-center justify-center text-sat-accent flex-shrink-0 ml-2">
                  <Radio className="w-3.5 h-3.5" />
                </div>
              </button>

              {/* Card 4: Bi-Temporal Growth */}
              <button
                onClick={() => onSelectScenario?.('change')}
                className="bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border hover:border-sat-borderLight p-3.5 rounded-xl text-left flex items-center justify-between group transition active:scale-[0.98]"
              >
                <div>
                  <h4 className="text-xs font-semibold text-white group-hover:text-purple-400 transition">
                    Bi-Temporal Growth
                  </h4>
                  <span className="text-[11px] text-sat-muted">
                    Multi-temporal surface change
                  </span>
                </div>
                <div className="w-6 h-6 rounded-md bg-purple-500/10 flex items-center justify-center text-purple-400 flex-shrink-0 ml-2">
                  <Layers className="w-3.5 h-3.5" />
                </div>
              </button>
            </div>

            {/* Bottom Drag Helper */}
            <div className="flex items-center space-x-1.5 text-[11px] font-mono text-sat-muted">
              <CloudUpload className="w-3.5 h-3.5" />
              <span>Drag & drop GeoTIFF, TIFF, PNG, or JPG anywhere</span>
            </div>
          </div>
        )}
      </div>

      {/* ================= BOTTOM FLOATING RETICLE & ZOOM CONTROLS ================= */}
      {/* Bottom Left: Coordinates & Zoom HUD */}
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

      {/* Bottom Right: Floating Canvas Controls HUD */}
      <div className="absolute bottom-3 right-3 bg-sat-panel/90 backdrop-blur border border-sat-border p-1 rounded-full flex items-center space-x-1 pointer-events-auto shadow-md z-20">
        {/* Zoom In */}
        <button
          onClick={() => setZoom((prev) => Math.min(8.0, prev * 1.25))}
          className="p-1.5 text-sat-muted hover:text-white hover:bg-sat-surface rounded-full transition"
          title="Zoom In"
        >
          <ZoomIn className="w-3.5 h-3.5" />
        </button>

        {/* Fit to Screen */}
        <button
          onClick={handleFit}
          className="px-2 py-1 text-[10px] font-mono font-semibold text-sat-muted hover:text-white hover:bg-sat-surface rounded-full transition"
          title="Fit to Canvas"
        >
          Fit
        </button>

        {/* Reset View */}
        <button
          onClick={handleReset}
          className="p-1.5 text-sat-muted hover:text-white hover:bg-sat-surface rounded-full transition"
          title="Reset Pan & Zoom"
        >
          <RotateCcw className="w-3 h-3" />
        </button>

        {/* Fullscreen */}
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

