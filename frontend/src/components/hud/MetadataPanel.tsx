import React from 'react';
import { Globe, Database, Compass, Calendar, Layers } from 'lucide-react';
import { GeoMetadata } from '../../types';

interface MetadataPanelProps {
  metadata?: GeoMetadata | null;
  title?: string;
}

export const MetadataPanel: React.FC<MetadataPanelProps> = ({
  metadata,
  title = 'GEOSPATIAL METADATA'
}) => {
  if (!metadata) {
    return (
      <div className="bg-sat-darker border border-sat-border rounded p-3 text-xs font-mono">
        <div className="flex items-center space-x-2 text-sat-muted mb-1">
          <Globe className="w-3.5 h-3.5 text-sat-muted" />
          <span className="font-bold text-sat-text">{title}</span>
        </div>
        <p className="text-sat-muted italic text-[11px]">Geospatial metadata unavailable for this image.</p>
      </div>
    );
  }

  return (
    <div className="bg-sat-darker border border-sat-border rounded p-3 text-xs font-mono space-y-2">
      <div className="flex items-center justify-between border-b border-sat-border pb-1.5">
        <div className="flex items-center space-x-2">
          <Globe className="w-3.5 h-3.5 text-sat-accent" />
          <span className="font-bold text-sat-text">{title}</span>
        </div>
        <span className={`text-[10px] px-1.5 py-0.5 rounded font-bold ${
          metadata.has_georeference ? 'bg-sat-accent/20 text-sat-accent border border-sat-accent/40' : 'bg-sat-surface text-sat-muted'
        }`}>
          {metadata.has_georeference ? 'GEO-REFERENCED' : 'STANDARD RASTER'}
        </span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-[11px]">
        {/* CRS */}
        <div className="bg-sat-panel p-1.5 rounded border border-sat-border">
          <span className="text-sat-muted block text-[10px]">CRS / PROJECTION:</span>
          <span className="text-sat-text font-bold truncate block" title={metadata.crs || 'Local Pixel Grid'}>
            {metadata.crs || 'LOCAL / EPSG:N/A'}
          </span>
        </div>

        {/* Dimensions */}
        <div className="bg-sat-panel p-1.5 rounded border border-sat-border">
          <span className="text-sat-muted block text-[10px]">DIMENSIONS:</span>
          <span className="text-sat-text font-bold">
            {metadata.width} × {metadata.height} px
          </span>
        </div>

        {/* Bands & Format */}
        <div className="bg-sat-panel p-1.5 rounded border border-sat-border">
          <span className="text-sat-muted block text-[10px]">BANDS & DRIVER:</span>
          <span className="text-sat-text font-bold">
            {metadata.count} Bands ({metadata.driver || 'GTiff'})
          </span>
        </div>

        {/* Spatial Resolution */}
        <div className="bg-sat-panel p-1.5 rounded border border-sat-border">
          <span className="text-sat-muted block text-[10px]">GSD RESOLUTION:</span>
          <span className="text-sat-text font-bold">
            {metadata.resolution ? `${metadata.resolution[0].toFixed(2)}m/px` : 'N/A'}
          </span>
        </div>
      </div>

      {/* Sensor / Mission Tag if available */}
      {metadata.sensor_info && (
        <div className="text-[11px] text-sat-muted flex items-center space-x-2 pt-0.5">
          <Database className="w-3 h-3 text-sat-accent" />
          <span>MISSION / SENSOR: <span className="text-sat-accent font-semibold">{metadata.sensor_info}</span></span>
          {metadata.acquisition_date && (
            <span>• ACQUIRED: <span className="text-sat-text">{metadata.acquisition_date}</span></span>
          )}
        </div>
      )}
    </div>
  );
};
