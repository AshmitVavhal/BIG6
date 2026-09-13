import React from 'react';
import { AlertTriangle, Info } from 'lucide-react';

interface ModelTransparencyAlertProps {
  warning?: string | null;
  type?: 'warning' | 'info';
}

export const ModelTransparencyAlert: React.FC<ModelTransparencyAlertProps> = ({
  warning,
  type = 'warning'
}) => {
  if (!warning) return null;

  return (
    <div className={`p-3 rounded border text-xs font-mono flex items-start space-x-2.5 ${
      type === 'warning'
        ? 'bg-sat-warningDim border-sat-warning/40 text-sat-warning'
        : 'bg-sat-cyanDim border-sat-cyan/40 text-sat-cyan'
    }`}>
      {type === 'warning' ? (
        <AlertTriangle className="w-4 h-4 flex-shrink-0 mt-0.5" />
      ) : (
        <Info className="w-4 h-4 flex-shrink-0 mt-0.5" />
      )}
      <div className="space-y-0.5">
        <span className="font-bold text-[11px] uppercase tracking-wider block">
          MODEL TRANSPARENCY & VALIDATION NOTICE
        </span>
        <p className="text-sat-text text-[11px] leading-relaxed font-sans">{warning}</p>
      </div>
    </div>
  );
};
