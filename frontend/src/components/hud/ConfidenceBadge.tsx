import React from 'react';
import { ShieldCheck, AlertCircle } from 'lucide-react';

interface ConfidenceBadgeProps {
  confidence: number;
  confidenceType?: 'model' | 'heuristic' | string;
  modelName?: string;
}

export const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({
  confidence,
  confidenceType = 'heuristic',
  modelName
}) => {
  const pct = Math.round(confidence * 100);
  const isModel = confidenceType === 'model';

  return (
    <div className="inline-flex items-center space-x-1.5 px-2.5 py-1 rounded border text-[11px] font-mono bg-sat-surface border-sat-border">
      {isModel ? (
        <ShieldCheck className="w-3.5 h-3.5 text-sat-accent" />
      ) : (
        <AlertCircle className="w-3.5 h-3.5 text-sat-warning" />
      )}
      <span className="text-sat-muted">CONFIDENCE:</span>
      <span className={`font-bold ${isModel ? 'text-sat-accent' : 'text-sat-warning'}`}>
        {pct}%
      </span>
      <span className="text-sat-muted text-[10px]">
        ({isModel ? 'Model Raw Score' : 'Heuristic Confidence'})
      </span>
      {modelName && (
        <span className="text-[10px] text-sat-muted border-l border-sat-border pl-1.5 ml-1">
          {modelName}
        </span>
      )}
    </div>
  );
};
