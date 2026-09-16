import React, { useState } from 'react';
import { X, FileText, Download, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { exportReport, resolveImageUrl } from '../../services/api';
import type { WorkspaceMode } from '../../types';

interface ExportReportModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentMode: WorkspaceMode;
  analysisData: any;
}

export const ExportReportModal: React.FC<ExportReportModalProps> = ({
  isOpen,
  onClose,
  currentMode,
  analysisData
}) => {
  const [analystNotes, setAnalystNotes] = useState<string>('Standard automated satellite assessment verified.');
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [reportFileName, setReportFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!isOpen) return null;

  const handleGenerate = async () => {
    if (!analysisData) {
      setError('No active analysis payload available. Please run an analysis workflow first.');
      return;
    }

    setIsGenerating(true);
    setError(null);
    try {
      const res = await exportReport(currentMode, analysisData, analystNotes);
      setDownloadUrl(res.download_url);
      setReportFileName(res.report_filename);
    } catch (e: any) {
      setError(e.message || 'Failed to generate PDF report.');
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4 z-50 select-none animate-in fade-in duration-150">
      <div className="bg-sat-panel border border-sat-border rounded-2xl shadow-2xl max-w-lg w-full overflow-hidden font-sans">
        {/* Header */}
        <div className="p-4 border-b border-sat-border flex items-center justify-between bg-sat-bg">
          <div className="flex items-center space-x-2.5">
            <div className="w-7 h-7 rounded-md bg-sat-surface border border-sat-border flex items-center justify-center text-sat-accent">
              <FileText className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-sm font-semibold text-white tracking-tight">EXPORT INTELLIGENCE DOSSIER</h2>
              <span className="text-[11px] text-sat-muted font-mono">ISRO SAC-26167 Intelligence Format</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-sat-surface rounded-lg text-sat-muted hover:text-white transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-5 space-y-4 text-xs font-sans">
          <div className="space-y-1">
            <span className="text-sat-muted block text-[11px] font-mono">ACTIVE WORKSPACE:</span>
            <span className="text-white font-semibold uppercase">{currentMode.replace('_', ' ')}</span>
          </div>

          <div className="space-y-1.5">
            <label className="text-sat-muted block text-[11px] font-mono">ANALYST NOTES / OBSERVATIONS:</label>
            <textarea
              rows={3}
              value={analystNotes}
              onChange={(e) => setAnalystNotes(e.target.value)}
              className="w-full bg-sat-surface border border-sat-border focus:border-sat-borderLight rounded-xl p-3 text-xs text-white placeholder-sat-muted resize-none focus:outline-none transition font-sans"
            />
          </div>

          {error && (
            <div className="p-3 bg-sat-redDim border border-sat-red/40 rounded-xl text-sat-red flex items-center space-x-2 text-xs">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {downloadUrl && (
            <div className="p-3.5 bg-sat-accentDim border border-sat-accent/40 rounded-xl space-y-2 text-xs">
              <div className="flex items-center space-x-2 text-sat-accent font-semibold">
                <CheckCircle className="w-4 h-4" />
                <span>REPORT GENERATED SUCCESSFULLY</span>
              </div>
              <p className="text-sat-text">
                Filename: <span className="font-mono text-sat-accent">{reportFileName}</span>
              </p>
              <a
                href={resolveImageUrl(downloadUrl) || downloadUrl}
                download
                target="_blank"
                rel="noopener noreferrer"
                className="w-full py-2 px-3 bg-sat-accent hover:bg-sat-accentHover text-sat-darker font-bold rounded-lg flex items-center justify-center space-x-2 transition shadow"
              >
                <Download className="w-4 h-4" />
                <span>DOWNLOAD DOSSIER PDF</span>
              </a>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 bg-sat-bg border-t border-sat-border flex items-center justify-end space-x-2 font-mono text-xs">
          <button
            onClick={onClose}
            className="px-3.5 py-1.5 bg-sat-surface hover:bg-sat-surfaceHover border border-sat-border rounded-lg text-sat-text transition"
          >
            Cancel
          </button>

          {!downloadUrl && (
            <button
              onClick={handleGenerate}
              disabled={isGenerating || !analysisData}
              className={`px-4 py-1.5 rounded-lg font-semibold flex items-center space-x-1.5 transition ${
                isGenerating || !analysisData
                  ? 'bg-sat-surface text-sat-muted cursor-not-allowed border border-sat-border'
                  : 'bg-white hover:bg-slate-200 text-slate-900 shadow'
              }`}
            >
              {isGenerating ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FileText className="w-3.5 h-3.5" />}
              <span>{isGenerating ? 'Compiling PDF...' : 'Generate Dossier'}</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};

