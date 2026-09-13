import React, { useState } from 'react';
import { X, FileText, Download, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import { exportReport } from '../../services/api';
import { WorkspaceMode } from '../../types';

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
    <div className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4 z-50 select-none">
      <div className="bg-sat-panel border border-sat-border rounded-lg shadow-2xl max-w-lg w-full overflow-hidden font-mono">
        {/* Header */}
        <div className="p-4 border-b border-sat-border flex items-center justify-between bg-sat-darker">
          <div className="flex items-center space-x-2.5">
            <FileText className="w-5 h-5 text-sat-accent" />
            <div>
              <h2 className="text-sm font-bold text-sat-text tracking-wide">EXPORT INTELLIGENCE PDF REPORT</h2>
              <span className="text-[11px] text-sat-muted">ISRO SAC-26167 Mission Format</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-sat-surface rounded text-sat-muted hover:text-sat-text transition"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Body */}
        <div className="p-4 space-y-4 text-xs font-mono">
          <div className="space-y-1">
            <span className="text-sat-muted block text-[11px]">ACTIVE ANALYSIS WORKFLOW:</span>
            <span className="text-sat-accent font-bold uppercase">{currentMode.replace('_', ' ')}</span>
          </div>

          <div className="space-y-1">
            <label className="text-sat-muted block text-[11px]">ANALYST NOTES / OBSERVATIONS:</label>
            <textarea
              rows={3}
              value={analystNotes}
              onChange={(e) => setAnalystNotes(e.target.value)}
              className="w-full bg-sat-darker border border-sat-border focus:border-sat-accent rounded p-2 text-xs font-sans text-sat-text resize-none focus:outline-none"
            />
          </div>

          {error && (
            <div className="p-2.5 bg-sat-redDim border border-sat-red/40 rounded text-sat-red flex items-center space-x-2 text-[11px]">
              <AlertCircle className="w-4 h-4 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {downloadUrl && (
            <div className="p-3 bg-sat-accentDim border border-sat-accent/40 rounded space-y-2 text-[11px]">
              <div className="flex items-center space-x-2 text-sat-accent font-bold">
                <CheckCircle className="w-4 h-4" />
                <span>REPORT GENERATED SUCCESSFULLY!</span>
              </div>
              <p className="text-sat-text font-sans">
                Filename: <span className="font-mono text-sat-accent">{reportFileName}</span>
              </p>
              <a
                href={downloadUrl}
                download
                className="w-full py-2 px-3 bg-sat-accent hover:bg-sat-accentHover text-sat-darker font-bold rounded flex items-center justify-center space-x-2 transition"
              >
                <Download className="w-4 h-4" />
                <span>DOWNLOAD PDF REPORT</span>
              </a>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-3 bg-sat-darker border-t border-sat-border flex items-center justify-end space-x-2">
          <button
            onClick={onClose}
            className="px-3 py-1.5 bg-sat-surface hover:bg-sat-surface/80 border border-sat-border rounded text-xs text-sat-text transition"
          >
            Cancel
          </button>

          {!downloadUrl && (
            <button
              onClick={handleGenerate}
              disabled={isGenerating || !analysisData}
              className={`px-4 py-1.5 rounded font-bold text-xs flex items-center space-x-1.5 transition ${
                isGenerating || !analysisData
                  ? 'bg-sat-surface text-sat-muted cursor-not-allowed border border-sat-border'
                  : 'bg-sat-accent hover:bg-sat-accentHover text-sat-darker'
              }`}
            >
              {isGenerating ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FileText className="w-3.5 h-3.5" />}
              <span>{isGenerating ? 'Compiling PDF...' : 'Generate PDF'}</span>
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
