import React, { useState, useEffect } from 'react';
import { TopNavbar } from './components/layout/TopNavbar';
import { Sidebar } from './components/sidebar/Sidebar';
import { ImageViewer } from './components/viewer/ImageViewer';
import { ComparisonViewer } from './components/viewer/ComparisonViewer';
import { BenchmarkModal } from './components/modals/BenchmarkModal';
import { ExportReportModal } from './components/modals/ExportReportModal';

import type {
  WorkspaceMode,
  SemanticModelChoice,
  SystemStatus,
  SampleDataset,
  VQAResponse,
  HighlightResponse,
  ChangeDetectionResponse,
  OpticalSARResponse,
  DetectedRegion
} from './types';

import {
  fetchSystemStatus,
  fetchSamples,
  uploadImage,
  analyzeVQA,
  analyzeHighlight,
  analyzeChange,
  analyzeOpticalSAR,
  clearSystemCache,
  resolveImageUrl
} from './services/api';

export const App: React.FC = () => {
  const [currentMode, setCurrentMode] = useState<WorkspaceMode>('vqa');
  const [semanticModel, setSemanticModel] = useState<SemanticModelChoice>('geochat');
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [samples, setSamples] = useState<SampleDataset[]>([]);
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const [errorToast, setErrorToast] = useState<string | null>(null);

  // Modals
  const [showBenchmarks, setShowBenchmarks] = useState<boolean>(false);
  const [showExport, setShowExport] = useState<boolean>(false);

  // VQA State
  const [vqaImage, setVqaImage] = useState<string | null>(null);
  const [vqaQuestion, setVqaQuestion] = useState<string>('What is visible in this satellite image?');
  const [vqaResult, setVqaResult] = useState<VQAResponse | null>(null);

  // Highlight State
  const [highlightImage, setHighlightImage] = useState<string | null>(null);
  const [highlightPrompt, setHighlightPrompt] = useState<string>('building');
  const [highlightThreshold, setHighlightThreshold] = useState<number>(0.25);
  const [useMask2Former, setUseMask2Former] = useState<boolean>(true);
  const [highlightResult, setHighlightResult] = useState<HighlightResponse | null>(null);
  const [selectedRegion, setSelectedRegion] = useState<DetectedRegion | null>(null);

  // Bi-Temporal State
  const [t1Image, setT1Image] = useState<string | null>(null);
  const [t2Image, setT2Image] = useState<string | null>(null);
  const [changeThreshold, setChangeThreshold] = useState<number>(0.40);
  const [changeResult, setChangeResult] = useState<ChangeDetectionResponse | null>(null);

  // Optical + SAR State
  const [opticalImage, setOpticalImage] = useState<string | null>(null);
  const [sarImage, setSarImage] = useState<string | null>(null);
  const [opticalSarQuestion, setOpticalSarQuestion] = useState<string>('Compare optical reflectance and SAR radar backscatter.');
  const [despeckleSar, setDespeckleSar] = useState<boolean>(true);
  const [opticalSarResult, setOpticalSarResult] = useState<OpticalSARResponse | null>(null);

  // Poll system status every 5 seconds
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const s = await fetchSystemStatus();
        setSystemStatus(s);
      } catch (err) {
        console.error('System status query failed:', err);
      }
    };
    checkStatus();
    const interval = setInterval(checkStatus, 5000);
    return () => clearInterval(interval);
  }, []);

  // Load samples on init
  useEffect(() => {
    const loadSamples = async () => {
      try {
        const sList = await fetchSamples();
        setSamples(sList);
      } catch (err) {
        console.error('Failed to load sample datasets:', err);
      }
    };
    loadSamples();
  }, []);

  const handleUploadFile = async (file: File, targetSlot = 'default') => {
    try {
      setIsProcessing(true);
      const res = await uploadImage(file);
      if (currentMode === 'vqa' || targetSlot === 'vqa') {
        setVqaImage(res.filename);
      } else if (currentMode === 'highlight' || targetSlot === 'highlight') {
        setHighlightImage(res.filename);
      } else if (targetSlot === 't1') {
        setT1Image(res.filename);
      } else if (targetSlot === 't2') {
        setT2Image(res.filename);
      } else if (targetSlot === 'optical') {
        setOpticalImage(res.filename);
      } else if (targetSlot === 'sar') {
        setSarImage(res.filename);
      } else {
        // Fallback to active mode
        if (currentMode === 'bitemporal') {
          if (!t1Image) setT1Image(res.filename);
          else setT2Image(res.filename);
        } else if (currentMode === 'optical_sar') {
          if (!opticalImage) setOpticalImage(res.filename);
          else setSarImage(res.filename);
        }
      }
    } catch (err: any) {
      showError(err.message || 'File upload failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const showError = (msg: string) => {
    setErrorToast(msg);
    setTimeout(() => setErrorToast(null), 6000);
  };

  // Scenario quick launcher from empty state
  const handleSelectScenario = (scenario: 'flood' | 'urban' | 'sar' | 'change') => {
    if (scenario === 'flood') {
      setCurrentMode('vqa');
      setVqaImage('VQA1.png');
      setVqaQuestion('Describe visible geographical features and water bodies in this scene.');
    } else if (scenario === 'urban') {
      setCurrentMode('highlight');
      setHighlightPrompt('building');
      setHighlightImage('VQA2.png');
    } else if (scenario === 'sar') {
      setCurrentMode('optical_sar');
      setOpticalImage('optical.png');
      setSarImage('sar.png');
      setOpticalSarQuestion('Compare optical reflectance and SAR radar backscatter.');
    } else if (scenario === 'change') {
      setCurrentMode('bitemporal');
      setT1Image('Bi_Temporal T1.png');
      setT2Image('Bi_Temporal T2.png');
    }
  };

  // Download test dataset
  const handleDownloadDataset = () => {
    const downloadUrl = 'https://drive.usercontent.google.com/download?id=1lwVLG62RdiPA4cNFfyzT36HZ-IZ44suY&export=download&confirm=t';
    const link = document.createElement('a');
    link.href = downloadUrl;
    link.setAttribute('download', 'Dataset.zip');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Execution Handlers
  const handleRunVQA = async () => {
    if (!vqaImage) {
      showError('Please attach a satellite scene first.');
      return;
    }
    setIsProcessing(true);
    try {
      const res = await analyzeVQA(vqaImage, vqaQuestion, semanticModel);
      setVqaResult(res);
    } catch (err: any) {
      showError(err.message || 'VQA Analysis failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunHighlight = async () => {
    if (!highlightImage) {
      showError('Please attach a satellite scene first.');
      return;
    }
    if (!highlightPrompt) {
      showError('Please provide an object prompt.');
      return;
    }
    setIsProcessing(true);
    try {
      const res = await analyzeHighlight(highlightImage, highlightPrompt, highlightThreshold, useMask2Former, semanticModel);
      setHighlightResult(res);
    } catch (err: any) {
      showError(err.message || 'Highlight Analysis failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunChange = async () => {
    if (!t1Image || !t2Image) {
      showError('Please attach both T1 and T2 satellite scenes.');
      return;
    }
    setIsProcessing(true);
    try {
      const res = await analyzeChange(t1Image, t2Image, changeThreshold, true, semanticModel);
      setChangeResult(res);
    } catch (err: any) {
      showError(err.message || 'Change Detection failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleRunOpticalSar = async () => {
    if (!opticalImage || !sarImage) {
      showError('Please attach both Optical and SAR satellite images.');
      return;
    }
    setIsProcessing(true);
    try {
      const res = await analyzeOpticalSAR(opticalImage, sarImage, opticalSarQuestion, despeckleSar, semanticModel);
      setOpticalSarResult(res);
    } catch (err: any) {
      showError(err.message || 'Optical + SAR Fusion failed.');
    } finally {
      setIsProcessing(false);
    }
  };

  const handleReset = async () => {
    try {
      await clearSystemCache();
    } catch (e) {}
    setVqaResult(null);
    setHighlightResult(null);
    setChangeResult(null);
    setOpticalSarResult(null);
    setSelectedRegion(null);
  };

  // Active analysis data for export
  const getActiveAnalysisData = () => {
    if (currentMode === 'vqa') return vqaResult;
    if (currentMode === 'highlight') return highlightResult;
    if (currentMode === 'bitemporal') return changeResult;
    if (currentMode === 'optical_sar') return opticalSarResult;
    return null;
  };

  return (
    <div className="flex flex-col h-screen w-screen bg-sat-bg text-sat-text overflow-hidden font-sans select-none">
      {/* Top Navbar */}
      <TopNavbar
        systemStatus={systemStatus}
        currentMode={currentMode}
        onSelectMode={setCurrentMode}
        onReset={handleReset}
        onOpenBenchmarks={() => setShowBenchmarks(true)}
        onOpenExport={() => setShowExport(true)}
        isProcessing={isProcessing}
      />

      {/* Error Toast notification */}
      {errorToast && (
        <div className="absolute top-14 right-6 max-w-md bg-sat-red text-sat-darker font-mono font-bold px-4 py-2.5 rounded-lg shadow-2xl z-50 text-xs border border-white/20 animate-bounce">
          [ERROR]: {errorToast}
        </div>
      )}

      {/* Main Workstation Container */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Mission Feed Sidebar */}
        <Sidebar
          currentMode={currentMode}
          onSelectMode={setCurrentMode}
          samples={samples}
          // VQA
          vqaImage={vqaImage}
          vqaQuestion={vqaQuestion}
          onSetVqaQuestion={setVqaQuestion}
          onSelectVqaImage={setVqaImage}
          onRunVQA={handleRunVQA}
          vqaResult={vqaResult}
          // Highlight
          highlightImage={highlightImage}
          highlightPrompt={highlightPrompt}
          onSetHighlightPrompt={setHighlightPrompt}
          onSelectHighlightImage={setHighlightImage}
          highlightThreshold={highlightThreshold}
          onSetHighlightThreshold={setHighlightThreshold}
          useMask2Former={useMask2Former}
          onToggleMask2Former={setUseMask2Former}
          onRunHighlight={handleRunHighlight}
          highlightResult={highlightResult}
          selectedRegion={selectedRegion}
          onSelectRegion={setSelectedRegion}
          // Bi-Temporal
          t1Image={t1Image}
          t2Image={t2Image}
          onSelectT1Image={setT1Image}
          onSelectT2Image={setT2Image}
          changeThreshold={changeThreshold}
          onSetChangeThreshold={setChangeThreshold}
          onRunChange={handleRunChange}
          changeResult={changeResult}
          // Optical + SAR
          opticalImage={opticalImage}
          sarImage={sarImage}
          onSelectOpticalImage={setOpticalImage}
          onSelectSarImage={setSarImage}
          opticalSarQuestion={opticalSarQuestion}
          onSetOpticalSarQuestion={setOpticalSarQuestion}
          despeckleSar={despeckleSar}
          onToggleDespeckle={setDespeckleSar}
          onRunOpticalSar={handleRunOpticalSar}
          opticalSarResult={opticalSarResult}
          // General
          onUploadFile={handleUploadFile}
          isProcessing={isProcessing}
          // Semantic Model Selection
          semanticModel={semanticModel}
          onSetSemanticModel={setSemanticModel}
        />

        {/* Center/Right Dominant Canvas Area */}
        <div className="flex-1 flex flex-col overflow-hidden relative">
          {/* 1. VQA Viewport */}
          {currentMode === 'vqa' && (
            <ImageViewer
              imageUrl={resolveImageUrl(vqaResult?.image_url || vqaImage)}
              title={`VQA Intelligence • "${vqaQuestion.slice(0, 32)}..." - WGS84 / UTM 2D GIS`}
              onUploadFile={handleUploadFile}
              onSelectScenario={handleSelectScenario}
              onDownloadDataset={handleDownloadDataset}
            />
          )}

          {/* 2. Highlight Viewport */}
          {currentMode === 'highlight' && (
            <ImageViewer
              imageUrl={resolveImageUrl(highlightResult?.original_image_url || highlightImage)}
              annotatedImageUrl={resolveImageUrl(highlightResult?.annotated_image_url)}
              maskImageUrl={resolveImageUrl(highlightResult?.mask_image_url)}
              detections={highlightResult?.detections}
              selectedRegion={selectedRegion}
              onSelectRegion={setSelectedRegion}
              title={`Target Detection • "${highlightPrompt}" - WGS84 / UTM 2D GIS`}
              onUploadFile={handleUploadFile}
              onSelectScenario={handleSelectScenario}
              onDownloadDataset={handleDownloadDataset}
            />
          )}

          {/* 3. Bi-Temporal Change Viewport */}
          {currentMode === 'bitemporal' && (
            <ComparisonViewer
              imageAUrl={resolveImageUrl(changeResult?.t1_image_url || t1Image)}
              imageBUrl={resolveImageUrl(
                changeResult
                  ? (changeResult.overlay_url || changeResult.t2_image_url)
                  : t2Image
              )}
              labelA="T1 (PRE-CHANGE BASELINE)"
              labelB="T2 (POST-CHANGE / OVERLAY)"
              title="Bi-Temporal Change Suite • WGS84 / UTM 2D GIS"
              onUploadFile={handleUploadFile}
              onSelectScenario={handleSelectScenario}
              scenarioType="change"
            />
          )}

          {/* 4. Optical + SAR Fusion Viewport */}
          {currentMode === 'optical_sar' && (
            <ComparisonViewer
              imageAUrl={resolveImageUrl(opticalSarResult?.optical_image_url || opticalImage)}
              imageBUrl={resolveImageUrl(
                opticalSarResult
                  ? (opticalSarResult.fused_image_url || opticalSarResult.sar_image_url)
                  : sarImage
              )}
              labelA="OPTICAL RGB REFLECTANCE"
              labelB="SAR / RADAR FUSED COMPOSITE"
              title="Optical + SAR Fusion Suite • WGS84 / UTM 2D GIS"
              onUploadFile={handleUploadFile}
              onSelectScenario={handleSelectScenario}
              scenarioType="sar"
            />
          )}
        </div>
      </div>

      {/* Modals */}
      <BenchmarkModal isOpen={showBenchmarks} onClose={() => setShowBenchmarks(false)} />
      <ExportReportModal
        isOpen={showExport}
        onClose={() => setShowExport(false)}
        currentMode={currentMode}
        analysisData={getActiveAnalysisData()}
      />
    </div>
  );
};

export default App;

