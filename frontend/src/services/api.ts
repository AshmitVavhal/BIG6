import type {
  SystemStatus,
  ModelStatusItem,
  BenchmarkRecord,
  SampleDataset,
  VQAResponse,
  HighlightResponse,
  ChangeDetectionResponse,
  OpticalSARResponse,
  SemanticModelChoice
} from '../types';

const API_BASE = '/api';

export async function fetchSystemStatus(): Promise<SystemStatus> {
  const res = await fetch(`${API_BASE}/system/status`);
  if (!res.ok) throw new Error(`Failed to fetch system status: ${res.statusText}`);
  return res.json();
}

export async function fetchModelsStatus(): Promise<{ models: ModelStatusItem[]; system_device: string; cuda_available: boolean }> {
  const res = await fetch(`${API_BASE}/models/status`);
  if (!res.ok) throw new Error(`Failed to fetch models status: ${res.statusText}`);
  return res.json();
}

export async function fetchBenchmarks(): Promise<{ summary: Record<string, any>; recent_runs: BenchmarkRecord[] }> {
  const res = await fetch(`${API_BASE}/benchmarks`);
  if (!res.ok) throw new Error(`Failed to fetch benchmarks: ${res.statusText}`);
  return res.json();
}

export async function clearSystemCache(): Promise<{ status: string; message: string }> {
  const res = await fetch(`${API_BASE}/system/clear-cache`, { method: 'POST' });
  if (!res.ok) throw new Error(`Failed to clear cache: ${res.statusText}`);
  return res.json();
}

export async function fetchSamples(): Promise<SampleDataset[]> {
  const res = await fetch(`${API_BASE}/files/samples`);
  if (!res.ok) throw new Error(`Failed to fetch samples: ${res.statusText}`);
  const data = await res.json();
  return data.samples || [];
}

export async function uploadImage(file: File): Promise<{
  filename: string;
  filepath: string;
  file_url: string;
  geo_metadata: any;
}> {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_BASE}/files/upload`, {
    method: 'POST',
    body: formData
  });
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Upload failed: ${res.statusText}`);
  }
  return res.json();
}

export async function analyzeVQA(
  imagePath: string,
  question: string,
  semanticModel: SemanticModelChoice = 'geochat'
): Promise<VQAResponse> {
  const res = await fetch(`${API_BASE}/vqa/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_path: imagePath,
      question,
      semantic_model: semanticModel
    })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `VQA analysis failed: ${res.statusText}`);
  }
  return res.json();
}

export async function analyzeHighlight(
  imagePath: string,
  prompt: string,
  boxThreshold = 0.25,
  useMask2Former = true,
  semanticModel: SemanticModelChoice = 'geochat'
): Promise<HighlightResponse> {
  const res = await fetch(`${API_BASE}/highlight/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      image_path: imagePath,
      prompt,
      box_threshold: boxThreshold,
      use_mask2former_refinement: useMask2Former,
      use_sam2_refinement: useMask2Former,
      semantic_model: semanticModel
    })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Highlight analysis failed: ${res.statusText}`);
  }
  return res.json();
}

export async function analyzeChange(
  t1Path: string,
  t2Path: string,
  threshold = 0.40,
  enableSemantic = true,
  semanticModel: SemanticModelChoice = 'geochat'
): Promise<ChangeDetectionResponse> {
  const res = await fetch(`${API_BASE}/change/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      t1_image_path: t1Path,
      t2_image_path: t2Path,
      threshold,
      enable_semantic_reasoning: enableSemantic,
      semantic_model: semanticModel
    })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Change detection failed: ${res.statusText}`);
  }
  return res.json();
}

export async function analyzeOpticalSAR(
  optPath: string,
  sarPath: string,
  question?: string,
  despeckle = true,
  semanticModel: SemanticModelChoice = 'geochat'
): Promise<OpticalSARResponse> {
  const res = await fetch(`${API_BASE}/optical-sar/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      optical_image_path: optPath,
      sar_image_path: sarPath,
      question,
      despeckle_sar: despeckle,
      semantic_model: semanticModel
    })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Optical-SAR fusion failed: ${res.statusText}`);
  }
  return res.json();
}

export async function exportReport(
  taskType: string,
  analysisData: any,
  analystNotes?: string
): Promise<{ report_filename: string; download_url: string; generated_at: string; file_size_bytes: number }> {
  const res = await fetch(`${API_BASE}/reports/export`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      title: 'SATQUERY SATELLITE INTELLIGENCE REPORT',
      task_type: taskType,
      analysis_data: analysisData,
      analyst_notes: analystNotes,
      mission_id: 'ISRO-SAC-26167'
    })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Report export failed: ${res.statusText}`);
  }
  return res.json();
}
