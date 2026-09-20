export type WorkspaceMode = 'vqa' | 'highlight' | 'bitemporal' | 'optical_sar';

export type ComparisonMode = 'split' | 'dual' | 'blink' | 'blend';

export type SemanticModelChoice = 'geochat' | 'geochat_lora';

export interface SystemStatus {
  online: boolean;
  device: string;
  gpu_name?: string | null;
  vram_total_gb?: number | null;
  vram_used_gb?: number | null;
  vram_free_gb?: number | null;
  ram_percent: number;
  cpu_percent: number;
  active_models_count: number;
  loaded_models: string[];
  geochat_model?: string;
  geochat_status?: string;
  lora_enabled?: boolean;
  lora_path?: string;
  lora_available?: boolean;
  gemini_configured?: boolean;
  gemini_model?: string;
}

export interface ModelStatusItem {
  name: string;
  key: string;
  loaded: boolean;
  device: string;
  vram_mb: number;
  task: string;
  description: string;
  path: string;
  ready: boolean;
}

export interface BenchmarkRecord {
  model: string;
  task: string;
  inference_time_ms: number;
  device: string;
  image_size: string;
  vram_usage_mb: number;
  timestamp: string;
  metadata?: Record<string, any>;
}

export interface ExecutionStage {
  stage: string;
  message: string;
  timestamp: string;
  duration_ms?: number | null;
  memory_mb?: number | null;
}

export interface GeoMetadata {
  crs?: string | null;
  bounds?: number[] | null;
  transform?: number[] | null;
  width: number;
  height: number;
  count: number;
  driver?: string | null;
  nodata?: number | null;
  resolution?: number[] | null;
  sensor_info?: string | null;
  acquisition_date?: string | null;
  has_georeference: boolean;
}

export interface DetectedRegion {
  id: number;
  label: string;
  confidence: number;
  confidence_type: 'model' | 'heuristic';
  bounding_box: [number, number, number, number]; // [x1, y1, x2, y2]
  normalized_box: [number, number, number, number];
  area_pixels: number;
  area_sq_meters?: number | null;
  mask_polygon?: [number, number][];
  segmentation_confidence?: number | null;
}

export interface ChangedRegion {
  region_id: number;
  bounding_box: [number, number, number, number];
  area_pixels: number;
  area_sq_meters?: number | null;
  centroid: [number, number];
  confidence: number;
  confidence_type: string;
  change_type?: string;
}

export interface ModalityFeatureStats {
  modality: string;
  mean_intensity: number;
  std_intensity: number;
  dynamic_range_db?: number | null;
  high_backscatter_ratio?: number | null;
  low_backscatter_ratio?: number | null;
}

export interface VQAResponse {
  task: string;
  status: string;
  question: string;
  answer: string;
  geochat_observations?: string;
  caption?: string | null;
  confidence: number;
  confidence_type: string;
  model: string;
  semantic_model?: string;
  device: string;
  processing_time_ms: number;
  specialized_time_ms?: number | null;
  semantic_time_ms?: number | null;
  image_url: string;
  geo_metadata?: GeoMetadata | null;
  execution_trace: ExecutionStage[];
  transparency_warning?: string;
}

export interface HighlightResponse {
  task: string;
  status: string;
  prompt: string;
  num_detections: number;
  detections: DetectedRegion[];
  total_highlighted_area_pixels: number;
  total_area_percentage: number;
  annotated_image_url: string;
  mask_image_url: string;
  original_image_url: string;
  model: string;
  detection_model?: string;
  segmentation_model?: string;
  semantic_model?: string;
  geochat_observations?: string;
  device: string;
  processing_time_ms: number;
  specialized_time_ms?: number | null;
  semantic_time_ms?: number | null;
  geo_metadata?: GeoMetadata | null;
  execution_trace: ExecutionStage[];
  semantic_summary?: string;
  transparency_warning?: string;
}

export interface BiTemporalStructuredAnalysis {
  overview: string;
  visible_features: string[];
  spatial_pattern: string;
  interpretation: string;
}

export interface ChangeDetectionResponse {
  task: string;
  status: string;
  change_percentage: number;
  num_regions: number;
  total_changed_pixels: number;
  total_pixels: number;
  regions: ChangedRegion[];
  t1_image_url: string;
  t2_image_url: string;
  mask_url: string;
  overlay_url: string;
  side_by_side_url: string;
  model: string;
  detection_model?: string;
  semantic_model: string;
  device: string;
  processing_time_ms: number;
  specialized_time_ms?: number | null;
  semantic_time_ms?: number | null;
  is_coregistered: boolean;
  coregistration_notes?: string | null;
  geo_metadata_t1?: GeoMetadata | null;
  geo_metadata_t2?: GeoMetadata | null;
  semantic_analysis: string;
  structured_analysis?: BiTemporalStructuredAnalysis | null;
  geochat_observations?: string;
  execution_trace: ExecutionStage[];
  transparency_warning?: string;
}

export interface OpticalSARResponse {
  task: string;
  status: string;
  optical_image_url: string;
  sar_image_url: string;
  fused_image_url: string;
  sar_filtered_url: string;
  difference_heatmap_url: string;
  optical_stats: ModalityFeatureStats;
  sar_stats: ModalityFeatureStats;
  detected_features: string[];
  cross_modal_analysis: string;
  geochat_observations?: string;
  model: string;
  optical_processing?: string;
  sar_processing?: string;
  semantic_model?: string;
  device: string;
  processing_time_ms: number;
  specialized_time_ms?: number | null;
  semantic_time_ms?: number | null;
  geo_metadata_optical?: GeoMetadata | null;
  geo_metadata_sar?: GeoMetadata | null;
  execution_trace: ExecutionStage[];
  transparency_warning: string;
}

export interface SampleDataset {
  filename: string;
  filepath: string;
  file_url: string;
  size_bytes: number;
  geo_metadata: GeoMetadata;
}
