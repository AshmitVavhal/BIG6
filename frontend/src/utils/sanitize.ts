/**
 * Utility to sanitize model names / technical model names from UI display strings.
 */
export function sanitizeModelText(text?: string | null): string {
  if (!text) return '';
  return text
    .replace(/GeoChat-7B/gi, 'VLM')
    .replace(/GeoChat/gi, 'VLM')
    .replace(/Google Gemini/gi, 'Multimodal Engine')
    .replace(/Gemini-1\.5-Flash/gi, 'Multimodal Engine')
    .replace(/Gemini 1\.5 Flash/gi, 'Multimodal Engine')
    .replace(/Gemini/gi, 'Multimodal Engine')
    .replace(/LAE-DINO/gi, 'Zero-Shot Detector')
    .replace(/Grounding DINO/gi, 'Object Detector')
    .replace(/Mask2Former/gi, 'Polygon Segmenter')
    .replace(/ChangeMamba/gi, 'Change Detection Engine')
    .replace(/ChangeFormer/gi, 'Change Detection Engine')
    .replace(/SAM2|SAM-2/gi, 'Segmentation Engine')
    .replace(/MBZUAI\/geochat-7B/gi, 'Remote-Sensing VLM');
}
