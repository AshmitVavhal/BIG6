from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ExportReportRequest(BaseModel):
    title: str = Field("SATQUERY INTELLIGENCE REPORT", description="Report title")
    task_type: str = Field(..., description="vqa, highlight, bi_temporal_change, or optical_sar")
    analysis_data: Dict[str, Any] = Field(..., description="Complete analysis result JSON payload")
    analyst_notes: Optional[str] = None
    mission_id: Optional[str] = "ISRO-SAC-26167"

class ExportReportResponse(BaseModel):
    status: str = "success"
    report_filename: str
    download_url: str
    generated_at: str
    file_size_bytes: int
