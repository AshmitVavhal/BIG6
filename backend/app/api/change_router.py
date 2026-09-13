from fastapi import APIRouter, HTTPException
from app.schemas.change import ChangeDetectionRequest, ChangeDetectionResponse
from app.pipelines.change_pipeline import change_pipeline
from app.utils.logger import logger

router = APIRouter(prefix="/change", tags=["Bi-Temporal Change"])

@router.post("/analyze", response_model=ChangeDetectionResponse)
async def analyze_change(request: ChangeDetectionRequest):
    try:
        return change_pipeline.run(request)
    except Exception as e:
        logger.error(f"Change detection failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
