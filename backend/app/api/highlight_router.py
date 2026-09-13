from fastapi import APIRouter, HTTPException
from app.schemas.highlight import HighlightRequest, HighlightResponse
from app.pipelines.highlight_pipeline import highlight_pipeline
from app.utils.logger import logger

router = APIRouter(prefix="/highlight", tags=["Highlight"])

@router.post("/analyze", response_model=HighlightResponse)
async def analyze_highlight(request: HighlightRequest):
    try:
        return highlight_pipeline.run(request)
    except Exception as e:
        logger.error(f"Highlight analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
