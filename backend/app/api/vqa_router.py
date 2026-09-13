from fastapi import APIRouter, HTTPException
from app.schemas.vqa import VQARequest, VQAResponse
from app.pipelines.vqa_pipeline import vqa_pipeline
from app.utils.logger import logger

router = APIRouter(prefix="/vqa", tags=["VQA"])

@router.post("/analyze", response_model=VQAResponse)
async def analyze_vqa(request: VQARequest):
    try:
        return vqa_pipeline.run(request)
    except Exception as e:
        logger.error(f"VQA Analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
