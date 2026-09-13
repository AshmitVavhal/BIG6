from fastapi import APIRouter, HTTPException
from app.schemas.optical_sar import OpticalSARRequest, OpticalSARResponse
from app.pipelines.optical_sar_pipeline import optical_sar_pipeline
from app.utils.logger import logger

router = APIRouter(prefix="/optical-sar", tags=["Optical + SAR"])

@router.post("/analyze", response_model=OpticalSARResponse)
async def analyze_optical_sar(request: OpticalSARRequest):
    try:
        return optical_sar_pipeline.run(request)
    except Exception as e:
        logger.error(f"Optical+SAR analysis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
