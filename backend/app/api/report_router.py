from fastapi import APIRouter, HTTPException
from app.schemas.report import ExportReportRequest, ExportReportResponse
from app.services.report_service import report_service
from app.utils.logger import logger

router = APIRouter(prefix="/reports", tags=["Reports"])

@router.post("/export", response_model=ExportReportResponse)
async def export_report(request: ExportReportRequest):
    try:
        return report_service.generate_pdf(request)
    except Exception as e:
        logger.error(f"Report generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
