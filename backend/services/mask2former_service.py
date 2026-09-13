"""
SatQuery AI - Mask2Former Service Proxy
Re-exports Mask2FormerService and mask2former_service from app.services.mask2former_service
"""
from app.services.mask2former_service import Mask2FormerService, mask2former_service

__all__ = ["Mask2FormerService", "mask2former_service"]
