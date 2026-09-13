"""
SatQuery AI - LAE-DINO Service Proxy
Re-exports LAEDINOService and lae_dino_service from app.services.lae_dino_service
"""
from app.services.lae_dino_service import LAEDINOService, lae_dino_service

__all__ = ["LAEDINOService", "lae_dino_service"]
