"""
SatQuery AI - ChangeMamba Service Proxy
Re-exports ChangeMambaService and changemamba_service from app.services.changemamba_service
"""
from app.services.changemamba_service import ChangeMambaService, changemamba_service

__all__ = ["ChangeMambaService", "changemamba_service"]
