"""
Root backend service entry point for GeoChatService.
"""
from app.services.geochat_service import GeoChatService, geochat_service

__all__ = ["GeoChatService", "geochat_service"]
