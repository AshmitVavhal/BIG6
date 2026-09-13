"""
SatQuery AI - Gemini Service Proxy
Re-exports GeminiService and gemini_service from app.services.gemini_service
"""
from app.services.gemini_service import GeminiService, gemini_service, GEMINI_SYSTEM_INSTRUCTION

__all__ = ["GeminiService", "gemini_service", "GEMINI_SYSTEM_INSTRUCTION"]
