import os
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from PIL import Image
import numpy as np

from app.services.gemini_service import GeminiService
from app.services.semantic_reasoning_service import semantic_reasoning_service

def test_gemini_service_initialization_without_key():
    """Test that GeminiService correctly reports unconfigured when no API key is provided."""
    service = GeminiService(api_key="")
    assert not service.is_configured
    result = service.analyze(
        image=Image.new("RGB", (100, 100)),
        user_question="What is this?"
    )
    assert not result["success"]
    assert "Gemini API key is not configured" in result["error"]
    assert "unconfigured" in result["model"]

def test_gemini_service_build_multimodal_prompt():
    """Test multimodal prompt construction with GeoChat observations, CV results, and verified metrics."""
    service = GeminiService(api_key="mock_test_key", model_name="gemini-1.5-flash")
    
    geochat_obs = {
        "scene": "Coastal port area with docks and ships",
        "visible_features": ["piers", "cargo vessels", "waterfront warehouses"],
        "spatial_patterns": "Linear dock structures along shoreline",
        "possible_interpretation": "Active maritime port",
        "uncertainties": "Vessel types cannot be resolved due to GSD"
    }
    cv_results = {
        "num_detections": 3,
        "labels": ["ship", "dock"],
        "model": "LAE-DINO + Mask2Former"
    }
    verified_metrics = {
        "change_percentage": 14.85,
        "changed_pixels": 45120,
        "valid_pixels": 303840
    }
    
    prompt = service._build_prompt(
        user_question="Identify maritime activity in this sector.",
        geochat_observations=geochat_obs,
        cv_results=cv_results,
        verified_metrics=verified_metrics
    )
    
    assert "Coastal port area" in prompt
    assert "LAE-DINO + Mask2Former" in prompt
    assert "14.85" in prompt
    assert "Identify maritime activity in this sector." in prompt
    assert "verified_metrics" in prompt

def test_gemini_service_analyze_mock_success():
    """Test successful Gemini analysis with mocked Google GenerativeAI API response."""
    service = GeminiService(api_key="mock_test_key", model_name="gemini-1.5-flash")
    
    mock_response = MagicMock()
    mock_response.text = "The satellite image shows a large industrial port facility with multiple cargo vessels docked."
    
    with patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.return_value = mock_response
        mock_model_cls.return_value = mock_model_instance
        
        img = Image.new("RGB", (256, 256), color=(50, 100, 150))
        res = service.analyze(
            image=img,
            user_question="What is shown in this scene?",
            geochat_observations={"scene": "Port facility"},
            cv_results={"detections": 2},
            verified_metrics={"area_sqkm": 2.5}
        )
        
        assert res["success"] is True
        assert "port facility" in res["answer"].lower()
        assert "gemini-1.5-flash" in res["model"]
        assert res["inference_time_ms"] >= 0
        assert "original_image" in res["sources"]
        assert "GeoChat" in res["sources"]
        assert "verified_metrics" in res["sources"]

def test_gemini_service_analyze_api_error():
    """Test graceful handling of Gemini API errors (e.g. Quota/Auth errors)."""
    service = GeminiService(api_key="mock_test_key")
    
    with patch("google.generativeai.GenerativeModel") as mock_model_cls:
        mock_model_instance = MagicMock()
        mock_model_instance.generate_content.side_effect = Exception("ResourceExhausted: 429 Rate limit exceeded")
        mock_model_cls.return_value = mock_model_instance
        
        img = Image.new("RGB", (256, 256))
        res = service.analyze(
            image=img,
            user_question="Describe the scene."
        )
        
        assert res["success"] is False
        assert "ResourceExhausted" in res["error"]
        assert res["answer"] == ""

def test_semantic_reasoning_pipeline_preserves_verified_change_metrics():
    """Verify that ChangeMamba metrics are strictly preserved and passed to reasoning without modification."""
    t1_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
    t2_rgb = np.ones((128, 128, 3), dtype=np.uint8) * 128
    
    # Mock Gemini response to ensure deterministic test
    mock_gemini_res = {
        "success": True,
        "answer": "Bi-temporal analysis confirms 36.68% of the area experienced significant surface changes.",
        "model": "ChangeMamba + GeoChat + Gemini (gemini-1.5-flash)",
        "semantic_source": "GeoChat + original image",
        "verified_metrics": {"change_percentage": "36.68%"},
        "inference_time_ms": 120.0,
        "sources": ["original_image", "GeoChat", "ChangeMamba (Quantitative)", "Gemini (gemini-1.5-flash)"]
    }
    
    with patch("app.services.semantic_reasoning_service.gemini_service.analyze", return_value=mock_gemini_res):
        with patch.object(GeminiService, "is_configured", new_callable=PropertyMock, return_value=True):
            result = semantic_reasoning_service.explain_bi_temporal(
                change_percentage=36.68,
                num_regions=7,
                regions=[],
                t1_rgb=t1_rgb,
                t2_rgb=t2_rgb,
                coregistration_notes="Sub-pixel phase correlation aligned"
            )
            
            assert "explanation" in result
            assert "36.68%" in result["explanation"]
            assert "geochat_observations" in result
            assert "sources" in result
            assert "ChangeMamba" in str(result["sources"])
