import os
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_system_status():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["online"] is True
    assert "device" in data
    assert "ram_percent" in data
    assert "geochat_model" in data
    assert "geochat_status" in data

def test_models_status():
    response = client.get("/api/models/status")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 4
    model_keys = [m["key"] for m in data["models"]]
    assert "geochat" in model_keys
    assert "changemamba" in model_keys
    assert "lae_dino" in model_keys
    assert "mask2former" in model_keys
    assert "qwen" not in model_keys
    assert "changeformer" not in model_keys
    assert "grounding_dino" not in model_keys
    assert "sam2" not in model_keys
    assert "gemini" not in model_keys

def test_samples_list():
    response = client.get("/api/files/samples")
    assert response.status_code == 200
    data = response.json()
    assert "samples" in data
    assert len(data["samples"]) >= 3

def test_vqa_analyze():
    response = client.post("/api/vqa/analyze", json={
        "image_path": "sample_geotiff_sac_scene.tif",
        "question": "What land-cover features and structures are visible in this scene?"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "answer" in data
    assert len(data["answer"]) > 10
    assert "GeoChat" in data["semantic_model"]
    assert "execution_trace" in data
    assert "has_georeference" in data["geo_metadata"]

def test_highlight_analyze():
    response = client.post("/api/highlight/analyze", json={
        "image_path": "bitemporal_t2_2026.png",
        "prompt": "building",
        "box_threshold": 0.2,
        "use_mask2former_refinement": True
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["num_detections"] > 0
    assert "detections" in data
    assert "annotated_image_url" in data
    assert "GeoChat" in data["semantic_model"]
    assert "LAE-DINO" in data["detection_model"]
    assert "Mask2Former" in data["segmentation_model"]

def test_change_analyze():
    response = client.post("/api/change/analyze", json={
        "t1_image_path": "bitemporal_t1_2024.png",
        "t2_image_path": "bitemporal_t2_2026.png",
        "threshold": 0.35,
        "enable_semantic_reasoning": True
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["change_percentage"] > 0.0
    assert data["num_regions"] > 0
    assert "mask_url" in data
    assert "overlay_url" in data
    assert "GeoChat" in data["semantic_model"]
    assert "ChangeMamba" in data["detection_model"]
    assert len(data["semantic_analysis"]) > 10

def test_optical_sar_analyze():
    response = client.post("/api/optical-sar/analyze", json={
        "optical_image_path": "optical_multispectral.png",
        "sar_image_path": "sar_sentinel1.png",
        "despeckle_sar": True
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "sar_stats" in data
    assert data["sar_stats"]["dynamic_range_db"] > 0
    assert "transparency_warning" in data
    assert "GeoChat" in data["semantic_model"]
    assert "fused_image_url" in data

def test_report_export():
    # Test report generation with GeoChat
    response = client.post("/api/reports/export", json={
        "title": "SATQUERY TEST INTELLIGENCE REPORT",
        "task_type": "bi_temporal_change",
        "analysis_data": {
            "model": "ChangeMamba + GeoChat",
            "detection_model": "ChangeMamba",
            "semantic_model": "GeoChat",
            "processing_time_ms": 1420.5,
            "change_percentage": 24.8,
            "num_regions": 42,
            "total_changed_pixels": 260100,
            "semantic_analysis": "Significant urban residential expansion and road network construction.",
            "coregistration_notes": "Spatial co-registration verified."
        }
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "satquery_report_" in data["report_filename"]
    assert data["file_size_bytes"] > 1000

def test_geochat_status_in_system():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert "geochat_model" in data
    assert "geochat_status" in data
    assert "lora_enabled" in data
    assert "lora_path" in data
    assert "lora_available" in data

def test_vqa_with_geochat():
    response = client.post("/api/vqa/analyze", json={
        "image_path": "sample_geotiff_sac_scene.tif",
        "question": "What land-cover features are visible?",
        "semantic_model": "geochat"
    })
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "GeoChat" in data["semantic_model"]
    assert "answer" in data
    assert len(data["answer"]) > 10
