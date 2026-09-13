import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from app.schemas.system import BenchmarkRecord

class BenchmarkTracker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(BenchmarkTracker, cls).__new__(cls)
            cls._instance.records: List[BenchmarkRecord] = []
        return cls._instance

    def record_run(
        self,
        model: str,
        task: str,
        inference_time_ms: float,
        device: str,
        image_size: str,
        vram_usage_mb: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> BenchmarkRecord:
        rec = BenchmarkRecord(
            model=model,
            task=task,
            inference_time_ms=round(inference_time_ms, 2),
            device=device,
            image_size=image_size,
            vram_usage_mb=round(vram_usage_mb, 2),
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            metadata=metadata or {}
        )
        self.records.append(rec)
        # Keep last 100 records
        if len(self.records) > 100:
            self.records.pop(0)
        return rec

    def get_records(self) -> List[BenchmarkRecord]:
        return self.records

    def get_summary(self) -> Dict[str, Any]:
        if not self.records:
            return {}
        summary = {}
        for rec in self.records:
            if rec.model not in summary:
                summary[rec.model] = {
                    "task": rec.task,
                    "avg_ms": 0.0,
                    "count": 0,
                    "total_ms": 0.0,
                    "last_device": rec.device,
                    "last_vram_mb": rec.vram_usage_mb
                }
            s = summary[rec.model]
            s["total_ms"] += rec.inference_time_ms
            s["count"] += 1
            s["avg_ms"] = round(s["total_ms"] / s["count"], 2)
            s["last_device"] = rec.device
            s["last_vram_mb"] = rec.vram_usage_mb
        return summary

benchmark_tracker = BenchmarkTracker()
