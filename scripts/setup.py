import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

def setup_project():
    print("=" * 60)
    print("SATQUERY AI - PROJECT ENVIRONMENT SETUP")
    print("=" * 60)

    # 1. Create directories
    for d in ["data/uploads", "data/outputs", "data/samples", "data/cache",
              "models/qwen", "models/changeformer", "models/sam2", "models/grounding_dino"]:
        path = BASE_DIR / d
        path.mkdir(parents=True, exist_ok=True)
        print(f"  Created directory: {d}")

    # 2. Check Python packages
    print("\n[Step 1] Checking Python dependencies...")
    subprocess.run([sys.executable, str(BASE_DIR / "scripts/check_gpu.py")])

    # 3. Generate sample datasets
    print("\n[Step 2] Generating sample satellite datasets (GeoTIFFs, Bi-Temporal pairs, Optical+SAR)...")
    subprocess.run([sys.executable, str(BASE_DIR / "scripts/generate_samples.py")])

    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("Start the backend with:  cd backend && python -m uvicorn app.main:app --reload")
    print("Start the frontend with: cd frontend && npm run dev")
    print("=" * 60)

if __name__ == "__main__":
    setup_project()
