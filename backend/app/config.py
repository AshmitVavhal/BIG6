import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent
BACKEND_DIR = BASE_DIR / "backend"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

class Settings(BaseSettings):
    PROJECT_NAME: str = "SATQUERY AI"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Hardware & Model device
    DEVICE: str = os.getenv("DEVICE", "cuda")  # cuda or cpu
    MAX_VRAM_GB: float = float(os.getenv("MAX_VRAM_GB", "5.5"))
    FORCE_CPU_FALLBACK: bool = False
    
    # Storage Paths
    BASE_DIR: Path = BASE_DIR
    DATA_PATH: Path = DATA_DIR
    UPLOAD_PATH: Path = DATA_DIR / "uploads"
    OUTPUT_PATH: Path = DATA_DIR / "outputs"
    SAMPLES_PATH: Path = DATA_DIR / "samples"
    CACHE_PATH: Path = DATA_DIR / "cache"
    
    # Model Paths
    MODELS_BASE_PATH: Path = MODELS_DIR
    GEOCHAT_MODEL_PATH: str = os.getenv("GEOCHAT_MODEL_PATH", "MBZUAI/geochat-7B")
    GEOCHAT_LORA_PATH: str = os.getenv("GEOCHAT_LORA_PATH", str(BASE_DIR / "outputs" / "satquery-geochat-lora"))
    GEOCHAT_USE_LORA: bool = os.getenv("GEOCHAT_USE_LORA", "false").lower() in ("true", "1", "yes")
    GEOCHAT_LOAD_IN_4BIT: bool = os.getenv("GEOCHAT_LOAD_IN_4BIT", "true").lower() in ("true", "1", "yes")
    GEOCHAT_LOAD_IN_8BIT: bool = os.getenv("GEOCHAT_LOAD_IN_8BIT", "false").lower() in ("true", "1", "yes")
    
    LAE_DINO_MODEL_PATH: str = os.getenv("LAE_DINO_MODEL_PATH", "jaychempan/LAE-DINO")
    MASK2FORMER_MODEL_PATH: str = os.getenv("MASK2FORMER_MODEL_PATH", "facebook/mask2former-swin-tiny-cityscapes-semantic")
    CHANGEMAMBA_MODEL_PATH: str = os.getenv("CHANGEMAMBA_MODEL_PATH", str(MODELS_DIR / "changemamba"))
    CHANGE_THRESHOLD: float = float(os.getenv("CHANGE_THRESHOLD", "0.5"))

    # Hugging Face Authentication (Backend only)
    HF_TOKEN: str | None = os.getenv("HF_TOKEN", None)

    # Google Gemini API Configuration (Backend only, never expose to frontend)
    GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY", None)
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

    # Max file upload size: 100MB
    MAX_UPLOAD_SIZE_BYTES: int = 100 * 1024 * 1024
    ALLOWED_IMAGE_EXTENSIONS: list[str] = [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".geotiff"]
    
    # CORS Configuration
    CORS_ORIGINS: list[str] = [
        "https://satquery-zeta.vercel.app",
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ]

    model_config = SettingsConfigDict(
        env_file=[str(BACKEND_DIR / ".env"), str(BASE_DIR / ".env"), ".env", "../.env"],
        extra="allow"
    )

settings = Settings()

if settings.HF_TOKEN:
    os.environ["HF_TOKEN"] = settings.HF_TOKEN

# Ensure directories exist
for p in [settings.UPLOAD_PATH, settings.OUTPUT_PATH, settings.SAMPLES_PATH, settings.CACHE_PATH,
          MODELS_DIR / "geochat", MODELS_DIR / "changemamba", MODELS_DIR / "lae_dino", MODELS_DIR / "mask2former"]:
    p.mkdir(parents=True, exist_ok=True)
