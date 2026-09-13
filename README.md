# SATQUERY AI • Satellite Imagery Analysis Platform & GeoChat LoRA Fine-Tuning Engine

![SatQuery AI](https://img.shields.io/badge/Platform-ISRO%20SAC--26167-10b981?style=for-the-badge&logo=satellite)
![CUDA GPU](https://img.shields.io/badge/Compute-NVIDIA%20CUDA%20%2F%20CPU%20Fallback-06b6d4?style=for-the-badge&logo=nvidia)
![LoRA PEFT](https://img.shields.io/badge/Fine--Tuning-PEFT%20•%20LoRA%20(Rank%2016)-ec4899?style=for-the-badge)
![Models](https://img.shields.io/badge/Models-GeoChat%20•%20ChangeFormer%20•%20Grounding%20DINO%20•%20SAM2-f59e0b?style=for-the-badge)

**SatQuery AI** is an advanced AI-powered satellite imagery analysis workstation and fine-tuning suite built for remote-sensing researchers, disaster-management operations, GIS analysts, and defense intelligence teams.

SatQuery couples specialized computer vision models (`ChangeFormer`, `Grounding DINO`, `SAM2`, SAR Signal Engine) with **GeoChat** (`MBZUAI/geochat-7B`), the pioneering multimodal Vision-Language Model purpose-built for remote sensing and Earth observation imagery.

---

## 1. What is LoRA & PEFT in SatQuery?

### Low-Rank Adaptation (LoRA)
Full fine-tuning of large Vision-Language Models (VLMs) like GeoChat (7 Billion parameters) updates billions of weights, requiring massive VRAM (>28 GB) and risking **catastrophic forgetting** of general vision-language capabilities.

**LoRA** freezes the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and decomposes the parameter update $\Delta W$ into low-rank intrinsic matrices:
$$\Delta W = \frac{\alpha}{r} (B \cdot A)$$
where $A \in \mathbb{R}^{r \times k}$ is initialized from a Gaussian distribution, $B \in \mathbb{R}^{d \times r}$ is initialized to zero, $r \ll \min(d, k)$ is the low rank (e.g., $r=16$), and $\alpha$ is a constant scaling factor (e.g., $\alpha=32$).

### Parameter-Efficient Fine-Tuning (PEFT)
Using the Hugging Face `peft` library, only the low-rank adapter matrices in the attention projections (`q_proj`, `v_proj`, `k_proj`, `o_proj`) are trainable:
- **Base GeoChat-7B Parameters**: ~7.06 Billion (**100% FROZEN**)
- **Trainable LoRA Parameters**: ~19.9 Million (**~0.28% of total**)
- **Adapter Weight File Size**: ~39 MB (vs ~14 GB for full weights)

### Why SatQuery Uses LoRA with GeoChat
1. **Domain Specialization**: Adapts GeoChat to mission-specific remote-sensing vocabulary (albedo, backscatter, multispectral bands, NDVI, specular absorption, crop parcel morphology, SAR double-bounce, wildfire albedo anomalies).
2. **Zero Catastrophic Forgetting**: Preserves base spatial reasoning and visual instruction following.
3. **Hardware Accessibility**: Enables reproducible fine-tuning on consumer/workstation GPUs (e.g., 6GB VRAM on RTX 4050 / 3060 Laptop GPUs) via 4-bit NF4 quantization, gradient checkpointing, and batch size 1.
4. **Modularity & Rapid Switching**: Lightweight adapter weights can be attached, detached, or swapped dynamically in memory without restarting the base engine.

---

## 2. SatQuery Model Responsibilities & Separation of Concerns

SatQuery enforces a strict architectural separation: **GeoChat (Base or LoRA) is strictly a semantic reasoning and VQA engine** and is never used to fabricate quantitative pixel masks or numerical change metrics.

```
                    SATELLITE SCENE INGESTION
                               │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
  [Single Image VQA]    [Object Highlighting]   [Bi-Temporal Change]
        │                       │                       │
     GeoChat                 Grounding DINO           ChangeFormer
  (Semantic Reasoning)          │                       │
                                ▼                       ▼
                          SAM2 Polygon           Bi-Temporal Change
                          Segmentation            Mask & Statistics
                                │                       │
                                └───────────┬───────────┘
                                            │
                                   [Structured Evidence]
                                            │
                                            ▼
                                         GeoChat
                                (Domain-Specific Summary)
```

| Component | Architecture | Responsibility / Output |
| :--- | :--- | :--- |
| **Semantic Reasoning / VQA** | `GeoChat-7B` (`MBZUAI/geochat-7B`) + `LoRA` | Natural-language VQA, land-cover explanation, remote-sensing terminology, multi-modal synthesis. |
| **Object Localization** | `Grounding DINO (tiny)` | Authoritative bounding box coordinates & open-vocabulary detection. |
| **Region Segmentation** | `SAM2 (Hiera-tiny)` | Authoritative pixel-accurate polygon segmentation masks. |
| **Change Detection** | `ChangeFormer (Siamese)` | Authoritative bi-temporal pixel change masks, change %, changed regions. |
| **Radar Analytics** | `Multimodal SAR Engine` | Deterministic Lee speckle filtering, dB dynamic range, backscatter ratios. |

---

## 3. Dataset Format & Remote-Sensing Categories

Fine-tuning datasets reside in `data/finetuning/` in JSONL format:

### Standard JSONL Schema
```json
{
  "image": "data/samples/optical_multispectral.png",
  "question": "What transportation and maritime infrastructure is visible along this coastline?",
  "answer": "The coastal port scene contains linear deep-water shipping berths, reinforced seawall docks, container storage staging yards, and an adjacent coastal runway oriented parallel to the shoreline.",
  "category": "transportation",
  "split_tag": "scene_port_01",
  "source": "ISRO-SAC-Verified-2024"
}
```

### 14 Standard Question Categories
1. `scene_understanding`: Overall landscape configuration and broad terrain classification.
2. `land_cover`: Forest, water, built-up, bare soil, and agriculture proportions.
3. `objects`: Aircraft, vessels, storage tanks, vehicles, towers, solar arrays.
4. `infrastructure`: Bridges, runways, road networks, dams, power stations, rail corridors.
5. `agriculture`: Crop parcel delineation, center-pivot fields, orchard canopies.
6. `water`: Inland lakes, drainage courses, coastlines, estuaries, reservoirs.
7. `vegetation`: Canopy density, forest boundaries, riparian greenways.
8. `urban_areas`: High-density settlements, residential parcels, street grids.
9. `transportation`: Highway interchanges, rail yards, shipping lanes, taxiways.
10. `disaster_assessment`: Wildfire burn scars, thermal fire complexes, flood inundation.
11. `environmental_analysis`: Soil erosion, aridification, deforestation, wetland health.
12. `spatial_relationships`: Cardinal directions, relative layout, proximity.
13. `comparative_reasoning`: Relative albedo, textural contrasts, multispectral band differences.
14. `remote_sensing_terminology`: Albedo, spatial resolution, multispectral reflectance, backscatter.

---

## 4. Hardware Requirements & Memory Management

The training and inference pipelines automatically adapt to available hardware:

| Hardware Tier | GPU VRAM | Recommended Configuration | Precision |
| :--- | :--- | :--- | :--- |
| **Entry / Laptop GPU** | **4 – 6 GB** (e.g. RTX 4050/3060) | `batch_size: 1`, `gradient_accumulation_steps: 4`, `gradient_checkpointing: true`, 4-bit BitsAndBytes NF4 | `float16` / `nf4` |
| **Mid-Tier Workstation** | **8 – 16 GB** (e.g. RTX 4070/4080) | `batch_size: 2`, `gradient_accumulation_steps: 2`, `gradient_checkpointing: true`, 8-bit or 4-bit | `bfloat16` / `float16` |
| **High-End / Datacenter** | **24+ GB** (e.g. RTX 4090 / A100) | `batch_size: 4`, `gradient_accumulation_steps: 1`, full precision | `bfloat16` / `float16` |
| **CPU Fallback** | System RAM | Multi-threaded CPU execution (development & debugging) | `float32` |

---

## 5. Step-by-Step PowerShell Workflow

### Step 1: Check Training Environment & Hardware Diagnostics
```powershell
python scripts/check_training_environment.py --dry-run
```
Verifies PyTorch, CUDA, GPU memory, BF16 support, PEFT, Accelerate, Datasets, SentencePiece, and GeoChat classes.

### Step 2: Prepare & Validate Dataset
```powershell
# Prepare train/validation/test splits with genuine calculated statistics
python scripts/prepare_dataset.py --output_dir data/finetuning

# Validate dataset integrity
python scripts/validate_dataset.py --dataset data/finetuning/train.jsonl
```

### Step 3: Add New Verified Samples (Optional)
```powershell
python scripts/add_sample.py `
  --image data/samples/sample_geotiff_sac_scene.tif `
  --question "What is visible in this satellite scene?" `
  --answer "The scene depicts rugged forested mountains with active wildfire thermal hotspots." `
  --category "disaster_assessment" `
  --target data/finetuning/train.jsonl
```

### Step 4: Run LoRA Fine-Tuning
```powershell
python training/train_lora.py --config training/config.yaml
```
Output artifacts generated:
- `outputs/satquery-geochat-lora/adapter_model.safetensors` (LoRA weights)
- `outputs/satquery-geochat-lora/adapter_config.json` (PEFT config)
- `outputs/satquery-geochat-lora/training_metadata.json` (Reproducibility metadata)
- `outputs/satquery-geochat-lora/loss_plot.png` (Training loss curve)
- `experiments/run_001/` (Archived experiment snapshot)

### Step 5: Benchmark Evaluation (Base GeoChat vs SatQuery LoRA)
```powershell
python training/evaluate.py --config training/config.yaml
```
Calculates genuine ROUGE-1/2/L, BLEU-1/4, and generates an interactive HTML human-grading report:
- `evaluation/base_geochat_results.json`
- `evaluation/lora_results.json`
- `evaluation/comparison.json`
- `evaluation/human_evaluation_report.html`

### Step 6: CLI Single-Image & Batch Inference
```powershell
# Run with SatQuery LoRA adapter:
python training/inference.py `
  --image data/samples/optical_multispectral.png `
  --question "What transportation features are visible?" `
  --adapter outputs/satquery-geochat-lora

# Run with Base GeoChat for comparison:
python training/inference.py `
  --image data/samples/optical_multispectral.png `
  --question "What transportation features are visible?" `
  --base
```

### Step 7: Optional Adapter Merging (Standalone Model)
```powershell
python training/merge_adapter.py `
  --adapter outputs/satquery-geochat-lora `
  --output_dir models/geochat_merged
```

---

## 6. SatQuery Web Application Integration

### Reasoning Engine in UI
The SatQuery workstation sidebar displays the active **GeoChat Remote Sensing VLM** engine:
- **Base GeoChat**: Zero-shot foundational remote-sensing vision-language reasoning.
- **SatQuery LoRA**: Domain-fine-tuned satellite reasoning with active LoRA weights when available.

### Environment Configuration (`.env`)
```bash
# Hardware
DEVICE=cuda
MAX_VRAM_GB=5.5

# GeoChat Base & LoRA Paths
GEOCHAT_MODEL_PATH=MBZUAI/geochat-7B
GEOCHAT_LOAD_IN_4BIT=true
GEOCHAT_LOAD_IN_8BIT=false
GEOCHAT_LORA_PATH=outputs/satquery-geochat-lora
GEOCHAT_USE_LORA=true

# Semantic Reasoning Provider
SEMANTIC_MODEL=geochat
```

### Running the Application

#### Start Backend (Terminal 1)
```powershell
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Start Frontend (Terminal 2)
```powershell
cd frontend
npm run dev
```
Access the application at [http://localhost:5173](http://localhost:5173).

---

## 7. Troubleshooting & Limitations

| Symptom | Cause | Solution |
| :--- | :--- | :--- |
| **CUDA Out of Memory during Training** | Sequence length or batch size too high for available VRAM. | Set `gradient_checkpointing: true`, `per_device_train_batch_size: 1`, `gradient_accumulation_steps: 4`, and `load_in_4bit: true`. |
| **LoRA Adapter Not Found in App** | `GEOCHAT_LORA_PATH` points to missing directory. | Run `python training/train_lora.py` to generate weights or set `GEOCHAT_USE_LORA=false` to use base model. |
| **Optical+SAR Reasoning Limitations** | GeoChat is an optical/multispectral remote-sensing VLM. | SatQuery uses physical signal algorithms for radar statistics (dB, speckle) and limits GeoChat to semantic cross-modal interpretation. |
| **Geographic Data Leakage** | Same spatial tile present in train and test splits. | `scripts/prepare_dataset.py` groups samples by `split_tag` to ensure scene-level disjoint splitting. |

---

## 8. License & Attribution
- GeoChat architecture developed by MBZUAI ([GeoChat Paper](https://arxiv.org/abs/2311.15826)).
- Built with PyTorch, Transformers, PEFT, Accelerate, FastAPI, React, and TailwindCSS.
- Developed for satellite remote-sensing analysis, disaster management, and geospatial intelligence operations.
