# Remote Sensing & Satellite VQA Datasets Reference

This document provides curated guidance on trusted, verified remote-sensing datasets suitable for fine-tuning Vision-Language Models like `Qwen2.5-VL-3B-Instruct` in the satellite domain.

---

## 1. Verified Remote-Sensing VQA Datasets

| Dataset | Modality | Samples | Question Types | License / Access | Source Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **RSVQA (HR & LR)** | High/Low Res Optical | ~1M Q-A pairs | Presence, Counting, Comparison, Area estimation | Open Research (CC-BY-4.0) | [Lobry et al., IEEE TGRS 2020](https://rsvqa.sylvainlobry.com/) |
| **RSIVQA** | High-Res Aerial/Satellite | ~110k Q-A pairs | Land cover, Object properties, Spatial relationships | Academic / Research | [Zheng et al., IEEE GRSL 2021](https://github.com/Z-Z-X/RSIVQA) |
| **EarthVQA** | Multi-sensor Satellite | ~50k Q-A pairs | Disaster response, Urban changes, Environmental | Open Benchmark | [EarthVQA Benchmark](https://github.com/earthvqa) |
| **LHRS-Bot Dataset** | Multi-resolution Optical | ~100k pairs | Multi-turn reasoning, GIS features, Remote sensing captions | Open Access (Apache 2.0) | [Mu et al., arXiv 2024](https://github.com/LHRS-Bot) |
| **RSICD** | High-Res Optical | ~54k Captions | Remote sensing image captioning, Land cover | Research (CC-BY-NC) | [Lu et al., IEEE TGRS 2017](https://github.com/201528014227051/RSICD_optimal) |
| **UCMerced Captions** | 21 Land-use Classes | ~10.5k Captions | Structural features, Agricultural, Coastal | Academic | [Qu et al., IGARSS 2016](http://weegee.vision.ucmerced.edu/datasets/landuse.html) |

---

## 2. Standard Question Categories Supported in SatQuery

Every sample in SatQuery's fine-tuning pipeline is categorized into one of 14 domain classes:

1. `scene_understanding`: Broad scene classification, terrain context, and overall landscape configuration.
2. `land_cover`: Identification and proportion of forest, water, built-up, bare soil, and agriculture.
3. `objects`: Detection and identification of vehicles, aircraft, ships, storage tanks, and towers.
4. `infrastructure`: Bridges, runways, road networks, dams, power stations, and railways.
5. `agriculture`: Crop parcel delineation, pivot irrigation fields, orchard canopies, and fallow land.
6. `water`: Inland lakes, river courses, coastlines, estuaries, and drainage basins.
7. `vegetation`: Canopy density, forest boundaries, riparian greenways, and shrublands.
8. `urban_areas`: Residential clusters, commercial complexes, high-density settlements, and street grids.
9. `transportation`: Highway intersections, rail yards, maritime shipping lanes, and taxiways.
10. `disaster_assessment`: Wildfire burn scars, active thermal fire complexes, smoke plumes, flood inundation.
11. `environmental_analysis`: Soil erosion, aridification, coastal erosion, deforestation, and wetland health.
12. `spatial_relationships`: Cardinal directions, proximity, orientation, and relative layout of features.
13. `comparative_reasoning`: Relative size, density differences, textural contrasts, and multispectral comparisons.
14. `remote_sensing_terminology`: Albedo, multispectral band characteristics, spatial resolution, and radiometric features.

---

## 3. Dataset Formatting Guide

The SatQuery pipeline uses standard JSONL files in `data/finetuning/`:

### Standard Format (`train.jsonl`, `validation.jsonl`, `test.jsonl`)
```json
{
  "image": "images/sac_scene_01.png",
  "question": "What primary land-cover categories are present in this satellite image?",
  "answer": "The scene contains dense urban built-up areas with high road connectivity in the northwest, transitioning into agricultural parcels and riparian vegetation along the river corridor.",
  "category": "land_cover",
  "source": "ISRO-SAC-Verified-2024",
  "verified_by": "Senior Remote Sensing Analyst"
}
```

### Conversational Messages Format (Native Qwen2.5-VL Chat Template)
```json
{
  "image": "images/sac_scene_01.png",
  "category": "land_cover",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image"},
        {"type": "text", "text": "What primary land-cover categories are present in this satellite image?"}
      ]
    },
    {
      "role": "assistant",
      "content": [
        {"type": "text", "text": "The scene contains dense urban built-up areas with high road connectivity in the northwest, transitioning into agricultural parcels and riparian vegetation along the river corridor."}
      ]
    }
  ]
}
```

---

## 4. Ethical, Privacy, and Licensing Guidelines

- **User Imagery Isolation**: User imagery uploaded during interactive sessions (`data/uploads/`) is **strictly segregated** and is NEVER automatically ingested into fine-tuning datasets.
- **Copyright & Attribution**: Datasets must comply with original open-access or research licenses. Always credit original annotators and institutions.
- **Verification Requirement**: Do not ingest synthetic text from ungrounded sources as "ground truth". Training data must be verified against authentic GIS layers or validated by domain analysts.
