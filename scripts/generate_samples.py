import os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

def create_synthetic_satellite_scenes():
    print(f"Generating realistic satellite sample datasets in {SAMPLES_DIR}...")
    np.random.seed(42)

    # =========================================================================
    # 1. Base Terrain Canvas (1024x1024)
    # =========================================================================
    w, h = 1024, 1024
    
    # -------------------------------------------------------------------------
    # Scene 1: Multi-date Bi-Temporal Disaster / Urban Expansion (T1: 2024, T2: 2026)
    # -------------------------------------------------------------------------
    # T1: Natural terrain + agricultural fields + river + a few rural structures
    t1_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Base soil/grass background (olive green / brown)
    t1_canvas[:, :, 0] = np.random.randint(90, 115, (h, w)) # R
    t1_canvas[:, :, 1] = np.random.randint(125, 155, (h, w)) # G
    t1_canvas[:, :, 2] = np.random.randint(65, 90, (h, w)) # B

    # Agricultural parcels (grid of fields)
    for row in range(0, h, 180):
        for col in range(0, w, 220):
            field_g = np.random.randint(110, 170)
            field_r = np.random.randint(70, 120)
            field_b = np.random.randint(50, 90)
            t1_canvas[row+5:min(h-5, row+175), col+5:min(w-5, col+215)] = [field_r, field_g, field_b]

    # Meandering river corridor
    pts = np.array([[0, 200], [250, 320], [500, 280], [750, 450], [1024, 520]], np.int32)
    for i in range(len(pts) - 1):
        cv2.line(t1_canvas, tuple(pts[i]), tuple(pts[i+1]), (35, 75, 120), 45) # Deep river water

    # Winding secondary road in T1
    cv2.line(t1_canvas, (100, 0), (120, 1024), (160, 160, 160), 10)
    cv2.line(t1_canvas, (120, 500), (1024, 600), (150, 150, 150), 8)

    # A few sparse rural buildings in T1
    for bx, by in [(160, 120), (190, 140), (220, 110), (700, 750), (740, 780)]:
        cv2.rectangle(t1_canvas, (bx, by), (bx+28, by+22), (180, 70, 60), -1) # terracotta roof
        cv2.rectangle(t1_canvas, (bx, by), (bx+28, by+22), (50, 50, 50), 1)

    # -------------------------------------------------------------------------
    # T2: Major Urban Expansion & Construction (2026)
    # -------------------------------------------------------------------------
    t2_canvas = t1_canvas.copy()

    # Major arterial multi-lane highway constructed
    cv2.line(t2_canvas, (0, 700), (1024, 850), (60, 60, 60), 22)
    cv2.line(t2_canvas, (0, 700), (1024, 850), (230, 230, 230), 2) # lane divider
    cv2.line(t2_canvas, (550, 0), (580, 1024), (70, 70, 70), 18)

    # Large residential & industrial complex (changed area!)
    for r in range(40, 350, 45):
        for c in range(350, 900, 55):
            roof_type = np.random.choice(["blue_metal", "white_concrete", "red_tile", "solar_panel"])
            if roof_type == "blue_metal":
                color = (40, 110, 190)
            elif roof_type == "white_concrete":
                color = (220, 225, 230)
            elif roof_type == "solar_panel":
                color = (20, 35, 75)
            else:
                color = (190, 80, 65)
            cv2.rectangle(t2_canvas, (c, r), (c+38, r+30), color, -1)
            cv2.rectangle(t2_canvas, (c, r), (c+38, r+30), (30, 30, 30), 1)

    # Commercial parking lots and paved terrain
    t2_canvas[600:780, 250:500] = [130, 135, 140]
    for px in range(260, 490, 20):
        for py in range(610, 770, 15):
            cv2.rectangle(t2_canvas, (px, py), (px+8, py+12), (240, 240, 245), -1)

    # Save T1 & T2 PNGs
    Image.fromarray(t1_canvas).save(SAMPLES_DIR / "bitemporal_t1_2024.png")
    Image.fromarray(t2_canvas).save(SAMPLES_DIR / "bitemporal_t2_2026.png")
    print("  [OK] Created: bitemporal_t1_2024.png & bitemporal_t2_2026.png")

    # -------------------------------------------------------------------------
    # Scene 2: Optical + SAR Modality Pair (Port Facility / Coastal Runway)
    # -------------------------------------------------------------------------
    opt_canvas = np.zeros((h, w, 3), dtype=np.uint8)
    sar_canvas = np.zeros((h, w), dtype=np.uint8)

    # Optical: deep blue ocean on left, land & port on right
    opt_canvas[:, :450] = [25, 60, 115] # Ocean water
    opt_canvas[:, 450:] = [140, 145, 130] # Concrete docks & industrial soil

    # Coastal breakwater & piers
    cv2.rectangle(opt_canvas, (400, 150), (450, 450), (160, 160, 165), -1)
    cv2.rectangle(opt_canvas, (400, 600), (450, 900), (160, 160, 165), -1)

    # Optical Ships in harbor
    for sx, sy in [(220, 250), (280, 720), (320, 400)]:
        cv2.ellipse(opt_canvas, (sx, sy), (40, 14), 15, 0, 360, (230, 230, 240), -1)
        cv2.rectangle(opt_canvas, (sx-10, sy-5), (sx+10, sy+5), (180, 50, 40), -1)

    # Optical Oil storage tanks
    for tx in [600, 720, 840]:
        for ty in [200, 320, 440]:
            cv2.circle(opt_canvas, (tx, ty), 35, (215, 220, 225), -1)
            cv2.circle(opt_canvas, (tx, ty), 35, (80, 80, 80), 2)

    # Runway strip
    cv2.rectangle(opt_canvas, (550, 650), (980, 720), (50, 50, 55), -1)
    cv2.line(opt_canvas, (560, 685), (970, 685), (240, 240, 240), 2)

    # SAR counterpart:
    # Radar physics:
    # 1. Ocean water = specular scatterer (smooth surface bounces radar away) -> Very Dark / Black (pixel ~10-25) + speckle
    # 2. Land/Soil = diffuse scatterer -> Medium Gray (pixel ~80-110)
    # 3. Metallic ships, storage tanks, cranes, vertical walls = double-bounce corner reflectors -> Super Bright / High backscatter (pixel ~230-255)
    
    # Base radar noise
    sar_noise = np.random.gamma(shape=2.0, scale=12.0, size=(h, w)).astype(np.float32)
    sar_land_noise = np.random.gamma(shape=3.0, scale=30.0, size=(h, w)).astype(np.float32)

    sar_canvas[:, :450] = np.clip(sar_noise[:, :450] + 15, 0, 255).astype(np.uint8)
    sar_canvas[:, 450:] = np.clip(sar_land_noise[:, 450:] + 75, 0, 255).astype(np.uint8)

    # Breakwater & docks in SAR
    sar_canvas[150:450, 400:450] = np.clip(np.random.normal(160, 20, (300, 50)), 0, 255).astype(np.uint8)
    sar_canvas[600:900, 400:450] = np.clip(np.random.normal(160, 20, (300, 50)), 0, 255).astype(np.uint8)

    # Ships in SAR (bright metallic double-bounce corner reflection!)
    for sx, sy in [(220, 250), (280, 720), (320, 400)]:
        cv2.ellipse(sar_canvas, (sx, sy), (40, 14), 15, 0, 360, 255, -1)
        # Radar sidelobes / bloom
        cv2.line(sar_canvas, (sx-50, sy), (sx+50, sy), 240, 2)
        cv2.line(sar_canvas, (sx, sy-20), (sx, sy+20), 240, 2)

    # Tanks in SAR (bright cylindrical perimeter reflections)
    for tx in [600, 720, 840]:
        for ty in [200, 320, 440]:
            cv2.circle(sar_canvas, (tx, ty), 35, 250, 4)
            cv2.circle(sar_canvas, (tx, ty), 33, 100, -1)

    # Runway in SAR (smooth asphalt = specular reflection away from radar -> dark strip in SAR)
    sar_canvas[650:720, 550:980] = np.clip(sar_noise[650:720, 550:980] + 20, 0, 255).astype(np.uint8)

    # Save Optical and SAR PNGs
    Image.fromarray(opt_canvas).save(SAMPLES_DIR / "optical_multispectral.png")
    Image.fromarray(sar_canvas).save(SAMPLES_DIR / "sar_sentinel1.png")
    print("  [OK] Created: optical_multispectral.png & sar_sentinel1.png")

    # -------------------------------------------------------------------------
    # Scene 3: GeoTIFF with real CRS (EPSG:32643 - UTM Zone 43N / ISRO SAC)
    # -------------------------------------------------------------------------
    if RASTERIO_AVAILABLE:
        geotiff_path = SAMPLES_DIR / "sample_geotiff_sac_scene.tif"
        # Coordinates around Ahmedabad / SAC ISRO (23.0°N, 72.5°E) in UTM 43N (approx Easting 250000, Northing 2550000)
        transform = from_origin(250000.0, 2550000.0, 0.5, 0.5) # 0.5m GSD resolution
        crs = CRS.from_epsg(32643)

        # Write 3-band GeoTIFF
        with rasterio.open(
            geotiff_path,
            'w',
            driver='GTiff',
            height=h,
            width=w,
            count=3,
            dtype=rasterio.uint8,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(t2_canvas[:, :, 0], 1)
            dst.write(t2_canvas[:, :, 1], 2)
            dst.write(t2_canvas[:, :, 2], 3)
            dst.update_tags(
                SENSOR_ID="Cartosat-3 / PAN-MX",
                PLATFORM="ISRO SAC",
                MISSION="SAC-26167",
                ACQUISITION_DATE="2026-04-18T05:30:00Z",
                RESOLUTION_METERS="0.5"
            )
        print(f"  [OK] Created: sample_geotiff_sac_scene.tif (EPSG:32643, 0.5m GSD, Cartosat-3 tags)")

    print("All sample datasets successfully generated!\n")

if __name__ == "__main__":
    create_synthetic_satellite_scenes()
