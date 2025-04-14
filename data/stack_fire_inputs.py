import rasterio
from rasterio import features
from rasterio.enums import Resampling
import geopandas as gpd
import numpy as np

# --- Step 1: Load SCLC+FBFM13 merged burn potential raster ---
with rasterio.open("clipped_sclc_fbfm13_burn.tif") as src:
    fuel_layer = src.read(1).astype(np.float32)
    profile = src.profile
    transform = src.transform
    shape = fuel_layer.shape
    crs = src.crs

# Ensure no NaNs in vegetation
fuel_layer = np.nan_to_num(fuel_layer, nan=0.0)

# --- Step 2: Rasterize roads using exact shape and transform ---
gdf_roads = gpd.read_file("clipped_osm_roads.geojson").to_crs(crs)
road_mask = features.rasterize(
    [(geom, 1) for geom in gdf_roads.geometry],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

# --- Step 3: Stack wind and damage layers per day ---
damage_files = {
    "08": "datasets/pallisades_fire/20250108_222134_ssc10_u0001_damage_predictions.gpkg",
    "09": "datasets/pallisades_fire/20250109_221527_ssc7_u0001_damage_predictions.gpkg",
    "10": "datasets/pallisades_fire/maxar_palisades_1050010040277500_damage_predictions.gpkg",
}

days = ["08", "09", "10"]
for day in days:
    # --- Load wind data and resample to match vegetation raster ---
    wind_file = f"wind_2025_01_{day}.tif"
    with rasterio.open(wind_file) as wind_src:
        wind_mag = wind_src.read(
            1, out_shape=shape, resampling=Resampling.bilinear
        ).astype(np.float32)
        wind_dir = wind_src.read(
            2, out_shape=shape, resampling=Resampling.bilinear
        ).astype(np.float32)

    # --- Rasterize damage layer (1 = damaged, 0 = not) ---
    gdf_damage = gpd.read_file(damage_files[day]).to_crs(crs)
    damaged_layer = features.rasterize(
        [(geom, int(val)) for geom, val in zip(gdf_damage.geometry, gdf_damage["damaged"])],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    ).astype(np.float32)

    # --- Step 4: Stack all 5 layers ---
    stacked = np.stack([
        fuel_layer,       # 1: Vegetation burn potential
        road_mask,        # 2: Binary road mask
        wind_mag,         # 3: Wind magnitude
        wind_dir,         # 4: Wind direction
        damaged_layer     # 5: Binary fire damage
    ])

    # --- Step 5: Write to GeoTIFF ---
    out_path = f"fire_inputs_2025_01_{day}.tif"
    out_profile = profile.copy()
    out_profile.update(count=5, dtype="float32")

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(stacked)

    print(f"✅ Saved: {out_path}")

