import rasterio
from rasterio import features
from rasterio.enums import Resampling
import geopandas as gpd
import numpy as np
import os

with rasterio.open("clipped_fbfm13.tif") as src:
    fuel_codes = src.read(1)
    profile = src.profile
    transform = src.transform
    shape = fuel_codes.shape
    crs = src.crs

# Source: Scott & Burgan (2005), RMRS-GTR-153
burn_potential_map = {
    1:  0.57,  # Short grass
    2:  0.57,  # Timber grass
    3:  1.00,  # Tall grass
    4:  0.86,  # Chaparral (short)
    5:  0.57,  # Brush
    6:  0.86,  # Dormant shrub
    7:  0.14,  # Pine litter
    8:  0.07,  # Hardwood litter
    9:  0.29,  # Long-needle litter
    10: 0.57,  # Timber + brush
    11: 0.29,  # Light slash
    12: 0.14,  # Medium/heavy slash
    13: 0.00,  # Non-burnable
}
fuel_layer = np.vectorize(burn_potential_map.get)(fuel_codes).astype(np.float32)
fuel_layer = np.nan_to_num(fuel_layer, nan=0.0)

gdf_roads = gpd.read_file("clipped_osm_roads.geojson").to_crs(crs)
road_mask = features.rasterize(
    [(geom, 1) for geom in gdf_roads.geometry],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype="uint8"
)

damage_files = {
    "08": "datasets/pallisades_fire/20250108_222134_ssc10_u0001_damage_predictions.gpkg",
    "09": "datasets/pallisades_fire/20250109_221527_ssc7_u0001_damage_predictions.gpkg",
    "10": "datasets/pallisades_fire/maxar_palisades_1050010040277500_damage_predictions.gpkg",
}

days = ["08", "09", "10"]
for day in days:
    wind_file = f"wind_2025_01_{day}.tif"
    with rasterio.open(wind_file) as src:
        wind_mag = src.read(1, out_shape=shape, resampling=Resampling.bilinear)
        wind_dir = src.read(2, out_shape=shape, resampling=Resampling.bilinear)

    wind_mag = wind_mag.astype(np.float32)
    wind_dir = wind_dir.astype(np.float32)

    gdf_damage = gpd.read_file(damage_files[day]).to_crs(crs)
    damaged_layer = features.rasterize(
        [(geom, int(val)) for geom, val in zip(gdf_damage.geometry, gdf_damage["damaged"])],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    ).astype(np.float32)

    # --- Step 6: Stack all 5 layers ---
    stacked = np.stack([
        fuel_layer,       # 1
        road_mask,        # 2
        wind_mag,         # 3
        wind_dir,         # 4
        damaged_layer     # 5
    ])

    out_path = f"fire_inputs_2025_01_{day}.tif"

    out_profile = profile.copy()
    out_profile.update(count=5, dtype="float32")

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(stacked)

    print(f" Saved: {out_path}")
