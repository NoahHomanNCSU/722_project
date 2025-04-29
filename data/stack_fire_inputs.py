import rasterio
from rasterio import features
from rasterio.enums import Resampling
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("clipped_sclc_fbfm13_burn.tif") as src:
    fuel_layer = src.read(1).astype(np.float32)
    profile = src.profile
    transform = src.transform
    shape = fuel_layer.shape
    crs = src.crs

fuel_layer = np.nan_to_num(fuel_layer, nan=0.0)


damage_files = {
    "08": "datasets/pallisades_fire/20250108_222134_ssc10_u0001_damage_predictions.gpkg",
    "09": "datasets/pallisades_fire/20250109_221527_ssc7_u0001_damage_predictions.gpkg",
    "10": "datasets/pallisades_fire/maxar_palisades_1050010040277500_damage_predictions.gpkg",
}


days = ["08", "09", "10"]
for i, day in enumerate(days):
    wind_file = f"wind_2025_01_{day}.tif"
    with rasterio.open(wind_file) as wind_src:
        wind_x = wind_src.read(
            1, out_shape=shape, resampling=Resampling.bilinear
        ).astype(np.float32)
        wind_y = wind_src.read(
            2, out_shape=shape, resampling=Resampling.bilinear
        ).astype(np.float32)

    gdf_damage = gpd.read_file(damage_files[day]).to_crs(crs)
    damaged_layer = features.rasterize(
        [(geom, int(val)) for geom, val in zip(gdf_damage.geometry, gdf_damage["damaged"])],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    ).astype(np.float32)

    stacked = np.stack([
        fuel_layer,       
        wind_x,         
        wind_y,      
        damaged_layer   
    ])

    # --- Step 5: Write to GeoTIFF ---
    out_path = f"fire_inputs_2025_01_{day}.tif"
    out_profile = profile.copy()
    out_profile.update(count=4, dtype="float32")

    with rasterio.open(out_path, "w", **out_profile) as dst:
        dst.write(stacked)

    print(f"Saved: {out_path}")
