import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import json
import numpy as np

# --- Step 1: Load AOI ---
fire_extent_path = "fire_extent.geojson"
gdf = gpd.read_file(fire_extent_path)

# --- Step 2: Clip SCLC11 raster to AOI ---
sclc_path = "SCLC11.tif"
with rasterio.open(sclc_path) as sclc_src:
    if gdf.crs != sclc_src.crs:
        gdf = gdf.to_crs(sclc_src.crs)

    geom = [json.loads(gdf.to_json())["features"][0]["geometry"]]
    sclc_clipped, sclc_transform = mask(sclc_src, geom, crop=True, filled=True)
    sclc_crs = sclc_src.crs
    sclc_meta = sclc_src.meta.copy()
    sclc_meta.update({
        "height": sclc_clipped.shape[1],
        "width": sclc_clipped.shape[2],
        "transform": sclc_transform,
        "count": 1,
        "dtype": "float32"
    })

sclc_classes = sclc_clipped[0]

# --- Step 3: Map SCLC land cover classes to burn potential ---
class_to_burn = {
    0: 0.0,  # Impervious
    1: 1.0,  # Tree
    2: 0.6,  # Grass
    3: 0.8,  # Shrub
    4: 0.5,  # NP vegetation
    5: 0.0   # Water
}
burn_layer = np.vectorize(class_to_burn.get)(sclc_classes).astype(np.float32)

# --- Step 4: Clip and resample FBFM13 to match SCLC grid ---
fbfm_path = "LC23_F13_240.tif"
with rasterio.open(fbfm_path) as fbfm_src:
    fbfm_resampled = np.empty(sclc_classes.shape, dtype=np.float32)
    reproject(
        source=rasterio.band(fbfm_src, 1),
        destination=fbfm_resampled,
        src_transform=fbfm_src.transform,
        src_crs=fbfm_src.crs,
        dst_transform=sclc_transform,
        dst_crs=sclc_crs,
        resampling=Resampling.nearest
    )

# --- Step 5: Map FBFM13 classes to burn potential ---
fbfm_to_burn = {
    1: 0.57, 2: 0.57, 3: 1.00, 4: 0.86, 5: 0.57,
    6: 0.86, 7: 0.14, 8: 0.07, 9: 0.29, 10: 0.57,
    11: 0.29, 12: 0.14, 13: 0.00
}
fbfm_burn_layer = np.vectorize(fbfm_to_burn.get)(fbfm_resampled.astype(np.uint8)).astype(np.float32)

# --- Step 6: Replace 255s in SCLC-derived layer with FBFM13 values ---
mask_255 = sclc_classes == 255
burn_layer[mask_255] = fbfm_burn_layer[mask_255]
burn_layer = np.nan_to_num(burn_layer, nan=0.0)

# --- Step 7: Save final burn potential layer ---
output_path = "clipped_sclc_fbfm13_burn.tif"
with rasterio.open(output_path, "w", **sclc_meta) as dst:
    dst.write(burn_layer, 1)

print(f"✅ Saved: {output_path}")




