import geopandas as gpd
import rasterio
from rasterio.mask import mask
import json

# Load your area of interest geometry
aoi_geojson_path = "fire_extent.geojson"
gdf = gpd.read_file(aoi_geojson_path)

# Path to the downloaded FBFM13 GeoTIFF
fbfm13_tif_path = "LF2023_FBFM13_240_CONUS/Tif/LC23_F13_240.tif"

# Open the FBFM13 raster
with rasterio.open(fbfm13_tif_path) as src:
    # Ensure the CRS matches between the raster and the vector data
    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)

    # Prepare the geometry for masking
    geom = [json.loads(gdf.to_json())["features"][0]["geometry"]]

    # Clip the raster using the geometry
    out_image, out_transform = mask(src, geom, crop=True)
    out_meta = src.meta.copy()
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

# Save the clipped raster
output_path = "clipped_fbfm13.tif"
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"Saved clipped raster to {output_path}")
