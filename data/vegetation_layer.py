import requests, zipfile, io, os
import geopandas as gpd
import rasterio
from rasterio.mask import mask
import json

fire_geojson_path = "fire_extent.geojson"
gdf = gpd.read_file(fire_geojson_path)

nlcd_img_path = "nlcd_2019/nlcd_2019_land_cover_l48_20210604.img"
with rasterio.open(nlcd_img_path) as src:
    print("Raster CRS:", src.crs)
    print("Fire extent CRS:", gdf.crs)

    if gdf.crs != src.crs:
        gdf = gdf.to_crs(src.crs)
        print("Reprojected fire extent to match raster CRS.")

    geom = [json.loads(gdf.to_json())["features"][0]["geometry"]]

    out_image, out_transform = mask(src, geom, crop=True)
    out_meta = src.meta.copy()
    out_meta.update({
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

output_path = "clipped_nlcd.tif"
with rasterio.open(output_path, "w", **out_meta) as dest:
    dest.write(out_image)

print(f"Saved clipped raster to {output_path}")
