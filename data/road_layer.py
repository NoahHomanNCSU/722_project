import geopandas as gpd
import osmnx as ox
from shapely.ops import unary_union

gdf_extent = gpd.read_file("fire_extent.geojson")
if gdf_extent.crs is None or gdf_extent.crs.to_string() != "EPSG:4326":
    print("Reprojecting fire extent to EPSG:4326...")
    gdf_extent = gdf_extent.to_crs(epsg=4326)

combined_geom = unary_union(gdf_extent.geometry).simplify(0.0001, preserve_topology=True)
minx, miny, maxx, maxy = combined_geom.bounds

print("Querying OSM...")
tags = {"highway": True}
gdf_roads_raw = ox.geometries_from_bbox(north=maxy, south=miny, east=maxx, west=minx, tags=tags)

print(f"Downloaded {len(gdf_roads_raw)} road features")

gdf_roads_raw = gdf_roads_raw[gdf_roads_raw.geometry.notnull()]
gdf_roads_raw = gdf_roads_raw.set_geometry("geometry")
matching_idx = gdf_roads_raw.sindex.query(combined_geom, predicate="intersects")
gdf_roads_clipped = gdf_roads_raw.iloc[matching_idx]

gdf_roads_clipped.to_file("clipped_osm_roads.geojson", driver="GeoJSON")
print("✅ Saved clipped road layer.")



