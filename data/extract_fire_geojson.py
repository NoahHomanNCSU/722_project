import geopandas as gpd
from shapely.ops import unary_union

gdf = gpd.read_file("data/datasets/pallisades_fire/maxar_palisades_1050010040277500_damage_predictions.gpkg")
combined_geom = unary_union(gdf["geometry"])

gdf_extent = gpd.GeoDataFrame(geometry=[combined_geom], crs=gdf.crs)
gdf_extent.to_file("fire_extent.geojson", driver="GeoJSON")
