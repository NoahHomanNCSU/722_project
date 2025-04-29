import cdsapi
import xarray as xr
import numpy as np
import rioxarray
import os
import geopandas as gpd

gdf = gpd.read_file("fire_extent.geojson").to_crs("EPSG:4326")
minx, miny, maxx, maxy = gdf.total_bounds
bbox = [maxy, minx, miny, maxx]
days = ['08', '09', '10', '11']

c = cdsapi.Client()

for day in days:
    fname = f"wind_2025_01_{day}.nc"
    
    if not os.path.exists(fname):
        print(f"Downloading ERA5 wind for Jan {day}...")
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': ['10m_u_component_of_wind', '10m_v_component_of_wind'],
                'year': '2025',
                'month': '01',
                'day': day,
                'time': [f'{h:02d}:00' for h in range(24)],
                'area': bbox,  # N, W, S, E
            },
            fname
        )

    print(f"Processing wind data for Jan {day}...")
    ds = xr.open_dataset(fname)
    u10 = ds["u10"].mean(dim="valid_time")
    v10 = ds["v10"].mean(dim="valid_time")

    magnitude = np.sqrt(u10**2 + v10**2)
    direction = (180 + np.degrees(np.arctan2(u10, v10))) % 360

    wind_stack = xr.concat([magnitude, direction], dim="band")
    wind_stack = wind_stack.assign_coords(band=["magnitude", "direction"])
    wind_stack.rio.write_crs("EPSG:4326", inplace=True)
    wind_stack.rio.to_raster(f"wind_2025_01_{day}.tif")

    print(f"✅ Saved: wind_2025_01_{day}.tif")
