import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap

# Load data
day = "08"
input_file = f"fire_inputs_2025_01_{day}.tif"

with rasterio.open(input_file) as src:
    fuel = src.read(1)  # Vegetation
    wind_x = src.read(2)  # Wind x-component
    wind_y = src.read(3)  # Wind y-component
    damage = src.read(4)  # Fire damage

print(f"Max wind_x (x-component): {wind_x.max().item()}")
print(f"Min wind_x (x-component): {wind_x.min().item()}")
print(f"Avg wind_x (x-component): {wind_x.mean().item()}")
print(f"Max wind_y (y-component): {wind_y.max().item()}")
print(f"Min wind_y (y-component): {wind_y.min().item()}")
print(f"Avg wind_y (y-component): {wind_y.mean().item()}")

# Create figure with 2x2 grid
fig, axes = plt.subplots(1, 2, figsize=(16, 12))
fig.suptitle(f"Fire Input Layers - January {day}, 2025", fontsize=16)

# --- 1. Damage Layer ---
damage_ax = axes[0, 0]
damage_cmap = ListedColormap(['none', 'red'])
damage_ax.imshow(fuel, cmap='YlOrRd', alpha=0.3)  # Background
damage_ax.imshow(damage, cmap=damage_cmap, alpha=0.7)
damage_ax.set_title(f"Fire Damage (Red = Burned)\nTotal Burned: {damage.sum():,} pixels")
damage_ax.axis('off')

# --- 2. Fuel Layer ---
fuel_ax = axes[0, 1]
fuel_plot = fuel_ax.imshow(fuel, cmap='YlOrRd', vmin=0, vmax=np.percentile(fuel, 95))
fuel_ax.set_title("Vegetation Burn Potential")
plt.colorbar(fuel_plot, ax=fuel_ax, label='Burn Intensity')
fuel_ax.axis('off')

# --- 3. Wind X Component ---
windx_ax = axes[0]
windx_plot = windx_ax.imshow(wind_x, cmap='coolwarm')
windx_ax.set_title("Wind X Component (East-West)")
plt.colorbar(windx_plot, ax=windx_ax)
windx_ax.axis('off')

# --- 4. Wind Y Component ---
windy_ax = axes[1]

windy_plot = windy_ax.imshow(wind_y, cmap='coolwarm')
windy_ax.set_title("Wind Y Component (North-South)")
plt.colorbar(windy_plot, ax=windy_ax)
windy_ax.axis('off')

plt.tight_layout()
plt.savefig(f"fire_layers_visualization_{day}.png", dpi=300, bbox_inches='tight')
plt.show()