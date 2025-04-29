import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap

data_dir = ""

def compare_prediction_to_initial(day, prediction_file):

    with rasterio.open(f"{data_dir}fire_inputs_2025_01_{day}.tif") as src:
        damage_init = src.read(4)[:-54, 19:] 
        damage_init_binary = (damage_init > 0).astype(np.float32)

    prediction = np.load(prediction_file)
    difference = prediction - damage_init_binary
    absolute_diff = np.abs(difference)
    print(absolute_diff.sum())
    
    n_pixels = prediction.size
    overlap = np.sum(prediction.astype(bool) & damage_init_binary.astype(bool)) / n_pixels * 100
    expansion = np.sum(prediction.astype(bool) & ~damage_init_binary.astype(bool)) / n_pixels * 100
    contraction = np.sum(~prediction.astype(bool) & damage_init_binary.astype(bool)) / n_pixels * 100
    unchanged = 100 - (expansion + contraction)

    print(f"Overlap with initial damage: {overlap:.2f}%")
    print(f"Expansion beyond initial: {expansion:.2f}%")
    print(f"Contraction from initial: {contraction:.2f}%")
    print(f"Unchanged areas: {unchanged:.2f}%")

    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(damage_init_binary, cmap='Reds')
    plt.title(f'Initial Damage (Day {day})')
    plt.colorbar()

    plt.subplot(2, 2, 2)
    plt.imshow(prediction, cmap='Reds')
    plt.title(f'Predicted Damage (Day {int(day)+1})')
    plt.colorbar()

    plt.subplot(2, 2, 3)
    plt.imshow(difference, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Difference (Red=Overprediction, Blue=Underprediction)')
    plt.colorbar(ticks=[-1, 0, 1])

    plt.subplot(2, 2, 4)
    metrics = ['Overlap', 'Expansion', 'Contraction', 'Unchanged']
    values = [overlap, expansion, contraction, unchanged]
    plt.bar(metrics, values, color=['gray', 'red', 'blue', 'green'])
    plt.ylim(0, 100)
    plt.title('Damage Change Metrics')
    plt.ylabel('Percentage of Total Area')
    
    plt.tight_layout()
    plt.show()


def fire_spread_graphic():
    days = ["08", "09", "10"]
    file_name = "fire_inputs_2025_01_{}.tif"

    def load_fuel_and_damage(file):
        with rasterio.open(file) as src:
            fuel = src.read(1)[:-2000, 5000:] 
            damage = src.read(4)[:-2000, 5000:]
        return fuel, damage


    fuel_08, damage_08 = load_fuel_and_damage(file_name.format(days[0]))
    fuel_09, damage_09 = load_fuel_and_damage(file_name.format(days[1]))
    fuel_10, damage_10 = load_fuel_and_damage(file_name.format(days[2]))

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    vegetation_cmap = ListedColormap(['#f0f0f0', '#d9f2a2', '#8cc269'])  
    fire_cmap = ListedColormap(['#00000000', '#ff0000'])  

    for ax, day, damage in zip(axs, days, [damage_08, damage_09, damage_10]):
        ax.imshow(fuel_08, cmap=vegetation_cmap, vmin=0, vmax=1)
        
        fire_mask = damage > 0.1  
        ax.imshow(fire_mask, cmap=fire_cmap, alpha=1.0, interpolation='none')
        
        ax.set_title(f"January {day}", fontsize=12, pad=10)
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def fire_layers_visualization():
    day = "08"
    input_file = f"fire_inputs_2025_01_{day}.tif"

    with rasterio.open(input_file) as src:
        fuel = src.read(1)  
        wind_x = src.read(2) 
        wind_y = src.read(3) 
        damage = src.read(4)  

    print(f"Max wind_x (x-component): {wind_x.max().item()}")
    print(f"Min wind_x (x-component): {wind_x.min().item()}")
    print(f"Avg wind_x (x-component): {wind_x.mean().item()}")
    print(f"Max wind_y (y-component): {wind_y.max().item()}")
    print(f"Min wind_y (y-component): {wind_y.min().item()}")
    print(f"Avg wind_y (y-component): {wind_y.mean().item()}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    fig.suptitle(f"Fire Input Layers - January {day}, 2025", fontsize=16)

    damage_ax = axes[0, 0]
    damage_cmap = ListedColormap(['none', 'red'])
    damage_ax.imshow(fuel, cmap='YlOrRd', alpha=0.3) 
    damage_ax.imshow(damage, cmap=damage_cmap, alpha=0.7)
    damage_ax.set_title(f"Fire Damage (Red = Burned)\nTotal Burned: {damage.sum():,} pixels")
    damage_ax.axis('off')

    fuel_ax = axes[0, 1]
    fuel_plot = fuel_ax.imshow(fuel, cmap='YlOrRd', vmin=0, vmax=np.percentile(fuel, 95))
    fuel_ax.set_title("Vegetation Burn Potential")
    plt.colorbar(fuel_plot, ax=fuel_ax, label='Burn Intensity')
    fuel_ax.axis('off')

    windx_ax = axes[0]
    windx_plot = windx_ax.imshow(wind_x, cmap='coolwarm')
    windx_ax.set_title("Wind X Component (East-West)")
    plt.colorbar(windx_plot, ax=windx_ax)
    windx_ax.axis('off')

    windy_ax = axes[1]

    windy_plot = windy_ax.imshow(wind_y, cmap='coolwarm')
    windy_ax.set_title("Wind Y Component (North-South)")
    plt.colorbar(windy_plot, ax=windy_ax)
    windy_ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"fire_layers_visualization_{day}.png", dpi=300, bbox_inches='tight')
    plt.show()