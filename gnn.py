import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GATConv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
from sklearn.feature_extraction.image import extract_patches_2d
from scipy.sparse import csr_matrix

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# 1. Data Preparation
def create_synthetic_data(num_nodes=100, num_days=3):
    """Create synthetic data for demonstration purposes"""
    # Generate grid-like adjacency (4-connected grid)
    edge_index = []
    grid_size = int(np.sqrt(num_nodes))
    
    for i in range(grid_size):
        for j in range(grid_size):
            node = i * grid_size + j
            if i > 0:
                edge_index.append([node, (i-1)*grid_size + j])  # Up
            if i < grid_size - 1:
                edge_index.append([node, (i+1)*grid_size + j])  # Down
            if j > 0:
                edge_index.append([node, i*grid_size + (j-1)])  # Left
            if j < grid_size - 1:
                edge_index.append([node, i*grid_size + (j+1)])  # Right
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Generate node features for each day
    all_data = []
    for day in range(num_days):
        # Random features (replace with real data)
        fire_status = torch.randint(0, 2, (num_nodes, 1)).float()  # Binary
        wind_speed = torch.rand(num_nodes, 1) * 30  # 0-30 mph
        wind_direction = torch.rand(num_nodes, 1) * 360  # 0-360 degrees
        dryness = torch.rand(num_nodes, 1)  # 0-1 scale
        
        x = torch.cat([fire_status, wind_speed, wind_direction, dryness], dim=1)
        
        # For the first two days, we have both input and target
        if day < num_days - 1:
            # Target is fire status at next timestep (simple propagation rule for demo)
            y = fire_status.clone()
            # Simple propagation: adjacent nodes catch fire with some probability
            for src, dst in edge_index.t().numpy():
                if x[src, 0] > 0.5 and x[dst, 0] < 0.5:  # If src is on fire and dst isn't
                    # Probability increases with wind speed and dryness
                    prob = 0.1 + 0.01 * x[src, 1] + 0.2 * x[dst, 3]
                    if np.random.rand() < prob:
                        y[dst] = 1.0
            
            data = Data(x=x, edge_index=edge_index, y=y)
            all_data.append(data)
    
    visualize_fire_data(all_data[0], day=0, grid_size=grid_size)  # Visualize first day
    return all_data

def visualize_fire_data(data, day, grid_size=10):
    """Visualize the synthetic fire spread data for a given day"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create colormaps
    fire_cmap = ListedColormap(['green', 'red'])  # Green: no fire, Red: fire
    dry_cmap = plt.cm.YlOrBr  # Yellow to brown for dryness
    wind_cmap = plt.cm.Blues  # Blue scale for wind speed
    
    # Extract features
    fire_status = data.x[:, 0].numpy().reshape(grid_size, grid_size)
    wind_speed = data.x[:, 1].numpy().reshape(grid_size, grid_size)
    wind_dir = data.x[:, 2].numpy().reshape(grid_size, grid_size)
    dryness = data.x[:, 3].numpy().reshape(grid_size, grid_size)
    
    # Create subplots
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    ax3 = plt.subplot2grid((3, 3), (1, 2))
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    # Main fire status plot
    im1 = ax1.imshow(fire_status, cmap=fire_cmap, vmin=0, vmax=1)
    ax1.set_title(f'Fire Status (Day {day})')
    plt.colorbar(im1, ax=ax1, ticks=[0.25, 0.75], label='Fire Status')
    
    # Add wind direction arrows
    for i in range(grid_size):
        for j in range(grid_size):
            # Convert wind direction to radians (0° = east, 90° = north)
            angle_rad = np.radians(270 - wind_dir[i,j])  # Convert to math coordinates
            dx = np.cos(angle_rad) * 0.4
            dy = np.sin(angle_rad) * 0.4
            
            # Scale arrow by wind speed
            arrow_scale = wind_speed[i,j] / 30
            ax1.arrow(j, i, dx * arrow_scale, dy * arrow_scale, 
                     head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    
    # Wind speed plot
    im2 = ax2.imshow(wind_speed, cmap=wind_cmap)
    ax2.set_title('Wind Speed (mph)')
    plt.colorbar(im2, ax=ax2)
    
    # Dryness plot
    im3 = ax3.imshow(dryness, cmap=dry_cmap)
    ax3.set_title('Dryness Level')
    plt.colorbar(im3, ax=ax3)
    
    # Statistics
    ax4.axis('off')
    stats_text = (f"Day {day} Statistics:\n"
                 f"Burning cells: {int(fire_status.sum())}/{grid_size**2}\n"
                 f"Avg wind speed: {wind_speed.mean():.1f} mph\n"
                 f"Avg dryness: {dryness.mean():.2f}")
    ax4.text(0.1, 0.5, stats_text, fontsize=12, va='center')
    
    plt.tight_layout()
    plt.show()


def raster_to_graph(day, patch_size=1, clip_bounds=None):
    """
    Convert raster data to graph nodes, with optional clipping.
    
    Args:
        day (str): Day identifier (e.g., "08")
        patch_size (int): Size of each patch (default: 1x1 pixel)
        clip_bounds (tuple): (left, bottom, right, top) in raster coordinates
    
    Returns:
        X (np.ndarray): Node features [mean_fuel, mean_wind_mag, mean_wind_dir]
        y (np.ndarray): Binary damage labels
    """
    with rasterio.open(f"fire_inputs_2025_01_{day}.tif") as src:
        # Clip the raster if bounds are provided
        if clip_bounds:
            left, bottom, right, top = clip_bounds
            window = src.window(left, bottom, right, top)
            fuel = src.read(1, window=window)
            wind_mag = src.read(2, window=window)
            wind_dir = src.read(3, window=window)
            damage = src.read(4, window=window)
        else:
            fuel = src.read(1)
            wind_mag = src.read(2)
            wind_dir = src.read(3)
            damage = src.read(4)

    # Handle edge cases where patch_size > clipped array size
    if patch_size > min(fuel.shape):
        raise ValueError(f"Patch size {patch_size} is larger than clipped raster dimensions {fuel.shape}")

    # Extract patches (nodes)
    def safe_extract(data, size):
        try:
            return extract_patches_2d(data, (size, size))
        except ValueError:
            # Pad if needed (reflect mode avoids edge artifacts)
            pad_width = ((0, size - data.shape[0] % size), (0, size - data.shape[1] % size))
            padded = np.pad(data, pad_width, mode='reflect')
            return extract_patches_2d(padded, (size, size))

    patches_fuel = safe_extract(fuel, patch_size)
    patches_wind_mag = safe_extract(wind_mag, patch_size)
    patches_wind_dir = safe_extract(wind_dir, patch_size)
    patches_damage = safe_extract(damage, patch_size)

    # Node features: [mean_fuel, mean_wind_mag, mean_wind_dir]
    X = np.stack([
        patches_fuel.mean(axis=(1, 2)),
        patches_wind_mag.mean(axis=(1, 2)),
        patches_wind_dir.mean(axis=(1, 2))
    ], axis=1)

    # Node targets: Binary damage (1 if any pixel in patch is burned)
    y = (patches_damage.max(axis=(1, 2)) > 0).astype(int)

    return X, y

def build_static_adjacency(patch_grid_shape=(1, 1), neighborhood=8):
    """Create a static adjacency matrix for a grid of patches."""
    rows, cols = patch_grid_shape
    n_nodes = rows * cols
    adj = np.zeros((n_nodes, n_nodes))

    # Connect patches based on neighborhood (4 or 8)
    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  # Skip self-connections
                    if neighborhood == 4 and (di != 0 and dj != 0):
                        continue  # Skip diagonals for 4-neighborhood
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbor_id = ni * cols + nj
                        adj[node_id, neighbor_id] = 1  # Undirected connection

    return csr_matrix(adj)

# 2. GAT Model Architecture
class FireSpreadGAT(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads, dropout=0.2):
        super(FireSpreadGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
        self.fc = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # First GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # Second GAT layer
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Final prediction layer
        x = self.fc(x)
        return torch.sigmoid(x)

# 3. Training and Evaluation
def train_model(data_list, epochs=100, lr=0.01):
    # Split data into train and test
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=42)
    
    # Create DataLoader
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
    
    # Initialize model
    num_features = train_data[0].x.shape[1]
    model = FireSpreadGAT(num_features=num_features, hidden_channels=32, num_heads=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for data in train_loader:
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        # Print training progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for data in test_loader:
            pred = model(data)
            y_true.extend(data.y.view(-1).tolist())
            y_pred.extend((pred > 0.5).float().view(-1).tolist())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f'\nTest Metrics:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    
    return model

# Main execution
if __name__ == "__main__":
    # # Create synthetic data (replace with your real data)
    # print("Creating synthetic data...")
    # data_list = create_synthetic_data(num_nodes=100, num_days=3)
    # model = train_model(data, epochs=50, lr=0.005)
    # # Example prediction on new data
    # print("\nMaking a prediction on new data...")
    # new_data = create_synthetic_data(num_nodes=100, num_days=1)[0]  # Single day
    # model.eval()
    # with torch.no_grad():
    #     prediction = model(new_data)
    
    # print(f"Predicted fire spread for {prediction.shape[0]} areas:")
    # print(f"Number of areas predicted to catch fire: {(prediction > 0.5).sum().item()}")

    # Example for 2 consecutive days
    X_day1, y_day1 = raster_to_graph("08", patch_size=1, clip_bounds=(-118.501, 34, -118.5, 34.009))
    # X_day2, y_day2 = raster_to_graph("09")

    adj_matrix = build_static_adjacency(patch_grid_shape=(X_day1.shape[0], X_day1.shape[1]), neighborhood=8)
    edge_index = torch.tensor(np.stack(adj_matrix.nonzero()), dtype=torch.long)

    data = Data(
        x=torch.tensor(X_day1, dtype=torch.float),
        edge_index=edge_index,
        y=torch.tensor(y_day1, dtype=torch.float)
    )
    
    # Train the model
    print("\nTraining model...")
    model = train_model(data, epochs=50, lr=0.005)
    
