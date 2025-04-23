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
from matplotlib.backends.backend_pdf import PdfPages
import rasterio
from sklearn.feature_extraction.image import extract_patches_2d
from skimage.util import view_as_blocks
from scipy.sparse import csr_matrix
import gc

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


def raster_to_tiles(day, patch_size):
    """Memory-efficient version using block processing"""
    with rasterio.open(f"fire_inputs_2025_01_{day}.tif") as src:
        # Read all bands with trimming
        fuel = src.read(1)[:-54, 19:]
        wind_mag = src.read(2)[:-54, 19:]
        wind_dir = src.read(3)[:-54, 19:]        
        damage_init = src.read(4)[:-54, 19:]

    next_day = f"{int(day) + 1:02d}"
    with rasterio.open(f"fire_inputs_2025_01_{next_day}.tif") as src:
        # Read all bands with trimming   
        damage_next = src.read(4)[:-54, 19:]

    if patch_size == 1:
        # Stack features and get binary damage
        X = np.stack([
            fuel, 
            wind_mag, 
            wind_dir, 
            (damage_init > 0).astype(np.float32)  # Convert to binary float
        ], axis=-1).reshape(-1, 4)
        y = (damage_next > 0).astype(np.uint8).reshape(-1)
        return X, y
    
    # Calculate number of patches
    n_rows = fuel.shape[0] // patch_size
    n_cols = fuel.shape[1] // patch_size
    n_patches = n_rows * n_cols
    
    # Pre-allocate arrays
    X = np.zeros((n_patches, 4), dtype=np.float32)
    y = np.zeros(n_patches, dtype=np.uint8)
    
    # Process in blocks
    for i, (fuel_patch, mag_patch, dir_patch, damage_init_patch, damage_next_patch) in enumerate(zip(
        view_as_blocks(fuel, (patch_size, patch_size)),
        view_as_blocks(wind_mag, (patch_size, patch_size)),
        view_as_blocks(wind_dir, (patch_size, patch_size)),
        view_as_blocks(damage_init, (patch_size, patch_size)),
        view_as_blocks(damage_next, (patch_size, patch_size))
    )):
        X[i] = [
            fuel_patch.mean(),
            mag_patch.mean(), 
            dir_patch.mean(),
            damage_init_patch.mean()
        ]
        y[i] = (damage_next_patch.max() > 0)
        
        # Free memory every 1000 patches
        if i % 1000 == 0:
            gc.collect()
    
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

def save_subgraphs(X_sub, y_sub, subgraph_size, i, j):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Vegetation plot
    veg_plot = ax1.imshow(X_sub[:,:,0], cmap='YlOrRd', vmin=0, vmax=1)
    ax1.set_title(f"Vegetation\nPosition: {i}-{i+subgraph_size}, {j}-{j+subgraph_size}")
    fig.colorbar(veg_plot, ax=ax1)
    
    # 2. Pre-damage plot (from X_sub's 4th channel)
    pre_dmg = ax2.imshow(X_sub[:,:,3], cmap='binary', vmin=0, vmax=1)
    ax2.set_title(f"Pre-Damage\nBurned={X_sub[:,:,3].sum()} pixels")
    fig.colorbar(pre_dmg, ax=ax2)
    
    # 3. Post-damage plot (from y_sub)
    post_dmg = ax3.imshow(y_sub, cmap='binary', vmin=0, vmax=1)
    ax3.set_title(f"Post-Damage\nBurned={y_sub.sum()} pixels")
    fig.colorbar(post_dmg, ax=ax3)
    
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def create_subgraphs(X_day1, y_day2, subgraph_size, stride, save=False, min_damage_patches=1):
    # Create the adjacency matrix for the subgraph (same for every subgraph).
    adj_matrix = build_static_adjacency(
        patch_grid_shape=(subgraph_size, subgraph_size),
        neighborhood=8
    )
    edge_index = torch.tensor(np.stack(adj_matrix.nonzero()), dtype=torch.long)

    # Define stride as half the subgraph size for (1/overlap)% overlap.
    n_rows, n_cols = X_day1.shape[0], X_day1.shape[1]
    subgraphs = []

    # Calculate maximum starting indices to avoid slicing beyond the boundaries.
    max_row_start = n_rows - subgraph_size + 1
    max_col_start = n_cols - subgraph_size + 1

    processed = 0

    # Use a sliding window approach with a stride equal to half of subgraph_size.
    for i in range(0, max_row_start, stride):
        for j in range(0, max_col_start, stride):
            # Define the subgraph boundaries.
            x_end = i + subgraph_size
            y_end = j + subgraph_size

            # Extract the 100x100 subsection with the given overlap.
            X_sub = X_day1[i:x_end, j:y_end, :]
            y_sub = y_day2[i:x_end, j:y_end]

            # Skip if no damage in this subgraph
            if X_sub[:,:,3].sum() < min_damage_patches:
                continue
            
            processed += 1
            # print(f"Processed subgraph {processed}: {i}-{x_end}, {j}-{y_end}")
            if save and processed < 1000: save_subgraphs(X_sub, y_sub, subgraph_size, i, j) # Save the subgraph visualization to PDF

            # Flatten features and labels.
            X_flat = X_sub.reshape(-1, 4)  # 4 features per node.
            y_flat = y_sub.reshape(-1)

            subgraphs.append(Data(
                x=torch.FloatTensor(X_flat),
                edge_index=edge_index,
                y=torch.FloatTensor(y_flat)
            ))
            
        gc.collect() # Free memory after each subgraph

    print(f"Created {len(subgraphs)} subgraphs of size {subgraph_size}x{subgraph_size} with {subgraph_size/stride}% overlap.")
    return subgraphs

def train_on_subgraphs(subgraphs, epochs=50, lr=0.001, early_stop_patience=5):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    device='cpu'
    print(f"Using device: {device}")

    # Split data (moved to GPU later)
    subgraphs = [graph.to(device) for graph in subgraphs]
    train_data, test_data = train_test_split(subgraphs, test_size=0.4, random_state=42)
    
    # Initialize model
    num_features = train_data[0].x.shape[1]
    model = FireSpreadGAT(
        num_features=num_features,
        hidden_channels=64,  # Increased capacity
        num_heads=4,
        dropout=0.3  # Added regularization
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    # Convert data to GPU batches
    def to_dev(batch):
        print(f"x: {batch.x.device}, edge_index: {batch.edge_index.device}, y: {batch.y.device}")
        return batch.to(device)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=to_dev)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, collate_fn=to_dev)

    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        for data in train_loader:
            data.x = data.x.to(device)
            data.y = data.y.to(device)
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            # Forward pass with mixed precision
            out = model(data).squeeze()
            loss = criterion(out, data.y.squeeze())
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for data in test_loader:
                out = model(data).squeeze()
                val_loss += criterion(out, data.y.squeeze()).item()
                y_true.extend(data.y.cpu().tolist())
                y_pred.extend((out > 0.5).float().cpu().tolist())
        
        # Metrics
        avg_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f'Epoch {epoch+1:03d} | '
              f'Train Loss: {avg_loss:.6f} | '
              f'Val Loss: {avg_val_loss:.6f} | '
              f'Acc: {accuracy:.4f} | '
              f'F1: {f1:.4f}')
        
        # Early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Memory cleanup
        gc.collect()

    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model.to('cpu')  # Return to CPU for inference

    
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

    # Process full dataset: 13454 x 19519
    X_day1, y_day2 = raster_to_tiles("09", patch_size=1)
    
    # Reshape to 2D grid (1300x1900 nodes)
    X_grid = X_day1.reshape(13400, 19500, 4)
    y_grid = y_day2.reshape(13400, 19500)

    # plt.figure(figsize=(12, 8))
    # plt.imshow(X_grid[:, :, 3], cmap='binary', vmin=0, vmax=1)
    # plt.title(f"Processed Fire Damage (Day 09)")
    # plt.colorbar(label='Damage (1=burned)')
    # plt.savefig(f"damage_verification_09.png", dpi=300)
    # plt.show()

    # plt.figure(figsize=(12, 8))
    # plt.imshow(y_grid, cmap='binary', vmin=0, vmax=1)
    # plt.title(f"Processed Fire Damage (Day 10)")
    # plt.colorbar(label='Damage (1=burned)')
    # plt.savefig(f"damage_verification_10.png", dpi=300)
    # plt.show()

    with PdfPages('subgraphs.pdf') as pdf:
        subgraphs = create_subgraphs(X_grid, y_grid, subgraph_size=100, stride=50, min_damage_patches=10)
    
    # Train the model
    print("\nTraining model...")
    model = train_on_subgraphs(subgraphs, epochs=50, lr=0.005)
    
