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
import psutil
import time


DATA_DIR = ""

torch.manual_seed(42)
np.random.seed(42)

def create_synthetic_data(num_nodes=100, num_days=3):
    """Create synthetic data for demonstration purposes"""
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
    
    all_data = []
    for day in range(num_days):
        fire_status = torch.randint(0, 2, (num_nodes, 1)).float()  # Binary
        wind_speed = torch.rand(num_nodes, 1) * 30  # 0-30 mph
        wind_direction = torch.rand(num_nodes, 1) * 360  # 0-360 degrees
        dryness = torch.rand(num_nodes, 1)  # 0-1 scale
        
        x = torch.cat([fire_status, wind_speed, wind_direction, dryness], dim=1)
        
        if day < num_days - 1:
            y = fire_status.clone()
            for src, dst in edge_index.t().numpy():
                if x[src, 0] > 0.5 and x[dst, 0] < 0.5:  # If src is on fire and dst isn't
                    prob = 0.1 + 0.01 * x[src, 1] + 0.2 * x[dst, 3]
                    if np.random.rand() < prob:
                        y[dst] = 1.0
            
            data = Data(x=x, edge_index=edge_index, y=y)
            all_data.append(data)
    
    visualize_fire_data(all_data[0], day=0, grid_size=grid_size)
    return all_data

def visualize_fire_data(data, day, grid_size=10):
    """Visualize the synthetic fire spread data for a given day"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create colormaps
    fire_cmap = ListedColormap(['green', 'red'])
    dry_cmap = plt.cm.YlOrBr  
    wind_cmap = plt.cm.Blues
    
    fire_status = data.x[:, 0].numpy().reshape(grid_size, grid_size)
    wind_speed = data.x[:, 1].numpy().reshape(grid_size, grid_size)
    wind_dir = data.x[:, 2].numpy().reshape(grid_size, grid_size)
    dryness = data.x[:, 3].numpy().reshape(grid_size, grid_size)
    
    ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=2, rowspan=2)
    ax2 = plt.subplot2grid((3, 3), (0, 2))
    ax3 = plt.subplot2grid((3, 3), (1, 2))
    ax4 = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    
    im1 = ax1.imshow(fire_status, cmap=fire_cmap, vmin=0, vmax=1)
    ax1.set_title(f'Fire Status (Day {day})')
    plt.colorbar(im1, ax=ax1, ticks=[0.25, 0.75], label='Fire Status')
    
    for i in range(grid_size):
        for j in range(grid_size):
            angle_rad = np.radians(270 - wind_dir[i,j])
            dx = np.cos(angle_rad) * 0.4
            dy = np.sin(angle_rad) * 0.4
            
            arrow_scale = wind_speed[i,j] / 30
            ax1.arrow(j, i, dx * arrow_scale, dy * arrow_scale, 
                     head_width=0.2, head_length=0.2, fc='blue', ec='blue')
    
    im2 = ax2.imshow(wind_speed, cmap=wind_cmap)
    ax2.set_title('Wind Speed (mph)')
    plt.colorbar(im2, ax=ax2)
    
    im3 = ax3.imshow(dryness, cmap=dry_cmap)
    ax3.set_title('Dryness Level')
    plt.colorbar(im3, ax=ax3)
    
    ax4.axis('off')
    stats_text = (f"Day {day} Statistics:\n"
                 f"Burning cells: {int(fire_status.sum())}/{grid_size**2}\n"
                 f"Avg wind speed: {wind_speed.mean():.1f} mph\n"
                 f"Avg dryness: {dryness.mean():.2f}")
    ax4.text(0.1, 0.5, stats_text, fontsize=12, va='center')
    
    plt.tight_layout()
    plt.show()

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
    # pdf.savefig(fig)
    plt.close(fig)


def raster_to_tiles(day, patch_size):
    """Memory-efficient version using block processing"""
    with rasterio.open(f"{DATA_DIR}fire_inputs_2025_01_{day}.tif") as src:
        fuel = src.read(1)[:-54, 19:]
        wind_x = src.read(2)[:-54, 19:]
        wind_y = src.read(3)[:-54, 19:]        
        damage_init = src.read(4)[:-54, 19:]

    next_day = f"{int(day) + 1:02d}"
    with rasterio.open(f"{DATA_DIR}fire_inputs_2025_01_{next_day}.tif") as src:
        damage_next = src.read(4)[:-54, 19:]

    if patch_size == 1:
        X = np.stack([
            fuel, 
            wind_x, 
            wind_y, 
            (damage_init > 0).astype(np.float32) 
        ], axis=-1)
        y = (damage_next > 0).astype(np.float32)
        return X, y
    
    n_rows = fuel.shape[0] // patch_size
    n_cols = fuel.shape[1] // patch_size
    n_patches = n_rows * n_cols
    
    X = np.zeros((n_patches, 4), dtype=np.float32)
    y = np.zeros(n_patches, dtype=np.uint8)
    
    for i, (fuel_patch, mag_patch, dir_patch, damage_init_patch, damage_next_patch) in enumerate(zip(
        view_as_blocks(fuel, (patch_size, patch_size)),
        view_as_blocks(wind_x, (patch_size, patch_size)),
        view_as_blocks(wind_y, (patch_size, patch_size)),
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
        
        if i % 1000 == 0:
            gc.collect()
    
    return X, y


def build_static_adjacency(patch_grid_shape=(1, 1), neighborhood=8):
    """Create a static adjacency matrix for a grid of patches."""
    rows, cols = patch_grid_shape
    n_nodes = rows * cols
    adj = np.zeros((n_nodes, n_nodes))

    for i in range(rows):
        for j in range(cols):
            node_id = i * cols + j
            for di in [-1, 0, 1]:
                for dj in [-1, 0, 1]:
                    if di == 0 and dj == 0:
                        continue  
                    if neighborhood == 4 and (di != 0 and dj != 0):
                        continue  
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols:
                        neighbor_id = ni * cols + nj
                        adj[node_id, neighbor_id] = 1 

    return csr_matrix(adj)


def create_subgraphs(day, subgraph_size, stride, min_damage_patches):
    X_day1, y_day2 = raster_to_tiles(day, patch_size=1)
    # X_day2, y_day3 = raster_to_tiles("09", patch_size=1)

    n_rows, n_cols = X_day1.shape[0], X_day1.shape[1]
    subgraphs = []

    max_row_start = n_rows - subgraph_size + 1
    max_col_start = n_cols - subgraph_size + 1

    processed = 0

    for i in range(0, max_row_start, stride):
        for j in range(0, max_col_start, stride):
            x_end = i + subgraph_size
            y_end = j + subgraph_size

            X_sub = X_day1[i:x_end, j:y_end, :]
            y_sub = y_day2[i:x_end, j:y_end]

            if y_sub[:, :].sum() < min_damage_patches:
                continue
            
            X_flat = torch.from_numpy(X_sub.reshape(-1, 4).astype(np.float32))
            y_flat = torch.from_numpy(y_sub.reshape(-1).astype(np.float32))

            subgraphs.append(Data(
                x=X_flat,
                edge_index=None,
                y=y_flat
            ))

            processed += 1

            if processed % 100 == 0:
                print(f"Processed subgraph {processed}: {i}-{x_end}, {j}-{y_end}")

        gc.collect()

    print(f"Created {len(subgraphs)} subgraphs of size {subgraph_size}x{subgraph_size} with {stride/subgraph_size:.2f} overlap.")
    return subgraphs


def train_on_subgraphs(day, subgraph_size, stride, min_damage_patches=5, epochs=20, lr=0.005, early_stop_patience=5):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    adj_matrix = build_static_adjacency(
        patch_grid_shape=(subgraph_size, subgraph_size),
        neighborhood=8
    )
    edge_index = torch.tensor(np.stack(adj_matrix.nonzero()), dtype=torch.long).to(device)
    subgraphs = create_subgraphs(day, subgraph_size=subgraph_size, stride=stride, min_damage_patches=min_damage_patches)
    if device.type != "cpu": subgraphs = [graph.to(device) for graph in subgraphs]

    num_features = subgraphs[0].x.shape[1]
    model = FireSpreadGAT(
        num_features=num_features,
        hidden_channels=64,  
        num_heads=4,
        dropout=0.3  
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
    
    train_data, test_data = train_test_split(subgraphs, test_size=0.4, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True) # if GPU: collate_fn=to_dev
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0

        start_time = time.time()
        
        for data in train_loader:  
            data.edge_index = edge_index

            optimizer.zero_grad(set_to_none=True) 
            
            out = model(data).squeeze()
            loss = criterion(out, data.y.squeeze())
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 
            optimizer.step()
            epoch_loss += loss.item()

            gc.collect()

        end_time = time.time()
        epoch_time = end_time - start_time
        
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        
        with torch.no_grad():
            for data in test_loader:
                data.edge_index = edge_index
                out = model(data).squeeze()
                val_loss += criterion(out, data.y.squeeze()).item()
                y_true.extend(data.y.cpu().tolist())
                y_pred.extend((out > 0.5).float().cpu().tolist())

        avg_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(test_loader)
        scheduler.step(avg_val_loss)
        
        accuracy = accuracy_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f'Epoch {epoch+1:03d} | '
              f'Train Loss: {avg_loss:.6f} | '
              f'Val Loss: {avg_val_loss:.6f} | '
              f'Acc: {accuracy:.4f} | '
              f'Recall: {recall:.4f} | '
              f'F1: {f1:.4f} | '
              f'Time: {epoch_time:.2f}s | ')

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_{subgraph_size}_{day}.pt')
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        gc.collect()

    model.load_state_dict(torch.load(f'best_model_{subgraph_size}_{day}.pt'))
    return model.to('cpu') 

    
class FireSpreadGAT(nn.Module):
    def __init__(self, num_features, hidden_channels, num_heads, dropout=0.2):
        super(FireSpreadGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout)
        self.fc = nn.Linear(hidden_channels, 1)
        self.dropout = dropout
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        x = self.fc(x)
        return torch.sigmoid(x)


def visualize_predictions(subgraph_size, day, best_model):
    X_day1, y_day2 = raster_to_tiles(day, patch_size=1)

    # subgraphs = create_subgraphs(X_day1=X_day09, y_day2=y_day10, subgraph_size=100, stride=100, min_damage_patches=0)

    model = FireSpreadGAT(
        num_features=X_day1.shape[2],
        hidden_channels=64, 
        num_heads=4,
        dropout=0.3 
    ).to("cpu")
    model.load_state_dict(torch.load(f'{DATA_DIR}{best_model}'))
    model.eval()

    adj_matrix = build_static_adjacency(
        patch_grid_shape=(subgraph_size, subgraph_size),
        neighborhood=8
    )
    edge_index = torch.tensor(np.stack(adj_matrix.nonzero()), dtype=torch.long)

    n_rows, n_cols = X_day1.shape[0], X_day1.shape[1]

    max_row_start = n_rows - subgraph_size + 1
    max_col_start = n_cols - subgraph_size + 1

    processed = 0
    full_pred = np.zeros_like(y_day2)
    for i in range(0, max_row_start, subgraph_size):
        for j in range(0, max_col_start, subgraph_size):
            x_end = i + subgraph_size
            y_end = j + subgraph_size


            X_sub = X_day1[i:x_end, j:y_end, :]
            
            X_flat = X_sub.reshape(-1, 4) 

            subgraph = Data(
                x=torch.FloatTensor(X_flat),
                edge_index=edge_index
            )
            processed += 1

            out = model(subgraph).squeeze()
            pred = (out > 0.5).float() 
            
            full_pred[i:x_end, j:y_end] = pred.cpu().numpy().reshape(subgraph_size, subgraph_size)

            print(f"Processed subgraph {processed}: {i}-{x_end}, {j}-{y_end}")

        gc.collect() 

    np.save(f'pred_{best_model}_{int(day) + 1:02d}.npy', full_pred)
    print(f"Saved predictions for day {day} + 1")
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(y_day2, cmap='Reds')
    plt.title('Ground Truth')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(full_pred, cmap='Reds')
    plt.title('Model Predictions')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    comparison = np.zeros((*y_day2.shape, 3))
    comparison[..., 0] = (full_pred.astype(bool) & y_day2.astype(bool)) 
    comparison[..., 1] = (~full_pred.astype(bool) & y_day2.astype(bool))  
    comparison[..., 2] = (full_pred.astype(bool) & ~y_day2.astype(bool))  
    
    plt.imshow(comparison)
    plt.title('Comparison (TP=Red, FN=Green, FP=Blue)')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("\nTraining model: day 08 -> 09, subgraph size 100")
    train_on_subgraphs(day="08", subgraph_size=100, stride=50)

    print("\nTraining model: day 09 -> 10, subgraph size 100")
    train_on_subgraphs(day="09", subgraph_size=100, stride=50)

    print("\nTraining model: day 08 -> 9, subgraph size 200")
    train_on_subgraphs(day="08", subgraph_size=200, stride=50)

    print("\nTraining model: day 9 -> 10, subgraph size 200")
    train_on_subgraphs(day="09", subgraph_size=200, stride=50)

    visualize_predictions(subgraph_size=100, day="08", best_model='best_model_100_08.pt')
    visualize_predictions(subgraph_size=100, day="09", best_model='best_model_100_09.pt')
    visualize_predictions(subgraph_size=100, day="08", best_model='best_model_200_08.pt')
    visualize_predictions(subgraph_size=100, day="09", best_model='best_model_200_08.pt')

    # full_pred =  np.load("pred_best_model_200_08.pt_10.npy")
    # _, y_day2 = raster_to_tiles("09", patch_size=1)
    # # Create visualization
    # plt.figure(figsize=(15, 5))
    
    # # Ground truth
    # plt.subplot(1, 3, 1)
    # plt.imshow(y_day2, cmap='Reds')
    # plt.title('Ground Truth')
    # plt.colorbar()
    
    # # Predictions
    # plt.subplot(1, 3, 2)
    # plt.imshow(full_pred, cmap='Reds')
    # plt.title('Model Predictions')
    # plt.colorbar()
    
    # # Overlay comparison
    # plt.subplot(1, 3, 3)
    # # Create RGB image where:
    # # - Red = True Positive (predicted 1, truth 1)
    # # - Blue = False Positive (predicted 1, truth 0)
    # # - Green = False Negative (predicted 0, truth 1)
    # comparison = np.zeros((*y_day2.shape, 3))
    # comparison[..., 0] = (full_pred.astype(bool) & y_day2.astype(bool))  # Red channel - True positives
    # comparison[..., 1] = (~full_pred.astype(bool) & y_day2.astype(bool))  # Green channel - False negatives
    # comparison[..., 2] = (full_pred.astype(bool) & ~y_day2.astype(bool))  # Blue channel - False positives
    
    # plt.imshow(comparison)
    # plt.title('Comparison (TP=Red, FN=Green, FP=Blue)')
    
    # plt.tight_layout()
    # plt.show()


