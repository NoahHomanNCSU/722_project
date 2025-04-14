import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GATConv, global_mean_pool  # Using Graph Attention
from torch_geometric.utils import to_networkx
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import math
from matplotlib import pyplot as plt

class FireSpreadGNN(nn.Module):
    def __init__(self, node_feature_dim, edge_feature_dim, hidden_dim=64, heads=4):
        super(FireSpreadGNN, self).__init__()
        
        # Node encoder (wind, vegetation, current fire state)
        self.node_encoder = nn.Linear(node_feature_dim, hidden_dim)
        
        # Edge encoder (distance, wind direction, etc.)
        self.edge_encoder = nn.Linear(edge_feature_dim, hidden_dim)
        
        # Graph Attention Layers (GAT)
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=heads, edge_dim=hidden_dim)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, edge_dim=hidden_dim)
        
        # Edge predictor (MLP for fire spread probability)
        self.edge_predictor = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Outputs probability [0,1]
        )
        
        # Node state predictor (MLP for fire at t+1)
        self.node_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Binary classification
        )
    
    def forward(self, data):
        # Encode node features
        x = self.node_encoder(data.x)
        
        # Encode edge features (if available)
        edge_attr = self.edge_encoder(data.edge_attr) if hasattr(data, 'edge_attr') else None
        
        # Graph convolutions
        x = F.relu(self.gat1(x, data.edge_index, edge_attr=edge_attr))
        x = F.relu(self.gat2(x, data.edge_index, edge_attr=edge_attr))
        
        # Predict edge weights (fire spread probability)
        src, dst = data.edge_index
        edge_features = torch.cat([x[src], x[dst]], dim=1)
        edge_pred = self.edge_predictor(edge_features).squeeze()
        
        # Predict node states (fire at t+1)
        node_pred = self.node_predictor(x).squeeze()
        
        return node_pred, edge_pred

    # Training loop
    def train():
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            node_pred, edge_pred = model(data)
            
            # Node loss (fire state at t+1)
            node_loss = criterion(node_pred, data.y)
            
            # Edge loss (optional: if you have ground-truth edge weights)
            edge_loss = 0  # Add if edge labels are available
            
            total_loss = node_loss + edge_loss
            total_loss.backward()
            optimizer.step()

    # Evaluation
    def test():
        model.eval()
        correct = 0
        for data in test_loader:
            node_pred, _ = model(data)
            pred = (node_pred > 0.5).float()
            correct += (pred == data.y).sum().item()
        accuracy = correct / len(test_loader.dataset)
        return accuracy    
    

def grid_to_graph(features):
    """
    Converts a 2D grid with features to a graph structure using 8-neighbor connectivity.
    
    :param features: Tensor of shape [Nx, Ny, 4] — [fire, wind_x, wind_y, fuel]
    :return: PyTorch Geometric Data object
    """
    Nx, Ny, num_features = features.shape
    num_nodes = Nx * Ny
    print(f"Grid shape: {Nx}x{Ny}, Number of nodes: {num_nodes}")

    node_feats = features.view(-1, num_features)  # [num_nodes, 4]

    edge_index = []
    edge_attr = []

    # 8 directions (dx, dy)
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  ( 0, -1),          ( 0, 1),
                  ( 1, -1), ( 1, 0), ( 1, 1)]

    for i in range(Nx):
        for j in range(Ny):
            idx = i * Ny + j

            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < Nx and 0 <= nj < Ny:
                    n_idx = ni * Ny + nj

                    # Get local features
                    fire_from = features[i, j, 0]
                    wind_x, wind_y = features[i, j, 1], features[i, j, 2]
                    fuel_to = features[ni, nj, 3]

                    # Vector from source to neighbor
                    dir_vec = torch.tensor([dx, dy], dtype=torch.float)
                    dir_vec = dir_vec / dir_vec.norm() if dir_vec.norm() > 0 else dir_vec

                    # Wind direction vector
                    wind_vec = torch.tensor([wind_x, wind_y])

                    # Spread potential: dot product of wind direction with neighbor direction,
                    # scaled by fuel and whether source is on fire
                    wind_alignment = torch.dot(wind_vec, dir_vec).clamp(min=0.0)
                    spread_potential = fire_from * wind_alignment * fuel_to

                    edge_index.append([idx, n_idx])
                    edge_attr.append([spread_potential.item()])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, E]
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)                    # [E, 1]

    data = Data(
        x=node_feats,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    G = to_networkx(data, to_undirected=True)

    # Layout: each node at its grid coordinate
    pos = {i: (i % Ny, -i // Ny) for i in range(Nx * Ny)}

    # Draw
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=False, node_color='red', node_size=50, edge_color='gray', width=0.5)
    plt.title("GNN Graph (10x10 Grid with 8-Neighbor Connectivity)")
    plt.axis("off")
    plt.show()

    return data

def make_grid_data():
    '''
    3D grid data for the fire spread problem.
    Dim 1: x-coordinates
    Dim 2: y-coordinates
    Dim 3: 3 for the following info: windx, windy, fire fuel potential
    '''
    Nx, Ny = 35, 35
    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 1.0
    x = torch.linspace(x_min, x_max, Nx)
    y = torch.linspace(y_min, y_max, Ny)
    x_grid, y_grid = torch.meshgrid(x, y, indexing="ij")  # Shape: [Nx, Ny]

    # Initialize feature tensor
    features = torch.zeros((Nx, Ny, 4))  # [fire, wind_x, wind_y, fuel_potential]

    # Smooth wind patterns
    features[..., 1] = 0.5 + 0.5 * torch.sin(2 * math.pi * x_grid)  # wind_x
    features[..., 2] = 0.5 + 0.5 * torch.cos(2 * math.pi * y_grid)  # wind_y

    # Fuel potential: random
    features[..., 3] = torch.rand((Nx, Ny))

    # Initial fire at center
    features[Nx//2, Ny//2, 0] = 1

    return x_grid, y_grid, features

def plot_grid_data(x_grid, y_grid, features):
    # Extract individual feature layers
    fire = features[..., 0]
    wind_x = features[..., 1]
    wind_y = features[..., 2]
    fuel = features[..., 3]

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Wind vector field (quiver plot)
    axs[0].quiver(x_grid, y_grid, wind_x, wind_y, color='blue')
    axs[0].set_title("Wind Vector Field")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    axs[0].set_aspect("equal")

    # 2. Fire status heatmap
    im1 = axs[1].imshow(fire.T, origin='lower', cmap='hot', extent=[0, 1, 0, 1])
    axs[1].set_title("Fire Status (0 = off, 1 = on)")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    fig.colorbar(im1, ax=axs[1])

    # 3. Fuel potential heatmap
    im2 = axs[2].imshow(fuel.T, origin='lower', cmap='YlGn', extent=[0, 1, 0, 1])
    axs[2].set_title("Fuel Potential (0 to 1)")
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    fig.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
  
    x, y, f = make_grid_data()
    plot_grid_data(x, y, f)
    data = grid_to_graph(f)

    quit(0)
        
    # Example usage:
    grids_t0 = [...]
    grids_t1 = [...] 
    wind_dirs = [...]

    # Create PyG graphs
    graphs = [grid_to_graph(t0, t1, wd) for t0, t1, wd in zip(grids_t0, grids_t1, wind_dirs)]

    # Split into train/test
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.2)

    # DataLoader
    train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=32)

    # Initialize model
    model = FireSpreadGNN(node_feature_dim=3, edge_feature_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()  # Binary cross-entropy for fire prediction

    # Train for N epochs
    for epoch in range(100):
        model.train()
        acc = model.test()
        print(f"Epoch {epoch}, Test Accuracy: {acc:.4f}")