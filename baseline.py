import numpy as np
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import gc

# --- Settings ---
patch_size = 100
stride = 50
batch_size = 16
epochs = 20
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- Dataset Class ---
class FirePatchDataset(Dataset):
    def __init__(self, input_tif, target_tif, patch_size=100, stride=50):
        self.inputs = []
        self.targets = []

        with rasterio.open(input_tif) as src_in:
            fuel = src_in.read(1)
            wind_mag = src_in.read(2)
            wind_dir = src_in.read(3)
            fire_mask = src_in.read(4)

        with rasterio.open(target_tif) as src_out:
            fire_target = src_out.read(4)

        h, w = fuel.shape
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                fuel_patch = fuel[i:i+patch_size, j:j+patch_size]
                wind_mag_patch = wind_mag[i:i+patch_size, j:j+patch_size]
                wind_dir_patch = wind_dir[i:i+patch_size, j:j+patch_size]
                fire_patch = fire_mask[i:i+patch_size, j:j+patch_size]
                target_patch = fire_target[i:i+patch_size, j:j+patch_size]

                # Only keep if there is fire present in input patch
                if np.sum(fire_patch > 0) > 0:
                    input_stack = np.stack([fuel_patch, wind_mag_patch, wind_dir_patch, fire_patch], axis=0)
                    self.inputs.append(input_stack.astype(np.float32))
                    self.targets.append((target_patch > 0).astype(np.float32))  # Binary 0/1

        print(f"Total patches extracted: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx])
        y = torch.tensor(self.targets[idx]).unsqueeze(0)  # (1, H, W)
        return x, y

# --- Model ---
class FirePatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)  # Output 1 channel

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))  # Predict per-pixel probability
        return x

# --- Load Data ---
train_dataset = FirePatchDataset(
    input_tif="data/fire_inputs_2025_01_08.tif",
    target_tif="data/fire_inputs_2025_01_09.tif",
    patch_size=patch_size,
    stride=stride
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = FirePatchDataset(
    input_tif="data/fire_inputs_2025_01_09.tif",
    target_tif="data/fire_inputs_2025_01_10.tif",
    patch_size=patch_size,
    stride=stride
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# --- Train Model ---
model = FirePatchCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.BCELoss()

for epoch in range(epochs):
    model.train()
    running_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        preds = model(x_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {running_loss / len(train_loader):.4f}")
    gc.collect()

# --- Evaluate ---
model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch)

        preds = preds.cpu().numpy() > 0.5
        targets = y_batch.cpu().numpy()

        y_true.append(targets.flatten())
        y_pred.append(preds.flatten())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("\n--- Test Metrics ---")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 Score: {f1:.4f}")

# --- Optional: Save model ---
torch.save(model.state_dict(), "baseline_patch_model.pt")
