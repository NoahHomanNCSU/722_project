import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio
import matplotlib.pyplot as plt

patch_size = 100
stride = 50
batch_size = 16
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FirePatchDataset(Dataset):
    def __init__(self, input_tif_list, target_tif_list, patch_size=100, stride=50):
        self.inputs = []
        self.targets = []
        self.coords = []

        for input_tif, target_tif in zip(input_tif_list, target_tif_list):
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

                    if np.sum(target_patch > 0) > 5:
                        input_stack = np.stack([fuel_patch, wind_mag_patch, wind_dir_patch, fire_patch], axis=0)
                        self.inputs.append(input_stack.astype(np.float32))
                        self.targets.append((target_patch > 0).astype(np.float32))
                        self.coords.append((i, j))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx])
        y = torch.tensor(self.targets[idx]).unsqueeze(0)
        coord = torch.tensor(self.coords[idx])
        return x, y, coord

class FirePatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = torch.sigmoid(self.conv3(x))
        return x
    
dataset = FirePatchDataset(
    input_tif_list=["data/fire_inputs_2025_01_08.tif", "data/fire_inputs_2025_01_09.tif"],
    target_tif_list=["data/fire_inputs_2025_01_09.tif", "data/fire_inputs_2025_01_10.tif"],
    patch_size=patch_size,
    stride=stride
)

max_i = max(i for i, _ in dataset.coords) + patch_size
max_j = max(j for _, j in dataset.coords) + patch_size

canvas_height = max_i
canvas_width = max_j

model = FirePatchCNN().to(device)
model.load_state_dict(torch.load("baseline_patch_model.pt", map_location=device))
model.eval()

loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

canvas_pred = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
canvas_true = np.zeros((canvas_height, canvas_width), dtype=np.uint8)

with torch.no_grad():
    for x_batch, y_batch, coord_batch in loader:
        x_batch = x_batch.to(device)
        preds = model(x_batch)

        preds_np = (preds.cpu().numpy() > 0.5).astype(np.uint8).squeeze(1)
        targets_np = (y_batch.numpy() > 0.5).astype(np.uint8).squeeze(1)
        coords_np = coord_batch.numpy()

        for patch_pred, patch_true, (i, j) in zip(preds_np, targets_np, coords_np):
            canvas_pred[i:i+patch_size, j:j+patch_size] = patch_pred
            canvas_true[i:i+patch_size, j:j+patch_size] = patch_true

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
for ax, img, title in zip(axes, [canvas_true, canvas_pred], ['Ground Truth', 'Predicted']):
    rgb_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    rgb_img[img == 0] = [0, 255, 0]  
    rgb_img[img == 1] = [255, 0, 0]  
    ax.imshow(rgb_img)
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()