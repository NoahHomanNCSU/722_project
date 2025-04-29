import numpy as np
import rasterio
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import gc

patch_size = 100
stride = 50
batch_size = 32
epochs = 20
lr = 0.001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class FirePatchDataset(Dataset):
    def __init__(self, input_tifs, target_tifs, patch_size=100, stride=50):
        self.inputs = []
        self.targets = []

        for input_tif, target_tif in zip(input_tifs, target_tifs):
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

                    if np.sum(fire_patch > 0) > 0:
                        input_stack = np.stack([fuel_patch, wind_mag_patch, wind_dir_patch, fire_patch], axis=0)
                        self.inputs.append(input_stack.astype(np.float32))
                        self.targets.append((target_patch > 0).astype(np.float32)) 

        print(f"Total patches extracted: {len(self.inputs)}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx])
        y = torch.tensor(self.targets[idx]).unsqueeze(0) 
        return x, y

dataset = FirePatchDataset(
    input_tifs=["data/fire_inputs_2025_01_08.tif", "data/fire_inputs_2025_01_09.tif"],
    target_tifs=["data/fire_inputs_2025_01_09.tif", "data/fire_inputs_2025_01_10.tif"],
    patch_size=patch_size,
    stride=stride
)

def flatten_dataset(dataset):
    X, y = [], []
    for x_patch, y_patch in dataset:
        x_patch = x_patch.numpy().transpose(1, 2, 0).reshape(-1, 4)
        y_patch = y_patch.numpy().reshape(-1)
        X.append(x_patch)
        y.append(y_patch)
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y

X_all, y_all = flatten_dataset(dataset)
print(f"Total pixel points: {X_all.shape[0]}")

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.4, random_state=42)
print(f"Train samples: {X_train.shape}, Test samples: {X_test.shape}")

logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train, y_train)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu', max_iter=50)
mlp.fit(X_train, y_train)

def evaluate_model(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n--- {model_name} Metrics ---")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

logreg_preds = logreg.predict(X_test)
evaluate_model("Logistic Regression", y_test, logreg_preds)

mlp_preds = mlp.predict(X_test)
evaluate_model("MLP", y_test, mlp_preds)