# train_unet.py
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import nibabel as nib
from torch.cuda.amp import autocast, GradScaler

# -----------------------------
# 配置（测试版）
# -----------------------------
DATA_DIR = Path(r"E:\M2\Projet R& D\project\data\LyNoS\Benchmark")
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 5
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

PATCH_SIZE = (24, 24, 24)
STRIDE = (24, 24, 24)
MAX_PATCHES_PER_PATIENT = 30   # ⭐ 核心降负

# -----------------------------
# 工具函数
# -----------------------------
def load_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)

def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def add_channel(img):
    return np.expand_dims(img, 0)

def pad_to_divisible(img, div=16):
    z, y, x = img.shape
    pad_z = (div - z % div) % div
    pad_y = (div - y % div) % div
    pad_x = (div - x % div) % div
    return np.pad(img, ((0,pad_z),(0,pad_y),(0,pad_x)), mode='constant')

# -----------------------------
# Dataset（⚡轻量 patch）
# -----------------------------
class PatchDataset(Dataset):
    def __init__(self, data_list, augment=False):
        self.samples = []
        self.augment = augment

        for data in data_list:
            img = normalize(pad_to_divisible(load_nifti(data["image"])))
            label = pad_to_divisible(load_nifti(data["label"])).astype(np.float32)

            z_max, y_max, x_max = img.shape
            pz, py, px = PATCH_SIZE
            dz, dy, dx = STRIDE

            count = 0
            for z in range(0, z_max - pz + 1, dz):
                for y in range(0, y_max - py + 1, dy):
                    for x in range(0, x_max - px + 1, dx):
                        self.samples.append((
                            add_channel(img[z:z+pz, y:y+py, x:x+px]),
                            add_channel(label[z:z+pz, y:y+py, x:x+px])
                        ))
                        count += 1
                        if count >= MAX_PATCHES_PER_PATIENT:
                            break
                    if count >= MAX_PATCHES_PER_PATIENT:
                        break
                if count >= MAX_PATCHES_PER_PATIENT:
                    break

        print(f" Total patches: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return (
            torch.tensor(img, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

# -----------------------------
# 构建数据列表
# -----------------------------
data_dicts = []
for pat_dir in sorted(DATA_DIR.iterdir()):
    if not pat_dir.is_dir():
        continue
    img_files = list(pat_dir.glob("*_data.nii*"))
    label_files = list(pat_dir.glob("*_labels_LymphNodes*.nii*"))
    if img_files and label_files:
        data_dicts.append({"image": str(img_files[0]), "label": str(label_files[0])})

print(f" Found {len(data_dicts)} patients.")

# ⚡ 只用 2 个病人
train_list = data_dicts[:2]
val_list = data_dicts[2:3]

train_ds = PatchDataset(train_list)
val_ds = PatchDataset(val_list)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

# -----------------------------
# Model（⚡小 UNet）
# -----------------------------
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(8, 16, 32),
    strides=(2, 2),
).to(DEVICE)

loss_function = DiceLoss(sigmoid=True)
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
    model.train()
    train_loss = 0

    for img, label in train_loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            output = model(img)
            loss = loss_function(output, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    print(f"Training Loss: {train_loss / len(train_loader):.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            with autocast():
                output = model(img)
                loss = loss_function(output, label)
            val_loss += loss.item()

    print(f"Validation Loss: {val_loss / len(val_loader):.4f}")

    torch.save(model.state_dict(), CHECKPOINT_DIR / f"unet_epoch{epoch+1}.pth")
    print(" Checkpoint saved")
