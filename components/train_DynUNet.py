import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from monai.networks.nets import DynUNet
import nibabel as nib
from torch.cuda.amp import autocast, GradScaler

# -----------------------------
# 配置
# -----------------------------
DATA_DIR = Path(r"E:\M2\Projet R& D\project\data\LyNoS\Benchmark")
CHECKPOINT_DIR = Path("checkpoints_dynunet")
CHECKPOINT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-4

PATCH_SIZE = (64, 64, 64)
NUM_PATCHES_PER_PATIENT = 500
POS_RATIO = 0.5

UNET_DOWN_FACTOR = 4

# -----------------------------
# 工具函数
# -----------------------------
def load_nifti(path):
    return np.expand_dims(
        np.array(nib.load(path).dataobj, dtype=np.float32), 0
    )  # (1,Z,Y,X)

def normalize(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

def pad_to_multiple(img, multiple=UNET_DOWN_FACTOR):
    c, z, y, x = img.shape
    pad = (
        (0, 0),
        (0, (multiple - z % multiple) % multiple),
        (0, (multiple - y % multiple) % multiple),
        (0, (multiple - x % multiple) % multiple),
    )
    return np.pad(img, pad, mode="constant")

# -----------------------------
# Dataset
# -----------------------------
class FixedPatchDataset(Dataset):
    def __init__(self, data_list):
        self.samples = []

        for data in data_list:
            img = pad_to_multiple(normalize(load_nifti(data["image"])))
            label = pad_to_multiple(load_nifti(data["label"]))
            label = (label > 0).astype(np.float32)

            c, z, y, x = img.shape
            pz, py, px = PATCH_SIZE

            pos_needed = int(NUM_PATCHES_PER_PATIENT * POS_RATIO)
            neg_needed = NUM_PATCHES_PER_PATIENT - pos_needed

            pos_count, neg_count = 0, 0
            while pos_count < pos_needed or neg_count < neg_needed:
                z0 = np.random.randint(0, z - pz + 1)
                y0 = np.random.randint(0, y - py + 1)
                x0 = np.random.randint(0, x - px + 1)

                img_patch = img[:, z0:z0+pz, y0:y0+py, x0:x0+px]
                lbl_patch = label[:, z0:z0+pz, y0:y0+py, x0:x0+px]

                has_fg = lbl_patch.sum() > 0
                if has_fg and pos_count < pos_needed:
                    self.samples.append((img_patch, lbl_patch))
                    pos_count += 1
                elif not has_fg and neg_count < neg_needed:
                    self.samples.append((img_patch, lbl_patch))
                    neg_count += 1

        print(f"Total patches: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, label = self.samples[idx]
        return torch.tensor(img, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

# -----------------------------
# 构建数据
# -----------------------------
data_dicts = []
for pat_dir in sorted(DATA_DIR.iterdir()):
    if not pat_dir.is_dir():
        continue
    img = list(pat_dir.glob("*_data.nii*"))
    lbl = list(pat_dir.glob("*_labels_LymphNodes*.nii*"))
    if img and lbl:
        data_dicts.append({"image": str(img[0]), "label": str(lbl[0])})

print(f"Found {len(data_dicts)} patients.")

train_ds = FixedPatchDataset(data_dicts[:12])
val_ds   = FixedPatchDataset(data_dicts[12:15])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -----------------------------
# Model
# -----------------------------
model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    kernel_size=[3,3,3,3],
    strides=[1,2,2,2],
    upsample_kernel_size=[2,2,2],
    filters=[16,32,64,128,256]
).to(DEVICE)

loss_function = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([10.0]).to(DEVICE)
)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
scaler = GradScaler()

# -----------------------------
# Training
# -----------------------------
best_val = 1e9

for epoch in range(NUM_EPOCHS):
    model.train()
    train_loss = 0
    for img, label in train_loader:
        img, label = img.to(DEVICE), label.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            out = model(img)
            loss = loss_function(out, label)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()
    print(f"[Epoch {epoch+1}] Train loss: {train_loss/len(train_loader):.4f}")

    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(DEVICE), label.to(DEVICE)
            with autocast():
                val_loss += loss_function(model(img), label).item()
    val_loss /= len(val_loader)
    print(f"[Epoch {epoch+1}] Val loss: {val_loss:.4f}")

    # 保存 checkpoint
    torch.save(model.state_dict(), CHECKPOINT_DIR / f"epoch_{epoch+1}.pth")
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), CHECKPOINT_DIR / "best_dynunet.pth")
        print("Best model updated")
