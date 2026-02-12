# components/inference_monai.py
import torch
import torch.nn as nn
import numpy as np
from scipy import ndimage
from pathlib import Path
import nibabel as nib
from monai.networks.nets import UNet
from monai.inferers import sliding_window_inference
from monai.networks.nets import DynUNet
from scipy.ndimage import label
#from monai.networks.nets import ResUNet

class ResidualBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out

class SimpleResUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=(16,32,64)):
        super().__init__()
        self.enc1 = ResidualBlock3D(in_channels, features[0])
        self.enc2 = ResidualBlock3D(features[0], features[1])
        self.enc3 = ResidualBlock3D(features[1], features[2])
        self.pool = nn.MaxPool3d(2)

        # 上采样
        self.up2 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)

        # 调整 skip channel
        self.conv_skip2 = nn.Conv3d(features[1], features[1], 1)
        self.conv_skip1 = nn.Conv3d(features[0], features[0], 1)

        self.dec2 = ResidualBlock3D(features[1], features[1])
        self.dec1 = ResidualBlock3D(features[0], features[0])
        self.final = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        d2 = self.dec2(self.up2(e3) + self.conv_skip2(e2))
        d1 = self.dec1(self.up1(d2) + self.conv_skip1(e1))

        out = self.final(d1)
        return out


def load_resunet(device='cpu', checkpoint_path=None):
    model = SimpleResUNet3D().to(device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def load_dynunet(device='cpu', checkpoint_path=None):
    """
    轻量版 DynUNet 3D, 用于测试 / 对比
    """
    model = DynUNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        kernel_size=[3, 3, 3, 3],    # 4 层卷积核
        strides=[1, 2, 2, 2],        # 下采样步长
        upsample_kernel_size=[2, 2, 2],  
        filters=[16, 32, 64, 128],   # 通道数，可根据显存调小
        res_block=True                # 启用残差块
    ).to(device)

    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=False)

    model.eval()
    return model



# 3D UNet
# ---------------------------
# Mask → Bounding Box
# ---------------------------
def mask_to_bboxes(mask_volume):
    bboxes = []
    labeled, num_features = ndimage.label(mask_volume)
    for i in range(1, num_features + 1):
        coords = np.argwhere(labeled == i)
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)
        bboxes.append([x_min, y_min, z_min, x_max, y_max, z_max])
    return bboxes

# ---------------------------
# 加载 MONAI UNet 模型
# ---------------------------
def load_monai_unet(device='cpu', checkpoint_path=None):
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64),
        #channels=(16, 32, 64, 128),
        strides=(2, 2),
    ).to(device)
    if checkpoint_path is not None:
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state, strict=False)
    model.eval()
    return model







# ---------------------------
# 单病例 NIfTI 推理
# ---------------------------
def inference_from_nii_clean(
    nii_path,
    model,
    device="cpu",
    threshold=0.5,
    min_size=100,
    roi_size=(64, 64, 64),
    overlap=0.25
):
    """
    3D UNet Safe Inference with:
    - Volume padding to match model downsampling
    - Sliding window inference
    - Sigmoid threshold
    - Connected component filtering
    """
    nii_path = Path(nii_path)

    # ---------- Load ----------
    img = nib.load(str(nii_path)).get_fdata().astype(np.float32)
    print("Original volume shape:", img.shape)

    # ---------- Normalize ----------
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # ---------- Pad to be multiple of UNet downsampling (16 for 3 levels) ----------
    def pad_to_multiple(vol, multiple=16):
        z, y, x = vol.shape
        pad_z = (multiple - z % multiple) % multiple
        pad_y = (multiple - y % multiple) % multiple
        pad_x = (multiple - x % multiple) % multiple
        vol_padded = np.pad(vol,
                            ((0, pad_z), (0, pad_y), (0, pad_x)),
                            mode='constant')
        return vol_padded, (z, y, x)
    
    img, orig_shape = pad_to_multiple(img, multiple=16)
    print("Padded volume shape:", img.shape)

    # ---------- To tensor ----------
    x = torch.from_numpy(img[None, None, ...]).to(device)

    # ---------- Sliding window inference ----------
    model.eval()
    with torch.no_grad():
        logits = sliding_window_inference(
            x,
            roi_size=roi_size,
            sw_batch_size=1,
            predictor=model,
            overlap=overlap,
        )
        probs = torch.sigmoid(logits)[0,0].cpu().numpy()

    # ---------- Threshold ----------
    pred_mask = (probs > threshold).astype(np.uint8)

    # ---------- Connected component filtering ----------
    labeled, num = label(pred_mask)
    clean_mask = np.zeros_like(pred_mask, dtype=np.uint8)
    for i in range(1, num+1):
        comp = (labeled == i)
        if comp.sum() >= min_size:
            clean_mask[comp] = 1

    # ---------- Crop back to original shape ----------
    z, y, x = orig_shape
    clean_mask = clean_mask[:z, :y, :x]

    # ---------- BBoxes ----------
    bboxes = mask_to_bboxes(clean_mask)

    print(f"Predicted objects: {len(bboxes)}, sum of voxels: {clean_mask.sum()}")

    return {
        "pred_mask": clean_mask,
        "bboxes": bboxes,
        "num_objects": len(bboxes)
    }

# ---------------------------
# 保存 mask 到 NIfTI
# ---------------------------
def save_mask(mask, reference_nii_path, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ref_img = nib.load(str(reference_nii_path))
    mask_nii = nib.Nifti1Image(mask, affine=ref_img.affine)
    nib.save(mask_nii, str(out_path))
