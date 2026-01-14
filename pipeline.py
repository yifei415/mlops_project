from pathlib import Path
import torch
import nibabel as nib
import numpy as np
import time
from components.inference_monai import inference_from_nii, load_monai_unet, load_resunet, load_dynunet, save_mask
from components.metrics import dice_score, precision_score, recall_score
from monai.inferers import sliding_window_inference

# -----------------------------
# 配置
# -----------------------------
DATA_DIR = Path("data/LyNoS/Benchmark/Pat1")
OUTPUT_DIR = Path("MLOPS_PROJECT/output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 多模型对比配置
models_to_test = {
    "U-Net": ("load_monai_unet", "checkpoints/unet_epoch5.pth"),
    "ResUNet": ("load_resunet", None),
    "DynUNet": ("load_dynunet", None)
}

# -----------------------------
# 设备
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# 读取 CT 与 GT
# -----------------------------
ct_files = sorted(DATA_DIR.glob("*_data.nii*"))          # 匹配 pat1_data.nii.gz
gt_files = sorted(DATA_DIR.glob("*_labels_LymphNodes*.nii*"))     # 匹配 pat1_labels_LymphNodes.nii.gz

assert len(ct_files) == len(gt_files), "CT 与 GT 数量不一致"
print(f"Found {len(ct_files)} cases to process")

# -----------------------------
# 推理 + 评估
# -----------------------------
for i, (ct_path, gt_path) in enumerate(zip(ct_files, gt_files)):
    print(f"\n▶ Processing case {i}: {ct_path.name}")
    start_case = time.time()

    # 读取 GT
    gt_mask = nib.load(gt_path).get_fdata()
    gt_mask = gt_mask[:, :, :8]  # 如果你只测试前8张slice
    gt_mask = (gt_mask > 0).astype(np.uint8)

    for model_name, (loader_func_name, ckpt_path) in models_to_test.items():
        print(f"\n--- Model: {model_name} ---")
        # 动态加载模型
        model = globals()[loader_func_name](device=device, checkpoint_path=ckpt_path)

        start_infer = time.time()
        result = inference_from_nii(
            nii_path=ct_path,
            model=model,
            device=device,
            threshold=0.5
        )
        end_infer = time.time()

        pred_mask = result['pred_mask']
        bboxes = result['bboxes']

        print(f"  -> Inference done! Time: {end_infer - start_infer:.2f} sec")
        print(f"  -> Detected {len(bboxes)} objects in this case")

        # 保存 mask
        save_mask(pred_mask, ct_path, OUTPUT_DIR / model_name / f"{ct_path.stem}_mask.nii.gz")

        # 计算指标
        dice = dice_score(pred_mask, gt_mask)
        prec = precision_score(pred_mask, gt_mask)
        rec = recall_score(pred_mask, gt_mask)

        print(
            f"Model {model_name} | Dice: {dice:.4f} | "
            f"Precision: {prec:.4f} | Recall: {rec:.4f}"
        )

    end_case = time.time()
    print(f"Total case time: {end_case - start_case:.2f} sec")
