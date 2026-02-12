from pathlib import Path
import torch
import nibabel as nib
import numpy as np
import csv
import time

from components.inference_monai import (
    inference_from_nii_clean,
    load_monai_unet,
    load_resunet,
    load_dynunet,
    save_mask
)
from components.metrics import dice_score, precision_score, recall_score


# =============================
# 数据集选择（单选）
# =============================
DATA_ROOT = Path("data/LyNoS/Benchmark")
assert DATA_ROOT.exists(), f"Dataset root not found: {DATA_ROOT}"

dataset_dirs = sorted([d for d in DATA_ROOT.iterdir() if d.is_dir()])[:3]

print("\nAvailable datasets:")
for idx, d in enumerate(dataset_dirs):
    print(f"  [{idx}] {d.name}")

user_input = input("\nPlease enter the dataset number to use: ").strip()
assert user_input.isdigit(), "Please enter a valid dataset index"

dataset_idx = int(user_input)
assert 0 <= dataset_idx < len(dataset_dirs), "Dataset index out of range"

DATA_DIR = dataset_dirs[dataset_idx]
print(f"\nSelected dataset: {DATA_DIR.name}")


# =============================
# 输出与报告
# =============================
OUTPUT_DIR = Path("MLOPS_PROJECT/output").resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "inference_report.csv"

if not REPORT_FILE.exists():
    with open(REPORT_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Dataset", "Case", "Model",
            "GT_sum", "Pred_sum",
            "Dice", "Precision", "Recall",
            "Num_Detected_Objects",
            "Inference_Time_sec",
            "Total_Case_Time_sec"
        ])


# =============================
# 模型配置
# =============================
models_to_test = {
    "U-Net": ("load_monai_unet", "checkpoints/best_unet.pth"),
    "ResUNet": ("load_resunet", None),
    "DynUNet": ("load_dynunet", "checkpoints_dynunet/best_dynunet.pth")
}

print("\nAvailable models:")
model_names = list(models_to_test.keys())
for idx, name in enumerate(model_names):
    print(f"  [{idx}] {name}")

user_input = input(
    "\nPlease enter the model number(s) to use "
    "(multiple selections supported, e.g. 0,2; Enter for all): "
).strip()

if user_input == "":
    selected_models = models_to_test
else:
    selected_indices = [int(i) for i in user_input.split(",")]
    selected_models = {
        model_names[i]: models_to_test[model_names[i]]
        for i in selected_indices
    }

print("\nSelected models:")
for name in selected_models:
    print(f"  - {name}")


# =============================
# 设备
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")


# =============================
# 读取 CT / GT
# =============================
ct_files = sorted(DATA_DIR.glob("*_data.nii*"))
gt_files = sorted(DATA_DIR.glob("*_labels_LymphNodes*.nii*"))

assert len(ct_files) == len(gt_files), "CT 与 GT 数量不一致"
print(f"\nFound {len(ct_files)} cases to process")


# =============================
# 推理 + 评估
# =============================
for case_idx, (ct_path, gt_path) in enumerate(zip(ct_files, gt_files)):
    print(f"\n▶ Processing case {case_idx}: {ct_path.name}")
    start_case = time.time()

    # 读取 GT
    gt_mask = nib.load(gt_path).get_fdata()
    gt_mask = (gt_mask > 0).astype(np.uint8)

    for model_name, (loader_func_name, ckpt_path) in selected_models.items():
        print(f"\n--- Model: {model_name} ---")

        # 加载模型
        model = globals()[loader_func_name](
            device=device,
            checkpoint_path=ckpt_path
        )

        # 推理
        start_infer = time.time()
        result = inference_from_nii_clean(
            str(ct_path),
            model,
            device=device
        )
        end_infer = time.time()

        pred_mask = result["pred_mask"]
        bboxes = result["bboxes"]

        print(f"  -> Inference time: {end_infer - start_infer:.2f} sec")
        print(f"  -> Detected objects: {len(bboxes)}")
        print(f"  -> GT sum: {gt_mask.sum()} | Pred sum: {pred_mask.sum()}")

        # 保存 mask（按 dataset / model 分目录）
        model_output_dir = OUTPUT_DIR / DATA_DIR.name / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        save_mask(
            pred_mask,
            ct_path,
            model_output_dir / f"{ct_path.stem}_mask.nii.gz"
        )

        # 指标
        dice = dice_score(pred_mask, gt_mask)
        prec = precision_score(pred_mask, gt_mask)
        rec = recall_score(pred_mask, gt_mask)

        print(
            f"  Dice: {dice:.4f} | "
            f"Precision: {prec:.4f} | "
            f"Recall: {rec:.4f}"
        )

        # 写入报告
        with open(REPORT_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                DATA_DIR.name,
                ct_path.stem,
                model_name,
                gt_mask.sum(),
                pred_mask.sum(),
                dice,
                prec,
                rec,
                len(bboxes),
                end_infer - start_infer,
                time.time() - start_case
            ])

    print(f"Total case time: {time.time() - start_case:.2f} sec")


print(f"\n All results saved to: {REPORT_FILE}")
