# app/pipeline_decorator.py

from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

import torch
from clearml import Task, Dataset
from clearml import PipelineDecorator

from app.inpaint_inference import run_inpaint_tryon


# ====================================================
#  PIPELINE ROOT
# ====================================================
@PipelineDecorator.pipeline(
    name="AR_TryOn_Inpaint_Batch_V3",
    project="AR_STOC",
    version="1.0.0",
    default_queue="ar_stoc",
    pipeline_execution_queue="ar_stoc",
)
def full_pipeline(
    CLEARML_DATASET_ID: str = "936ce7ce676a41eca85cecfc59f1d6db",
    train_pairs_relpath: str = "train_pairs.txt",
    sample_ratio: float = 0.02,          # 2% for quick experiments
    output_dir: str = "./pipeline_inpaint_outputs",
):
    task = Task.current_task()
    logger = task.get_logger()
    logger.report_text("🚀 Starting AR_TryOn_Inpaint_Batch_V3 pipeline...")

    csv_path, dataset_root = Preprocess_InpaintV3(
        CLEARML_DATASET_ID=CLEARML_DATASET_ID,
        train_pairs_relpath=train_pairs_relpath,
        sample_ratio=sample_ratio,
    )

    out_dir = TryOn_InpaintV3(
        csv_path=csv_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )

    metrics = Eval_InpaintV3(output_dir=out_dir)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.report_text(f"✅ Pipeline done. Device: {device_name}")
    return metrics


# ====================================================
#  PREPROCESSING COMPONENT
# ====================================================
@PipelineDecorator.component(
    name="Preprocess_InpaintV3",
    task_type=Task.TaskTypes.data_processing,
    return_values=["csv_path", "dataset_root"],
)
def Preprocess_InpaintV3(
    CLEARML_DATASET_ID: str,
    train_pairs_relpath: str,
    sample_ratio: float,
):
    task = Task.current_task()
    logger = task.get_logger()

    logger.report_text(f"📦 Fetching ClearML dataset: {CLEARML_DATASET_ID}")
    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    dataset_root = Path(dataset.get_local_copy())

    train_pairs_path = dataset_root / train_pairs_relpath
    if not train_pairs_path.exists():
        raise FileNotFoundError(f"train_pairs file not found at {train_pairs_path}")

    # VTON-HD style train_pairs: "00001_00.jpg 00001_02.jpg"
    df = pd.read_csv(
        train_pairs_path,
        delim_whitespace=True,
        header=None,
        names=["person_id", "cloth_id"],
    )

    total = len(df)
    logger.report_text(f"Found {total} pairs in train_pairs.txt")

    if 0 < sample_ratio < 1.0:
        sample_size = max(1, int(total * sample_ratio))
        df = df.sample(n=sample_size, random_state=42)
        logger.report_text(f"Sampling {sample_size} pairs ({sample_ratio*100:.1f}% of data)")
    else:
        logger.report_text("Using full dataset (no sampling).")

    resolved_csv = dataset_root / "train_pairs_resolved_inpaint_v3.csv"
    df.to_csv(resolved_csv, index=False)

    logger.report_table(
        title="Sampled Train Pairs",
        series="pairs_preview",
        table_plot=df.head(10),
    )

    logger.report_text(f"✅ Preprocessing done. Saved CSV to: {resolved_csv}")

    return str(resolved_csv), str(dataset_root)


# ====================================================
#  INFERENCE COMPONENT
# ====================================================
@PipelineDecorator.component(
    name="TryOn_InpaintV3",
    task_type=Task.TaskTypes.inference,
    return_values=["output_dir"],
)
def TryOn_InpaintV3(
    csv_path: str,
    dataset_root: str,
    output_dir: str,
):
    task = Task.current_task()
    logger = task.get_logger()

    df = pd.read_csv(csv_path)
    root = Path(dataset_root)

    image_dir = root / "image"
    cloth_dir = root / "cloth"
    mask_dir = root / "cloth-mask"
    openpose_dir = root / "openpose_json"   # optional, not used for now

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.report_text(
        f"🖼 Running inpaint try-on for {len(df)} pairs...\n"
        f"image_dir={image_dir}\ncloth_dir={cloth_dir}\nmask_dir={mask_dir}"
    )

    for idx, row in df.iterrows():
        pid = str(row["person_id"]).strip()
        cid = str(row["cloth_id"]).strip()

        person_path = image_dir / pid
        cloth_path = cloth_dir / cid
        mask_path = mask_dir / cid.replace(".jpg", ".png")  # VTON-HD layout

        if not person_path.exists():
            logger.report_text(f"[WARN] Missing person image: {person_path}")
            continue
        if not cloth_path.exists():
            logger.report_text(f"[WARN] Missing cloth image: {cloth_path}")
            continue
        if not mask_path.exists():
            logger.report_text(f"[WARN] Missing cloth mask: {mask_path}")
            continue

        person_img = Image.open(person_path).convert("RGB")
        cloth_img = Image.open(cloth_path).convert("RGB")
        mask_img = Image.open(mask_path).convert("L")

        try:
            result = run_inpaint_tryon(
                person_image=person_img,
                cloth_image=cloth_img,
                mask_image=mask_img,
                num_inference_steps=25,
                guidance_scale=7.5,
                prompt="a realistic studio photo of the person wearing the garment",
            )

            save_path = out_path / f"tryon_{idx:05d}.png"
            result.save(save_path)

            logger.report_image(
                title=f"Try-On {idx}",
                series="inpaint_results_v3",
                local_path=str(save_path),
            )
        except Exception as e:
            logger.report_text(f"[ERROR] Pair {idx} ({pid}, {cid}) failed: {e}")

    task.upload_artifact("inpaint_tryon_outputs_v3", artifact_object=str(out_path))

    logger.report_text(f"✅ Try-on generation complete. Outputs in: {out_path}")
    return str(out_path)


# ====================================================
#  EVALUATION COMPONENT
# ====================================================
@PipelineDecorator.component(
    name="Eval_InpaintV3",
    task_type=Task.TaskTypes.testing,
    return_values=["metrics"],
)
def Eval_InpaintV3(output_dir: str):
    task = Task.current_task()
    logger = task.get_logger()

    out_path = Path(output_dir)
    image_files = list(out_path.glob("*.png"))

    if not image_files:
        logger.report_text("⚠ No generated images to evaluate.")
        return {"metrics": []}

    stats = []
    for p in image_files:
        img = np.array(Image.open(p).convert("RGB"))
        stats.append(
            {
                "image": p.name,
                "mean": float(img.mean()),
                "std": float(img.std()),
            }
        )

    df_stats = pd.DataFrame(stats)
    logger.report_table(
        title="Inpaint V3 Pixel Stats",
        series="inpaint_v3_eval",
        table_plot=df_stats,
    )

    logger.report_text(f"✅ Evaluated {len(stats)} images.")
    return {"metrics": stats}
