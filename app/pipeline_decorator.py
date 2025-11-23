# app/pipeline_decorator.py

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch

from clearml import Task, Dataset
from clearml import PipelineDecorator

# Robust import so it works both locally and under ClearML agent
try:
    from .inpaint_inference import run_inpaint_tryon
except ImportError:
    from inpaint_inference import run_inpaint_tryon


# ----------------------------------------------------
# PIPELINE ROOT
# ----------------------------------------------------
@PipelineDecorator.pipeline(
    name="AR_TryOn_Inpaint_Batch",
    project="AR_STOC",
    version="1.0.0",
    default_queue="ar_stoc",
    pipeline_execution_queue="ar_stoc",
)
def full_pipeline(
    CLEARML_DATASET_ID: str = "936ce7ce676a41eca85cecfc59f1d6db",
    train_pairs_relpath: str = "train_pairs.txt",
    sample_ratio: float = 0.02,  # 2% by default; use 1.0 for full
    output_dir: str = "./pipeline_inpaint_outputs",
):
    task = Task.current_task()
    logger = task.get_logger()
    logger.report_text("Starting AR Inpaint Try-On Pipeline")

    csv_path, dataset_root = Preprocessing(
        CLEARML_DATASET_ID=CLEARML_DATASET_ID,
        train_pairs_relpath=train_pairs_relpath,
        sample_ratio=sample_ratio,
    )

    output_dir = TryOnInference(
        csv_path=csv_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )

    metrics = Evaluation(output_dir)

    device_name = (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    )
    logger.report_text(f"Pipeline completed on device: {device_name}")

    return metrics


# ----------------------------------------------------
# Preprocessing
# ----------------------------------------------------
@PipelineDecorator.component(
    name="Preprocessing",
    task_type=Task.TaskTypes.data_processing,
    return_values=["csv_path", "dataset_root"],
    cache=False,
)
def Preprocessing(
    CLEARML_DATASET_ID: str,
    train_pairs_relpath: str,
    sample_ratio: float,
):
    task = Task.current_task()
    logger = task.get_logger()

    logger.report_text(f"Fetching dataset: {CLEARML_DATASET_ID}")
    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    dataset_root = Path(dataset.get_local_copy())

    train_pairs_path = dataset_root / train_pairs_relpath
    if not train_pairs_path.exists():
        raise FileNotFoundError(f"train_pairs.txt not found at: {train_pairs_path}")

    # train_pairs.txt: "00001_00.jpg 00001_01.jpg"
    df = pd.read_csv(
        train_pairs_path,
        sep=r"\s+",
        header=None,
        names=["person_id", "cloth_id"],
    )

    total = len(df)
    logger.report_text(f"Found {total} pairs in train_pairs.txt")

    if 0 < sample_ratio < 1.0:
        sample_size = max(1, int(total * sample_ratio))
        df = df.sample(sample_size, random_state=42)
        logger.report_text(f"Sampling {sample_size} pairs ({sample_ratio * 100:.1f}%)")

    resolved_csv = dataset_root / "train_pairs_resolved.csv"
    df.to_csv(resolved_csv, index=False)

    logger.report_table(
        title="Sampled Train Pairs",
        series="pairs_preview",
        table_plot=df.head(10),
    )
    logger.report_text(f"Preprocessing done. Saved CSV to: {resolved_csv}")

    return str(resolved_csv), str(dataset_root)


# ----------------------------------------------------
# Try-On Inference
# ----------------------------------------------------
@PipelineDecorator.component(
    name="TryOnInference_Inpaint",
    task_type=Task.TaskTypes.inference,
    return_values=["output_dir"],
    cache=False,
)
def TryOnInference(
    csv_path: str,
    dataset_root: str,
    output_dir: str,
):
    task = Task.current_task()
    logger = task.get_logger()

    df = pd.read_csv(csv_path)
    dataset_root = Path(dataset_root)

    image_dir = dataset_root / "image"
    cloth_dir = dataset_root / "cloth"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.report_text(f"Starting inpaint try-on for {len(df)} pairs")

    for idx, row in df.iterrows():
        pid = str(row["person_id"]).strip()
        cid = str(row["cloth_id"]).strip()

        person_path = image_dir / pid
        cloth_path = cloth_dir / cid

        if not person_path.exists():
            logger.report_text(f"[WARN] Missing person image: {person_path}")
            continue
        if not cloth_path.exists():
            logger.report_text(f"[WARN] Missing cloth image: {cloth_path}")
            continue

        try:
            person_img = Image.open(person_path).convert("RGB")
            cloth_img = Image.open(cloth_path).convert("RGB")

            # Call inpaint try-on (no mask/openpose needed)
            result = run_inpaint_tryon(
                person_image=person_img,
                cloth_image=cloth_img,
                num_inference_steps=25,
                guidance_scale=7.5,
                prompt="a realistic photo of the person wearing the clothing item",
            )

            out_path = output_path / f"tryon_{idx:05d}.png"
            result.save(out_path)

            logger.report_image(
                title=f"TryOn_{idx:05d}",
                series="inpaint_results",
                local_path=str(out_path),
            )

        except Exception as e:
            logger.report_text(f"[ERROR] Failed for {pid} + {cid}: {e}")

    logger.report_text(f"Try-on images saved to: {output_path}")
    return str(output_path)


# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------
@PipelineDecorator.component(
    name="Evaluation",
    task_type=Task.TaskTypes.testing,
    return_values=["metrics"],
    cache=False,
)
def Evaluation(output_dir: str):
    task = Task.current_task()
    logger = task.get_logger()

    output_dir = Path(output_dir)
    images = list(output_dir.glob("*.png"))

    if not images:
        logger.report_text("No output images to evaluate.")
        return {"metrics": []}

    metrics = []
    for path in images:
        img = np.array(Image.open(path).convert("RGB"))
        metrics.append(
            {
                "image": path.name,
                "mean_pixel": float(img.mean()),
                "std_pixel": float(img.std()),
            }
        )

    df_metrics = pd.DataFrame(metrics)

    logger.report_table(
        title="Evaluation Summary",
        series="pixel_stats",
        table_plot=df_metrics.head(20),
    )

    logger.report_text(f"Evaluated {len(metrics)} images.")
    return {"metrics": metrics}


if __name__ == "__main__":
    # Local test run (this will also register a pipeline in ClearML)
    full_pipeline(
        CLEARML_DATASET_ID="936ce7ce676a41eca85cecfc59f1d6db",
        train_pairs_relpath="train_pairs.txt",
        sample_ratio=0.02,
        output_dir="./pipeline_inpaint_outputs",
    )
