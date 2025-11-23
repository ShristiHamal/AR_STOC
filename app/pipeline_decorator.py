# app/pipeline_decorator.py

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

from clearml import Task, Dataset
from clearml import PipelineDecorator

# your actual inpaint model
from app.inpaint_inference import run_inpaint_tryon


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
    sample_ratio: float = 1.0,
    output_dir: str = "./pipeline_inpaint_outputs",
):
    task = Task.current_task()
    logger = task.get_logger()
    logger.report_text("Starting AR Inpaint Try-On Pipeline...")

    csv_path, dataset_root = Preprocessing(
        CLEARML_DATASET_ID,
        train_pairs_relpath,
        sample_ratio,
    )

    output_dir = TryOnInference(
        csv_path=csv_path,
        dataset_root=dataset_root,
        output_dir=output_dir,
    )

    metrics = Evaluation(output_dir)

    logger.report_text("Pipeline completed successfully.")
    return metrics


# ----------------------------------------------------
# Preprocessing
# ----------------------------------------------------
@PipelineDecorator.component(
    name="Preprocessing",
    task_type=Task.TaskTypes.data_processing,
    return_values=["csv_path", "dataset_root"],
)
def Preprocessing(CLEARML_DATASET_ID: str, train_pairs_relpath: str, sample_ratio: float):

    task = Task.current_task()
    logger = task.get_logger()

    logger.report_text(f"Fetching dataset {CLEARML_DATASET_ID}")
    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    dataset_root = Path(dataset.get_local_copy())

    train_pairs_path = dataset_root / train_pairs_relpath
    if not train_pairs_path.exists():
        raise FileNotFoundError(f"train_pairs.txt not found: {train_pairs_path}")

    df = pd.read_csv(train_pairs_path, sep=r"\s+", header=None, names=["person_id", "cloth_id"])

    if 0 < sample_ratio < 1:
        df = df.sample(int(len(df) * sample_ratio), random_state=42)
        logger.report_text(f"Sampled to {len(df)} rows")

    resolved_csv = dataset_root / "train_pairs_resolved.csv"
    df.to_csv(resolved_csv, index=False)

    return str(resolved_csv), str(dataset_root)


# ----------------------------------------------------
# Try-On Inference
# ----------------------------------------------------
@PipelineDecorator.component(
    name="TryOnInference_Inpaint",
    task_type=Task.TaskTypes.inference,
    return_values=["output_dir"],
)
def TryOnInference(csv_path: str, dataset_root: str, output_dir: str):

    task = Task.current_task()
    logger = task.get_logger()

    df = pd.read_csv(csv_path)
    dataset_root = Path(dataset_root)

    image_dir = dataset_root / "image"
    cloth_dir = dataset_root / "cloth"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.report_text(f"Running inference for {len(df)} pairs...")

    for idx, row in df.iterrows():

        pid = row["person_id"]
        cid = row["cloth_id"]

        person_path = image_dir / pid
        cloth_path = cloth_dir / cid

        if not person_path.exists():
            logger.report_text(f"[WARN] Missing person: {person_path}")
            continue
        if not cloth_path.exists():
            logger.report_text(f"[WARN] Missing cloth: {cloth_path}")
            continue

        # Load
        person_img = Image.open(person_path).convert("RGB")
        cloth_img = Image.open(cloth_path).convert("RGB")

        try:
            result = run_inpaint_tryon(
                person_image=person_img,
                cloth_image=cloth_img,
            )

            out_path = output_path / f"tryon_{idx:05d}.png"
            result.save(out_path)

            logger.report_image("TryOn Results", f"tryon-{idx}", str(out_path))

        except Exception as e:
            logger.report_text(f"[ERROR] Failed {pid}+{cid}: {e}")

    return str(output_path)


# ----------------------------------------------------
# Evaluation
# ----------------------------------------------------
@PipelineDecorator.component(
    name="Evaluation",
    task_type=Task.TaskTypes.testing,
    return_values=["metrics"],
)
def Evaluation(output_dir: str):

    task = Task.current_task()
    logger = task.get_logger()

    imgs = list(Path(output_dir).glob("*.png"))
    if len(imgs) == 0:
        logger.report_text("No images to evaluate.")
        return {"metrics": []}

    stats = []
    for path in imgs:
        img = np.array(Image.open(path))
        stats.append({
            "image": path.name,
            "mean": float(img.mean()),
            "std": float(img.std()),
        })

    df_stats = pd.DataFrame(stats)
    logger.report_table("Evaluation Summary", "pixel_stats", df_stats)

    return {"metrics": stats}
