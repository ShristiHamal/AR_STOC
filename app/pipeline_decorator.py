# app/pipeline_decorator.py

import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image

from clearml import Task, Dataset, PipelineDecorator

from inpaint_inference import run_inpaint_tryon


# ---------------------------------------------------------
# FULL PIPELINE CONTROLLER
# ---------------------------------------------------------
@PipelineDecorator.pipeline(
    name="AR_Inpaint_TryOn_Full",
    project="AR_STOC",
    version="1.0.0",
    default_queue="ar_stoc",
    pipeline_execution_queue="ar_stoc",
)
def full_pipeline(
    CLEARML_DATASET_ID="936ce7ce676a41eca85cecfc59f1d6db",
    train_pairs_relpath="train_pairs.txt",
    sample_ratio=1.0,
    output_dir="./inpaint_pipeline_outputs",
):
    task = Task.current_task()
    logger = task.get_logger()

    logger.report_text("Starting INPAINT try-on pipeline...")

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

    metrics = Evaluation(output_dir=output_dir)

    logger.report_text("Pipeline completed!")
    return metrics


# ---------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------
@PipelineDecorator.component(
    name="Preprocessing",
    return_values=["csv_path", "dataset_root"],
)
def Preprocessing(CLEARML_DATASET_ID, train_pairs_relpath, sample_ratio):

    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    dataset_root = Path(dataset.get_local_copy())

    train_pairs = dataset_root / train_pairs_relpath
    if not train_pairs.exists():
        raise FileNotFoundError(f"Missing: {train_pairs}")

    df = pd.read_csv(train_pairs, sep=r"\s+", header=None, names=["person_id", "cloth_id"])

    if 0 < sample_ratio < 1:
        df = df.sample(int(len(df) * sample_ratio), random_state=42)

    out_csv = dataset_root / "pairs_resolved.csv"
    df.to_csv(out_csv, index=False)

    return str(out_csv), str(dataset_root)


# ---------------------------------------------------------
# TRY-ON INFERENCE (INPAINT)
# ---------------------------------------------------------
@PipelineDecorator.component(
    name="Inpaint_TryOn",
    return_values=["output_dir"],
)
def TryOnInference(csv_path, dataset_root, output_dir):

    df = pd.read_csv(csv_path)
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir   = dataset_root / "image"
    cloth_dir   = dataset_root / "cloth"
    mask_dir    = dataset_root / "cloth-mask"

    for idx, row in df.iterrows():
        pid = row["person_id"]
        cid = row["cloth_id"]

        person_path = image_dir / pid
        cloth_path  = cloth_dir / cid
        mask_path   = mask_dir  / cid.replace(".jpg", ".png")

        if not person_path.exists() or not cloth_path.exists() or not mask_path.exists():
            continue

        person = Image.open(person_path)
        cloth  = Image.open(cloth_path)
        mask   = Image.open(mask_path)

        try:
            result = run_inpaint_tryon(
                person_image=person,
                cloth_image=cloth,
                mask_image=mask,
                pose_json=None,
            )

            result.save(output_dir / f"tryon_{idx:05d}.png")

        except Exception as e:
            print(f"ERROR @ {pid}+{cid}: {e}")

    return str(output_dir)


# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------
@PipelineDecorator.component(
    name="Evaluation",
    return_values=["metrics"],
)
def Evaluation(output_dir):

    imgs = list(Path(output_dir).glob("*.png"))
    if len(imgs) == 0:
        return {"metrics": []}

    stats = []
    for path in imgs:
        arr = np.array(Image.open(path))
        stats.append({
            "image": path.name,
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        })

    return {"metrics": stats}
