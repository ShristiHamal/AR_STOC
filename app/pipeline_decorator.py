# app/pipeline_decorator.py

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import numpy as np

from clearml import Task, Dataset
from clearml import PipelineDecorator

# Import your real virtual try-on function
from app.controlnet_inference import run_controlnet_inference


# define full pipeline

@PipelineDecorator.pipeline(
    name="AR_TryOn_VirtualTryOn_Full",
    project="AR_STOC",
    version="1.0.0",
    default_queue="ar_stoc",
    pipeline_execution_queue="ar_stoc",
)
def full_pipeline(
    CLEARML_DATASET_ID: str = "936ce7ce676a41eca85cecfc59f1d6db",
    output_dir: str = "./pipeline_tryon_outputs",
):
    task = Task.current_task()
    logger = task.get_logger()
    logger.report_text("Starting Virtual Try-On Pipeline...")

    # Step 1 — Preprocessing
    csv_path = Preprocessing(CLEARML_DATASET_ID)

    # Step 2 — Try-On Inference
    outputs_dir = TryOnInference(csv_path, output_dir)

    # Step 3 — Evaluation
    metrics = Evaluation(outputs_dir)

    logger.report_text("Pipeline Finished Successfully.")
    return metrics


#preprocessing component

@PipelineDecorator.component(
    name="Preprocessing",
    task_type=Task.TaskTypes.data_processing,
    return_values=["csv_out"],
    cache=False,
)
def Preprocessing(CLEARML_DATASET_ID: str):
    task = Task.current_task()
    logger = task.get_logger()

    logger.report_text(f"Fetching dataset {CLEARML_DATASET_ID} from ClearML...")

    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    dataset_path = Path(dataset.get_local_copy())

    train_pairs = dataset_path / "train" / "train_pairs.csv"
    if not train_pairs.exists():
        raise FileNotFoundError(f"CSV not found: {train_pairs}")

    df = pd.read_csv(train_pairs)

    # Only take a small subset for testing (2%)
    sample_size = max(1, int(len(df) * 0.02))
    df = df.sample(n=sample_size, random_state=42)

    resolved_csv = dataset_path / "train" / "train_pairs_resolved.csv"
    df.to_csv(resolved_csv, index=False)

    logger.report_table(
        title="Sampled Train Pairs",
        series="pairs_preview",
        table_plot=df.head(10),
    )
    logger.report_text(f"Preprocessing complete. {sample_size} items saved to CSV.")

    return str(resolved_csv)


#try-on image generation
@PipelineDecorator.component(
    name="TryOnInference",
    task_type=Task.TaskTypes.inference,
    return_values=["output_dir"],
    cache=False,
)
def TryOnInference(csv_path: str, output_dir: str):
    task = Task.current_task()
    logger = task.get_logger()

    df = pd.read_csv(csv_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.report_text(f"Generating try-on images for {len(df)} samples...")

    for idx, row in df.iterrows():

        person = row["person_image"]
        cloth = row["cloth_image"]
        mask = row["mask_image"]
        pose = row["pose_json"]

        try:
            result = run_controlnet_inference(
                person_img_path=person,
                cloth_img_path=cloth,
                mask_img_path=mask,
                pose_json_path=pose,
                num_inference_steps=20,
                strength=0.65,
            )

            out_path = output_path / f"tryon_{idx:05d}.png"
            result.save(out_path)

            logger.report_image(
                title=f"Try-On Result {idx}",
                series="generated",
                local_path=str(out_path)
            )

        except Exception as e:
            logger.report_text(f"[ERROR] Failed for pair {idx}: {e}")

    logger.report_text(f"Try-on generation completed. Output saved to: {output_path}")
    return str(output_path)


#evaluation component

@PipelineDecorator.component(
    name="Evaluation",
    task_type=Task.TaskTypes.testing,
    return_values=["metrics"],
    cache=False,
)
def Evaluation(output_dir: str):
    task = Task.current_task()
    logger = task.get_logger()

    image_files = list(Path(output_dir).glob("*.png"))

    if len(image_files) == 0:
        logger.report_text("No output images to evaluate.")
        return {"metrics": []}

    metrics = []
    for path in image_files:
        img = np.array(Image.open(path).convert("RGB"))
        metrics.append({
            "image": path.name,
            "mean_pixel": float(img.mean()),
            "std_pixel": float(img.std()),
        })

    df_eval = pd.DataFrame(metrics)

    logger.report_table(
        title="Evaluation Summary",
        series="eval_overview",
        table_plot=df_eval.head(10),
    )

    logger.report_text(f"Evaluated {len(metrics)} try-on images.")

    return {"metrics": metrics}


if __name__ == "__main__":
    full_pipeline(output_dir="./pipeline_tryon_outputs")
