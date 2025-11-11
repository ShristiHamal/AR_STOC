# app/pipeline_decorator.py
from clearml import PipelineDecorator, Task, Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from controlnet_inference import run_controlnet_inference  # using your refined function

# -------------------- Pipeline --------------------
@PipelineDecorator.pipeline(
    name="AR_TryOn_Full",
    project="AR_STOC",
    version="1.0.0",
    default_queue="ar_stoc",
    pipeline_execution_queue="ar_stoc"
)
def full_pipeline(
    output_dir: str,
    CLEARML_DATASET_ID: str = "936ce7ce676a41eca85cecfc59f1d6db"
):
    task = Task.current_task()
    logger = task.get_logger()
    logger.report_text("Starting full AR Try-On pipeline...")

    # Preprocessing
    df_path = Preprocessing(CLEARML_DATASET_ID)

    #  ControlNet Inference
    processed_images_dir = ControlNetInference(df_path, output_dir)

    # Training (optional dummy)
    model_path = Training(df_path)

    #  Evaluation
    metrics = Evaluation(processed_images_dir)

    logger.report_text("Pipeline completed successfully.")
    return metrics


# -------------------- 1. PREPROCESSING --------------------
@PipelineDecorator.component(
    name="Preprocessing",
    return_values=["df_path"],
    cache=True,
    task_type=Task.TaskTypes.data_processing
)
def Preprocessing(CLEARML_DATASET_ID: str):
    task = Task.current_task()
    logger = task.get_logger()

    logger.report_text(f"Loading dataset ID '{CLEARML_DATASET_ID}' from ClearML...")
    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    dataset_path = Path(dataset.get_local_copy())

    train_pairs_path = dataset_path / "train_pairs.txt"
    if not train_pairs_path.exists():
        raise FileNotFoundError(f"train_pairs.txt not found at: {train_pairs_path}")

    df = pd.read_csv(train_pairs_path, sep=r"\s+", engine="python", header=None)
    df.columns = ["person_image", "cloth_image"]

    # Select 2% of dataset for faster testing
    sample_size = max(1, int(len(df) * 0.02))
    df = df.sample(n=sample_size, random_state=42)

    df["person_image"] = df["person_image"].apply(lambda x: str(dataset_path / "train" / x))
    df["cloth_image"] = df["cloth_image"].apply(lambda x: str(dataset_path / "train" / x))

    csv_path = dataset_path / "train_pairs_resolved.csv"
    df.to_csv(csv_path, index=False)

    logger.report_table(title="Sample Training Pairs", series="train_pairs_preview", table_plot=df.head(10))
    logger.report_text(f"Preprocessing complete: {len(df)} pairs saved to {csv_path}")
    return str(csv_path)


# -------------------- 2. CONTROLNET INFERENCE --------------------
@PipelineDecorator.component(
    name="ControlNetInference",
    return_values=["processed_images_dir"],
    cache=False,
    task_type=Task.TaskTypes.inference
)
def ControlNetInference(df_path: str, output_dir: str):
    task = Task.current_task()
    logger = task.get_logger()

    df = pd.read_csv(df_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.report_text(f"Running ControlNet inference on {len(df)} pairs...")
    for idx, row in df.iterrows():
        try:
            person_path = row["person_image"]
            cloth_path = row["cloth_image"]

            out_im = run_controlnet_inference(person_path, cloth_path, prompt=f"A person wearing {os.path.basename(cloth_path)}")
            out_path = output_path / f"tryon_{idx:05d}.png"
            out_im.save(out_path)
            logger.report_image(title="Try-On Result", series="generated", local_path=str(out_path))

        except Exception as e:
            logger.report_text(f"Error in inference for pair {idx}: {e}")

    logger.report_text(f"Inference complete. Results saved to: {output_path}")
    return str(output_path)


# -------------------- 3. TRAINING (optional placeholder) --------------------
@PipelineDecorator.component(
    name="Training",
    return_values=["model_path"],
    cache=False,
    task_type=Task.TaskTypes.training
)
def Training(df_path: str):
    task = Task.current_task()
    logger = task.get_logger()
    df = pd.read_csv(df_path)
    logger.report_text(f"Simulated training with {len(df)} samples")
    model_path = Path("./trained_model.pt")
    with open(model_path, "w") as f:
        f.write("fake model weights")
    return str(model_path)


# -------------------- 4. EVALUATION --------------------
@PipelineDecorator.component(
    name="Evaluation",
    return_values=["metrics"],
    cache=False,
    task_type=Task.TaskTypes.testing
)
def Evaluation(processed_images_dir: str):
    task = Task.current_task()
    logger = task.get_logger()

    image_files = list(Path(processed_images_dir).glob("*.png"))
    if not image_files:
        logger.report_text("No images found for evaluation.")
        return {"metrics": []}

    results = []
    for img_path in image_files:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        results.append({"image": img_path.name, "mean_pixel": float(arr.mean()), "std_pixel": float(arr.std())})

    df_eval = pd.DataFrame(results)
    logger.report_table(title="Evaluation Summary", series="metrics_overview", table_plot=df_eval.head(10))
    logger.report_text(f"Evaluated {len(results)} generated images.")
    return {"metrics": results}


# -------------------- ENTRY POINT --------------------
if __name__ == "__main__":
    pipeline_instance = full_pipeline(output_dir="./controlnet_outputs")
    pipeline_instance.start(repo="https://github.com/ShristiHamal/AR_STOC.git", branch="main")
