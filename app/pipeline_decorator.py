# pipeline_decorator.py

from clearml import PipelineDecorator, Task, Dataset
from pathlib import Path
from PIL import Image
import pandas as pd
import numpy as np
import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector

# ----------------------------------------------------
# Full Pipeline Definition
# ----------------------------------------------------
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
    """
    Full AR Try-On pipeline:
    Preprocessing → ControlNet Inference → Training → Evaluation
    """
    task = Task.current_task()
    logger = task.get_logger()
    logger.report_text("Starting full AR Try-On pipeline...")

    # Step 1: Preprocessing
    df_path = Preprocessing(CLEARML_DATASET_ID)

    # Step 2: ControlNet Inference
    processed_images_dir = ControlNetInference(df_path, output_dir)

    # Step 3: Training (dummy placeholder)
    model_path = Training(df_path)

    # Step 4: Evaluation
    metrics = Evaluation(processed_images_dir)

    logger.report_text("Pipeline completed successfully.")
    return metrics


# -------------------- 1. PREPROCESSING -------------------- #
@PipelineDecorator.component(
    name="Preprocessing",
    return_values=["df_path"],
    cache=True,
    task_type=Task.TaskTypes.data_processing
)
def Preprocessing(CLEARML_DATASET_ID: str = "83697f8f15c2423fb3b961bfc130dac1"):
    """
    Load ClearML dataset using dataset ID and parse train_pairs.txt
    """
    task = Task.current_task()
    logger = task.get_logger()

    logger.report_text(f"Loading dataset ID '{CLEARML_DATASET_ID}' from ClearML...")
    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    dataset_path = Path(dataset.get_local_copy())

    train_pairs_path = dataset_path / "train_pairs.txt"
    if not train_pairs_path.exists():
        raise FileNotFoundError(f"train_pairs.txt not found at: {train_pairs_path}")

    try:
        df = pd.read_csv(train_pairs_path, sep=r"\s+", engine="python", header=None)
    except Exception as e:
        raise RuntimeError(f"Error reading train_pairs.txt: {e}")

    df.columns = ["person_image", "cloth_image"]

    # Resolve paths relative to dataset root
    df["person_image"] = df["person_image"].apply(lambda x: str(dataset_path / "train" / x))
    df["cloth_image"] = df["cloth_image"].apply(lambda x: str(dataset_path / "train" / x))

    csv_path = dataset_path / "train_pairs_resolved.csv"
    df.to_csv(csv_path, index=False)

    logger.report_table(
        title="Sample Training Pairs",
        series="train_pairs_preview",
        table_plot=df.head(10)
    )
    logger.report_text(f"Preprocessing complete: {len(df)} pairs saved to {csv_path}")
    return str(csv_path)


# -------------------- 2. CONTROLNET INFERENCE -------------------- #
@PipelineDecorator.component(
    name="ControlNetInference",
    return_values=["processed_images_dir"],
    cache=False,
    task_type=Task.TaskTypes.inference
)
def ControlNetInference(df_path: str, output_dir: str):
    """
    Generate try-on results using ControlNet (OpenPose-based conditioning)
    """
    task = Task.current_task()
    logger = task.get_logger()

    df = pd.read_csv(df_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.report_text("Loading ControlNet and Stable Diffusion models...")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-openpose", torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    logger.report_text(f"Running inference for {len(df)} pairs...")
    for idx, row in df.iterrows():
        try:
            person_path = row["person_image"]
            cloth_path = row["cloth_image"]

            if not os.path.exists(person_path) or not os.path.exists(cloth_path):
                logger.report_text(f"Missing file(s) for pair {idx}: {person_path}, {cloth_path}")
                continue

            person_img = Image.open(person_path).convert("RGB")
            pose_img = openpose(person_img)
            cloth_name = os.path.basename(cloth_path)

            # Generate try-on image
            with torch.autocast(device_type=device, dtype=dtype):
                result = pipe(
                    prompt=f"A person wearing {cloth_name}",
                    image=pose_img,
                    num_inference_steps=20,
                    guidance_scale=9.0
                ).images[0]

            out_path = output_path / f"tryon_{idx:05d}.png"
            result.save(out_path)
            logger.report_image(title="Try-On Result", series="generated", local_path=str(out_path))

        except Exception as e:
            logger.report_text(f"Error in inference for pair {idx}: {e}")

    logger.report_text(f"Inference complete. Results saved to: {output_path}")
    return str(output_path)


# -------------------- 3. TRAINING COMPONENT -------------------- #
@PipelineDecorator.component(
    name="Training",
    return_values=["model_path"],
    cache=False,
    task_type=Task.TaskTypes.training
)
def Training(df_path: str):
    """
    Dummy training placeholder
    """
    task = Task.current_task()
    logger = task.get_logger()

    df = pd.read_csv(df_path)
    logger.report_text(f"Training with {len(df)} samples (simulated)...")

    model_path = Path("./trained_model.pt")
    with open(model_path, "w") as f:
        f.write("fake model weights for demo")

    logger.report_text("Training completed successfully.")
    return str(model_path)


# -------------------- 4. EVALUATION -------------------- #
@PipelineDecorator.component(
    name="Evaluation",
    return_values=["metrics"],
    cache=False,
    task_type=Task.TaskTypes.testing
)
def Evaluation(processed_images_dir: str):
    """
    Evaluate generated results using simple image statistics
    """
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
        results.append({
            "image": img_path.name,
            "mean_pixel": float(arr.mean()),
            "std_pixel": float(arr.std())
        })

    df_eval = pd.DataFrame(results)
    logger.report_table(
        title="Evaluation Summary",
        series="metrics_overview",
        table_plot=df_eval.head(10)
    )
    logger.report_text(f"Evaluated {len(results)} generated images.")
    return {"metrics": results}


# -------------------- Pipeline Entry Point -------------------- #
if __name__ == "__main__":
    pipeline_instance = full_pipeline(
        output_dir=r"C:/Users/shris/OneDrive - UTS/Documents/GitHub/AR_STOC/controlnet_outputs"
    )

    pipeline_instance.start(
        repo="https://github.com/ShristiHamal/AR_STOC.git",
        branch="main"
    )
