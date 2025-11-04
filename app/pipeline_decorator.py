# app/pipeline_decorator.py

from clearml import PipelineDecorator, Dataset, Logger
from pathlib import Path
import pandas as pd
import torch
import os
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import OpenposeDetector
from PIL import Image
import numpy as np


# -------------------- 1. PREPROCESSING -------------------- #
@PipelineDecorator.component(return_values=["df_path"])
def Preprocessing(dataset_name: str, dataset_project: str):
    """
    Step 1: Load ClearML dataset and parse train_pairs.txt
    (each line contains 'person_image cloth_image')
    """
    print(f"🔹 Loading dataset '{dataset_name}' from ClearML project '{dataset_project}'...")
    dataset = Dataset.get(dataset_name=dataset_name, dataset_project=dataset_project)
    dataset_path = Path(dataset.get_local_copy())

    train_pairs_path = dataset_path / "train_pairs.txt"
    if not train_pairs_path.exists():
        raise FileNotFoundError(f" train_pairs.txt not found at: {train_pairs_path}")

    try:
        df = pd.read_csv(train_pairs_path, sep=r"\s+", engine="python", header=None)
    except Exception as e:
        raise RuntimeError(f"Error reading train_pairs.txt: {e}")

    # Each row: person image | cloth image
    df.columns = ["person_image", "cloth_image"]

    # Resolve paths (relative to dataset root)
    df["person_image"] = df["person_image"].apply(lambda x: str(dataset_path / "train" / x))
    df["cloth_image"] = df["cloth_image"].apply(lambda x: str(dataset_path / "train" / x))

    csv_path = dataset_path / "train_pairs_resolved.csv"
    df.to_csv(csv_path, index=False)

    Logger.current_logger().report_table(
        title="Sample Training Pairs",
        series="train_pairs_preview",
        table_plot=df.head(10)
    )
    print(f"Preprocessing complete: {len(df)} pairs saved to {csv_path}")
    return str(csv_path)


# -------------------- 2. CONTROLNET INFERENCE -------------------- #
@PipelineDecorator.component(return_values=["processed_images_dir"])
def ControlNetInference(df_path: str, output_dir: str):
    """
    Step 2: Generate try-on results using ControlNet (OpenPose-based conditioning)
    """
    df = pd.read_csv(df_path)
    os.makedirs(output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    logger = Logger.current_logger()

    print(" Loading ControlNet + Stable Diffusion...")
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-openpose",
        torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype
    ).to(device)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")

    print(f" Running inference for {len(df)} pairs...")
    for idx, row in df.iterrows():
        try:
            person_path = row["person_image"]
            cloth_path = row["cloth_image"]

            if not os.path.exists(person_path) or not os.path.exists(cloth_path):
                logger.report_text(f" Missing file(s) for pair {idx}: {person_path}, {cloth_path}")
                continue

            person_img = Image.open(person_path).convert("RGB")
            pose = openpose(person_img)
            cloth_name = os.path.basename(cloth_path)

            # Generate try-on image
            result = pipe(
                prompt=f"A person wearing {cloth_name}",
                image=pose,
                num_inference_steps=20,
                guidance_scale=9.0
            ).images[0]

            out_path = os.path.join(output_dir, f"tryon_{idx:05d}.png")
            result.save(out_path)
            logger.report_image(title="Try-On Result", series="generated", local_path=out_path)
        except Exception as e:
            logger.report_text(f" Error in inference for pair {idx}: {e}")

    logger.report_text(f"Inference complete. Results in: {output_dir}")
    return output_dir


# -------------------- 3. TRAINING COMPONENT -------------------- #
@PipelineDecorator.component(return_values=["model_path"])
def Training(df_path: str):
    """
    Step 3: Dummy training placeholder — replace with actual training code if needed.
    """
    df = pd.read_csv(df_path)
    print(f"Training with {len(df)} samples (simulated)...")

    model_path = "./trained_model.pt"
    with open(model_path, "w") as f:
        f.write("fake model weights for demo")

    Logger.current_logger().report_text("Training completed successfully.")
    return model_path


# -------------------- 4. EVALUATION -------------------- #
@PipelineDecorator.component(return_values=["metrics"])
def Evaluation(processed_images_dir: str):
    """
    Step 4: Evaluate generated results using simple image stats.
    """
    image_files = list(Path(processed_images_dir).glob("*.png"))
    if not image_files:
        Logger.current_logger().report_text("⚠️ No images found for evaluation.")
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
    Logger.current_logger().report_table(
        title="Evaluation Summary",
        series="metrics_overview",
        table_plot=df_eval.head(10)
    )
    Logger.current_logger().report_text(f"✅ Evaluated {len(results)} generated images.")
    return {"metrics": results}


# -------------------- 5. FULL PIPELINE -------------------- #
@PipelineDecorator.pipeline(
    name="AR Smart Try-On Pipeline",
    project="AR Smart Try-On",
    version="1.0",
    default_queue="ar_stoc",
    pipeline_execution_queue="ar_stoc"
)
def main_pipeline():
    df_path = Preprocessing(dataset_name="AR_TryOn_Train", dataset_project="AR Smart Try-On")
    processed_images_dir = ControlNetInference(df_path, output_dir="./results")
    model_path = Training(df_path)
    metrics = Evaluation(processed_images_dir)

    print(" Pipeline completed successfully.")
    return metrics


if __name__ == "__main__":
    PipelineDecorator.run_pipeline()
