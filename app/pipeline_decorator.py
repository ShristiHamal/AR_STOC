# app/pipeline_decorator.py
import os
import csv
from PIL import Image
import numpy as np

from pathlib import Path
from clearml import PipelineDecorator, Dataset, Task
from pathlib import Path
from .data_prep import make_csv
from .controlnet_inference import run_controlnet_inference
from .sd_inference import run_sd_inference

# -------------------- Config --------------------
# CLEARML_DATASET_ID = "8832df278eb245b2856da6c202aaa876"
CLEARML_DATASET_ID = "c1fca92f4cc1402fac5fd6026c1128e5"
PIPELINE_NAME = "AR_TryOn"
PROJECT_NAME = "AR_STOC"
PIPELINE_VERSION = "1.0.0"
QUEUE_NAME = "ar_stoc"

# -------------------- Pipeline --------------------
@PipelineDecorator.pipeline(
    name=PIPELINE_NAME,
    project=PROJECT_NAME,
    version=PIPELINE_VERSION,
    pipeline_execution_queue=QUEUE_NAME
)
def full_pipeline(output_dir: str):
    """
    Full pipeline: preprocessing -> ControlNet inference -> SD evaluation
    """
    # Fetch dataset from ClearML
    dataset = Dataset.get(dataset_id=CLEARML_DATASET_ID)
    root_dir = dataset.get_local_copy()

    # Preprocessing
    csv_path = preprocessing_component(root_dir)

    # ControlNet inference
    controlnet_output_dir = controlnet_component(csv_path, output_dir)

    # SD inpainting evaluation
    sd_metrics = sd_component(controlnet_output_dir, output_dir)

    return sd_metrics

# -------------------- Pipeline Components --------------------
@PipelineDecorator.component(return_values=["csv_path"])
def preprocessing_component(root_dir: str) -> str:
    """
    Preprocessing: generate CSV of valid pairs from dataset
    """
    root_path = Path(root_dir)
    csv_path = make_csv(root_path, split="train")  # You can adjust split if needed
    print(f"[Preprocessing] CSV saved at {csv_path}")
    return str(csv_path)

@PipelineDecorator.component(return_values=["controlnet_output_dir"])
def controlnet_component(csv_path: str, output_dir: str) -> str:
    """
    Run ControlNet inference for virtual try-on
    """
    output_path = Path(output_dir) / "controlnet_outputs"
    output_path.mkdir(parents=True, exist_ok=True)

    from controlnet_inference import run_controlnet_inference
    import csv

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    for i, row in enumerate(rows, start=1):
        try:
            result = run_controlnet_inference(
                row['person_image'],
                row['mask_image'],
                prompt="Virtual try-on"
            )
            out_file = output_path / f"tryon_{i}.png"
            result.save(out_file)
            print(f"[ControlNet] Saved {out_file}")
        except Exception as e:
            print(f"[ControlNet] Failed row {i}: {e}")

    return str(output_path)

@PipelineDecorator.component(return_values=["sd_metrics"])
def sd_component(controlnet_output_dir: str, output_dir: str) -> dict:
    """
    Run SD inpainting evaluation on ControlNet outputs
    """
    controlnet_path = Path(controlnet_output_dir)
    sd_output_path = Path(output_dir) / "sd_outputs"
    sd_output_path.mkdir(parents=True, exist_ok=True)

    from sd_inference import run_sd_inference
    from PIL import Image
    import numpy as np

    metrics = []
    for img_file in controlnet_path.glob("tryon_*.png"):
        try:
            result = run_sd_inference(img_file, img_file, img_file, sd_output_path)
            out_file = sd_output_path / img_file.name
            result.save(out_file)

            arr = np.array(result.convert("RGB"))
            metrics.append({
                "file": out_file.name,
                "mean_pixel": float(arr.mean()),
                "std_pixel": float(arr.std())
            })
            print(f"[SD] Saved {out_file}")
        except Exception as e:
            print(f"[SD] Failed {img_file.name}: {e}")

    print(f"[SD] Evaluation completed for {len(metrics)} images")
    return {"image_stats": metrics}

# -------------------- Main --------------------
if __name__ == "__main__":
    OUTPUT_DIR = r"C:\Users\shris\OneDrive - UTS\Documents\GitHub\AR_STOC\pipeline_outputs"
    
    # Run pipeline
    pipeline_instance = full_pipeline(output_dir=OUTPUT_DIR)

    # Enqueue pipeline to ClearML agent
    pipeline_instance.start(
        queue=QUEUE_NAME,
        repo="https://github.com/ShristiHamal/AR_STOC.git"
    )
