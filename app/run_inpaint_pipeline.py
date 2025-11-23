# app/run_inpaint_pipeline.py

from clearml import Task
from pipeline_decorator import full_pipeline

if __name__ == "__main__":
    Task.init(project_name="AR_STOC", task_name="Run Inpaint Pipeline")

    full_pipeline(
        CLEARML_DATASET_ID="936ce7ce676a41eca85cecfc59f1d6db",
        train_pairs_relpath="train_pairs.txt",
        sample_ratio=0.05,   # use 5% for speed
        output_dir="./inpaint_outputs",
    )
