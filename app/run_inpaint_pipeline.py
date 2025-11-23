# app/run_inpaint_pipeline.py

from clearml import Task
from app.pipeline_decorator import full_pipeline

if __name__ == "__main__":
    # Optional: explicitly create a Task wrapper for running the pipeline
    task = Task.init(
        project_name="AR_STOC",
        task_name="Run AR_TryOn_Inpaint_Batch",
        task_type=Task.TaskTypes.controller,
    )

    full_pipeline(
        CLEARML_DATASET_ID="936ce7ce676a41eca85cecfc59f1d6db",
        train_pairs_relpath="train_pairs.txt",
        sample_ratio=0.02,
        output_dir="./pipeline_inpaint_outputs",
    )

    task.close()
