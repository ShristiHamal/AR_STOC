# trigger_pipeline.py
from app.pipeline_decorator import full_pipeline

# Create pipeline instance with parameters
pipeline_instance = full_pipeline(
    split="train",
    output_dir="controlnet_outputs"
)

# Enqueue to ClearML agent
pipeline_instance.start(queue="mlops_pipeline")