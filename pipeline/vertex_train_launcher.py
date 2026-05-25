#!/usr/bin/env python
import os
import sys
from google.cloud import aiplatform

def log(message: str, level: str = "INFO"):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)

def main():
    log("=== VERTEX AI FINE-TUNING JOB LAUNCHER ===")

    # Retrieve parameters from environment variables with safe defaults
    PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "your-gcp-project-id")
    REGION = os.environ.get("GCP_REGION", "us-central1")
    BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "your-ie-finetuning-bucket")
    HF_TOKEN = os.environ.get("HF_TOKEN", "")

    # Check for placeholders
    if PROJECT_ID == "your-gcp-project-id" or BUCKET_NAME == "your-ie-finetuning-bucket":
        log("Please configure GCP_PROJECT_ID and GCS_BUCKET_NAME environment variables first,", "WARNING")
        log("or edit this script with your actual project details.", "WARNING")

    log(f"Initializing Vertex AI SDK (Project: '{PROJECT_ID}', Region: '{REGION}')...")
    try:
        aiplatform.init(
            project=PROJECT_ID,
            location=REGION,
            staging_bucket=f"gs://{BUCKET_NAME}"
        )
    except Exception as e:
        log(f"Failed to initialize Vertex AI SDK: {e}", "ERROR")
        sys.exit(1)

    # GCS output paths
    output_dir = f"gs://{BUCKET_NAME}/models/gemma-industrial-finetuned"
    dataset_gcs_path = f"gs://{BUCKET_NAME}/datasets/dataset_gold_standards.jsonl"

    log("Defining training job container and resources...")
    
    # Official Vertex AI PyTorch GPU training container
    container_uri = "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-1:latest"

    # Define training arguments passed to gemma_train_task.py
    training_args = [
        "--model_name_or_path", os.environ.get("BASE_MODEL_NAME", "google/gemma-2-27b-it"),
        "--dataset_path", dataset_gcs_path,
        "--output_dir", output_dir,
        "--hf_token", HF_TOKEN,
        "--learning_rate", os.environ.get("LEARNING_RATE", "2e-4"),
        "--num_train_epochs", os.environ.get("NUM_EPOCHS", "3"),
        "--per_device_train_batch_size", os.environ.get("BATCH_SIZE", "2"),
        "--gradient_accumulation_steps", os.environ.get("GRADIENT_ACCUMULATION", "4"),
        "--lora_r", os.environ.get("LORA_R", "32"),
        "--lora_alpha", os.environ.get("LORA_ALPHA", "16")
    ]

    # Select GPU configuration based on environment variable
    # Defaults to highly available L4 GPUs (G2-standard nodes) or can use A100 (A2 nodes)
    gpu_type = os.environ.get("GPU_TYPE", "NVIDIA_L4")
    
    if gpu_type == "NVIDIA_A100_80GB":
        machine_type = "a2-ultragpu-1g"
        accelerator_count = 1
        accelerator_type = "NVIDIA_A100_80GB"
    elif gpu_type == "NVIDIA_L4":
        machine_type = "g2-standard-48" # Provision 2x NVIDIA L4 (24GB VRAM each)
        accelerator_count = 2
        accelerator_type = "NVIDIA_L4"
    else:
        # Fallback to standard 1x L4 g2 node
        machine_type = "g2-standard-24"
        accelerator_count = 1
        accelerator_type = "NVIDIA_L4"

    log(f"Provisioning Cloud Resource: {machine_type} ({accelerator_count}x {accelerator_type})...")

    # Construct the CustomJob from local script
    try:
        # Vertex AI copies gemma_train_task.py automatically to the training instance
        job = aiplatform.CustomJob.from_local_script(
            display_name="gemma-industrial-triumvirate-fine-tune",
            script_path="gemma_train_task.py",
            container_uri=container_uri,
            requirements=[
                "transformers>=4.40.0",
                "peft>=0.10.0",
                "datasets>=2.18.0",
                "accelerate>=0.28.0",
                "bitsandbytes>=0.42.0",
                "trl>=0.8.1"
            ],
            machine_type=machine_type,
            accelerator_type=accelerator_type,
            accelerator_count=accelerator_count,
            args=training_args
        )

        log("Submitting custom fine-tuning job to Vertex AI clusters...")
        log("The process is running asynchronously in the cloud. Streaming logs will follow...")
        
        # This will block and stream standard console output from Vertex AI cluster
        job.run(sync=True)
        
        log(f"=== FINE-TUNING COMPLETE ===")
        log(f"Your fine-tuned adapter weights and checkpoints have been saved in: {output_dir}")
        log(f"To download them to your local Mac Mini M4 Pro for conversion, run:")
        log(f"  gcloud storage cp -r {output_dir} ./gemma-industrial-local")

    except Exception as e:
        log(f"An error occurred during Vertex AI job orchestration: {e}", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()
