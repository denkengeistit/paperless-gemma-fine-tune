#!/usr/bin/env python
import os
import argparse
import sys
import shutil
from datetime import datetime

def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)

# Parse command line parameters
def parse_args():
    parser = argparse.ArgumentParser(description="Gemma Fine-Tuning PyTorch Task")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-2-27b-it", help="Pre-trained Gemma model name or path")
    parser.add_argument("--dataset_path", type=str, required=True, help="Local path or GCS path (gs://...) to the training ChatML JSONL dataset")
    parser.add_argument("--output_dir", type=str, required=True, help="Local or GCS destination path for fine-tuned weights")
    parser.add_argument("--hf_token", type=str, default="", help="Hugging Face API token for gated access to Gemma models")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA Rank parameter")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA Alpha scaling parameter")
    parser.add_argument("--max_seq_length", type=int, default=4096, help="Maximum training context sequence length")
    return parser.parse_args()

def download_from_gcs(gcs_uri: str, local_path: str):
    """
    Downloads a file from a Google Cloud Storage URI to a local filesystem path.
    """
    log(f"Downloading {gcs_uri} to local path {local_path}...")
    try:
        from google.cloud import storage
        storage_client = storage.Client()
        
        # Parse gs:// bucket name and blob path
        path_parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = path_parts[0]
        blob_path = path_parts[1]
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        log("File download completed successfully.")
    except Exception as e:
        log(f"Failed to download dataset from GCS: {e}", "ERROR")
        raise e

def upload_to_gcs(local_dir: str, gcs_uri: str):
    """
    Uploads an entire local directory to a GCS folder.
    """
    log(f"Uploading local directory {local_dir} to GCS URI {gcs_uri}...")
    try:
        from google.cloud import storage
        storage_client = storage.Client()
        
        # Parse bucket and prefix
        path_parts = gcs_uri.replace("gs://", "").split("/", 1)
        bucket_name = path_parts[0]
        gcs_prefix = path_parts[1] if len(path_parts) > 1 else ""
        
        bucket = storage_client.bucket(bucket_name)
        
        for root, _, files in os.walk(local_dir):
            for file in files:
                local_file_path = os.path.join(root, file)
                # Compute relative path to maintain folder hierarchy
                relative_path = os.path.relpath(local_file_path, local_dir)
                gcs_blob_name = os.path.join(gcs_prefix, relative_path).replace("\\", "/")
                
                log(f"Uploading {relative_path} to gs://{bucket_name}/{gcs_blob_name}...")
                blob = bucket.blob(gcs_blob_name)
                blob.upload_from_filename(local_file_path)
                
        log("Directory upload completed successfully.")
    except Exception as e:
        log(f"Failed to upload model weights to GCS: {e}", "ERROR")
        raise e

def main():
    args = parse_args()
    log("=== STARTING CLOUD GPU FINE-TUNING TASK ===")

    # 1. Login to Hugging Face if gated model access is required
    if args.hf_token:
        log("Logging into Hugging Face Hub...")
        try:
            from huggingface_hub import login
            login(token=args.hf_token)
        except Exception as e:
            log(f"Hugging Face login failed: {e}", "WARNING")
    else:
        log("No Hugging Face token specified. Proceeding with public models or local cache.")

    # 2. Resiliently stage the dataset locally
    local_dataset_path = "./dataset.jsonl"
    if args.dataset_path.startswith("gs://"):
        download_from_gcs(args.dataset_path, local_dataset_path)
    else:
        local_dataset_path = args.dataset_path
        log(f"Using local dataset path: {local_dataset_path}")

    # 3. Load Dataset
    log("Loading and parsing JSONL dataset...")
    try:
        from datasets import load_dataset
        # Load local JSONL dataset
        dataset = load_dataset("json", data_files=local_dataset_path, split="train")
        log(f"Loaded {len(dataset)} approved dialog training examples.")
    except Exception as e:
        log(f"Failed to load dataset: {e}", "ERROR")
        sys.exit(1)

    # 4. Import PyTorch and Transformers
    log("Loading deep learning frameworks...")
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig, TaskType, get_peft_model
    from trl import SFTTrainer

    # Check for available CUDA GPUs
    device_count = torch.cuda.device_count()
    log(f"PyTorch reports {device_count} GPU(s) available.")
    for i in range(device_count):
        log(f"  GPU [{i}]: {torch.cuda.get_device_name(i)}")

    # 5. Initialize Tokenizer & Model
    log(f"Initializing base model and tokenizer: '{args.model_name_or_path}'")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.hf_token)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Padding side must be right for CausalLM training
    
    # Check VRAM limits and precision support (bfloat16 is native to modern GPUs like A100/L4)
    # L4 and A100 support BF16 natively with massive performance improvements.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    log(f"Selected computation precision dtype: {compute_dtype}")

    # Load Model with device_map auto to automatically leverage multi-GPU setups on the node
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=compute_dtype,
        device_map="auto",
        token=args.hf_token
    )

    # 6. Apply PEFT/LoRA (Quantization-free LoRA or QLoRA depending on setup)
    # Since we are running on premium cloud GPUs (A100 or 2x/4x L4), we run full-precision LoRA 
    # inside bfloat16 for maximum convergence speed and gradient fidelity.
    log(f"Configuring LoRA Adapter parameters (Rank={args.lora_r}, Alpha={args.lora_alpha})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 7. Preprocess chat dialogue into Tokenizer-compatible formatting
    # Converts {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
    # into standard model chat formatted strings.
    def format_prompts(batch):
        formatted_dialogs = []
        for messages in batch["messages"]:
            # Uses the model's native chat template (Gemma supports standard chat formats)
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            formatted_dialogs.append(text)
        return {"text": formatted_dialogs}

    log("Preprocessing dataset with chat formatting templates...")
    dataset = dataset.map(format_prompts, batched=True)

    # 8. Setup SFT Training Arguments
    # Local directory to run the training and save temporary checkpoints
    local_output_dir = "./results"
    
    training_args = TrainingArguments(
        output_dir=local_output_dir,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim="paged_adamw_32bit", # Memory resilient optimizer
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="no",
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        report_to="none",
        ddp_find_unused_parameters=False
    )

    # 9. Initialize Supervised Fine-Tuning (SFT) Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=lora_config
    )

    # 10. Execute Training
    log("=== LAUNCHING PYTORCH OPTIMIZED TRAINING LOOP ===")
    trainer.train()
    log("Training loop completed successfully!")

    # 11. Save final PEFT Adapters locally
    final_local_dir = os.path.join(local_output_dir, "final_adapters")
    log(f"Saving final adapter weights locally to: {final_local_dir}")
    trainer.save_model(final_local_dir)
    tokenizer.save_pretrained(final_local_dir)

    # 12. Persist output back to GCS bucket for Mac Mini retrieval
    if args.output_dir.startswith("gs://"):
        upload_to_gcs(final_local_dir, args.output_dir)
    else:
        # Move local results if output_dir is just a local folder
        log(f"Copying final results from {final_local_dir} to local path {args.output_dir}...")
        os.makedirs(args.output_dir, exist_ok=True)
        for item in os.listdir(final_local_dir):
            s = os.path.join(final_local_dir, item)
            d = os.path.join(args.output_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)

    log("=== VERTEX AI DEEP LEARNING WORKER EXITED SUCCESSFULLY ===")

if __name__ == "__main__":
    main()
