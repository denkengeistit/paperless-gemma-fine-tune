# Paperless-ngx Gemma Automatic Learning Triumvirate Loop

An autonomous multi-agent pipeline designed to curate and synthesize highly structured multi-turn dialogue training datasets from your **Paperless-ngx** documents and **AnythingLLM** space, and fine-tune a copy of **Gemma** in **Google Cloud Vertex AI** for offline local running on your **Mac Mini M4 Pro**.

---

## 🏛️ System Architecture

Our custom multi-agent framework operates as a lightweight container directly inside your Paperless-ngx Docker stack:

```mermaid
graph TD
    subgraph GCP VPS (Docker Compose Stack)
        A["Paperless-ngx (webserver)"] <-->|Local Net| B["Pipeline Container (Python)"]
        B <-->|Logs & Database| C[("SQLite Checkpoint (curated_documents.db)")]
    end

    subgraph Google Cloud Platform (GCP)
        B -->|1. Automatic GCS Upload| D[("Google Cloud Storage Bucket (gs://...)")]
        D -->|2. Ingest Dataset| E["Vertex AI Model Garden / Custom Job"]
        E -->|3. Fine-tune on NVIDIA A100 or L4 GPUs| F["Fine-Tuned Gemma Model Weights"]
        F -->|4. Save Output| G[("GCS Output Directory")]
    end

    subgraph Local Workstation (Mac Mini M4 Pro)
        G -->|5. Download via gcloud storage cp| H["Mac Mini M4 Pro (48GB)"]
        H -->|6. Local Inference (Ollama / AnythingLLM)| I["Gemma-IE Local Expert Chatbot"]
    end
```

### The Multi-Agent Triumvirate Roles:
*   **Agent 1 (The Student)**: Naive junior engineer; reads raw paper sections from Paperless-ngx and asks specific, technically challenging implementation questions.
*   **Agent 2 (The Teacher)**: Informed senior AI; queries AnythingLLM RAG API, reasons step-by-step inside `<thought>` tags, and explains using center-pinnacle standards.
*   **Agent 3 (The Professor)**: Chief quality auditor; reviews dialogue turns, scores them, and injects dynamic system prompt modifications directly back into the Teacher's state.
*   **Agent 4 (The Scribe)**: Downstream compiler; harvests approved conversations (Professor score $\ge$ 4.0/5.0), formats them into standard ChatML JSONL, and uploads them to a GCS bucket.

## 🧪 Local Testing & Sandboxing (All-in-One Sandbox)

Before deploying to your live GCP VM instance, you can spin up a fully self-contained local testing sandbox containing **Paperless-ngx** (preconfigured with `admin`/`admin` credentials), **AnythingLLM**, **IBM Docling**, and our **Triumvirate Pipeline** container all in one.

### 1. Launch the Sandbox
On your local machine (e.g., your Mac Mini), clone this repository and spin up the complete test environment:
```bash
git clone https://github.com/denkengeistit/paperless-gemma-fine-tune.git
cd paperless-gemma-fine-tune

# Launch all 5 containers (Redis, Paperless, AnythingLLM, Docling, and Pipeline)
docker compose -f docker-compose.local-test.yml up -d --build
```

### 2. Configure the Local Services
Once the containers are running, access and configure them:
*   **Paperless-ngx**: Available at [http://localhost:8010](http://localhost:8010). Log in with credentials `admin` / `admin`, go to settings, and generate an API Token.
*   **AnythingLLM**: Available at [http://localhost:3001](http://localhost:3001). Complete the brief initial setup, generate an API key in the settings tab, and create your test workspace.
*   **IBM Docling API**: Running at [http://localhost:5001](http://localhost:5001). You can view the interactive OpenAPI documentation at `/docs` and the web UI parser playground at `/ui`.

### 3. Bind the Pipeline & Run
Open `docker-compose.local-test.yml` and update the `pipeline` environment variables with your generated test keys:
```yaml
    environment:
      - PAPERLESS_TOKEN=your_test_paperless_api_token_here
      - ANYTHINGLLM_KEY=your_local_anythingllm_api_key_here
      - WORKSPACE_SLUG=your_test_workspace_slug_here
      - OPENROUTER_API_KEY=your_openrouter_api_key_here
```
Restart the pipeline daemon to begin the automated ingestion and learning triumvirate loop:
```bash
# Restart pipeline with active credentials
docker compose -f docker-compose.local-test.yml restart pipeline

# Monitor active student-teacher dialogues
docker compose -f docker-compose.local-test.yml logs -f pipeline
```

---

## 🚀 Step 1: Deploy Ingestion on your GCP Compute VM

Since your Paperless-ngx instance is already running inside Docker, you can run this container directly on the same Docker bridge network.

### 1. Clone this Repo to your VM
Clone your newly created repository directly into your server's Paperless-ngx root directory:
```bash
git clone https://github.com/denkengeistit/paperless-gemma-fine-tune.git
```

### 2. Configure Environment Variables
Copy and rename the compose override template into your stack, then open `docker-compose.override.yml` and enter your credentials:
```yaml
    environment:
      - PAPERLESS_URL=http://webserver:8000           # Update 'webserver' to your actual Paperless container service name
      - PAPERLESS_TOKEN=your_real_paperless_token
      - ANYTHINGLLM_URL=https://useanything.com/api/v1
      - ANYTHINGLLM_KEY=your_real_anythingllm_api_key
      - WORKSPACE_SLUG=your_actual_workspace_slug
      - OPENROUTER_API_KEY=your_real_openrouter_api_key
      
      # --- GCS Upload Config ---
      - GCS_BUCKET_NAME=your-ie-finetuning-bucket     # Name of your GCS bucket
      - GCS_BLOB_PATH=datasets/dataset_gold_standards.jsonl
```

### 3. Build & Start the Pipeline
```bash
# Build and run the pipeline
docker compose up -d pipeline --build

# Monitor live pedagogical logs
docker compose logs -f pipeline
```
Once the curation completes, the Scribe will automatically upload your dataset to GCS at `gs://your-ie-finetuning-bucket/datasets/dataset_gold_standards.jsonl`.

---

## ☁️ Step 2: Fine-Tuning in Google Cloud Vertex AI

With your dataset compiled and residing safely in GCS, you can train **Gemma 4 31B** (or Gemma 2 27B) in the cloud.

### 1. Setup Environment Variables
Configure your GCP credentials and GPU requirements on your terminal:
```bash
export GCP_PROJECT_ID="your-gcp-project-id"
export GCS_BUCKET_NAME="your-ie-finetuning-bucket"
export HF_TOKEN="your_hugging_face_gated_model_token" # To load gated Gemma weights
export GPU_TYPE="NVIDIA_L4"                            # "NVIDIA_L4" (2x L4 GPUs) or "NVIDIA_A100_80GB"
```

### 2. Submit the Fine-Tuning Job
Install the official Vertex SDK and run the pre-configured launcher:
```bash
pip install google-cloud-aiplatform
python pipeline/vertex_train_launcher.py
```
This triggers a high-performance training job. Once completed, your fine-tuned adapter weights are saved in: `gs://your-ie-finetuning-bucket/models/gemma-industrial-finetuned/`.

---

## 💻 Step 3: Run Offline on your Mac Mini M4 Pro

Once Vertex AI finishes training and saves the weights to your GCS output folder, retrieve and compile them:

### 1. Download Model Weights
On your Mac Mini M4 Pro, open a terminal and pull down the weights:
```bash
gcloud storage cp -r gs://your-ie-finetuning-bucket/models/gemma-industrial-finetuned ./gemma-industrial-local
```

### 2. Convert to GGUF (for Ollama/AnythingLLM Desktop)
Convert your fine-tuned model into GGUF using `llama.cpp` tools:
```bash
# Clone llama.cpp and run conversion
python convert-hf-to-gguf.py ./gemma-industrial-local --outfile ./gemma-industrial-31b.gguf --outtype q4_k_m
```

### 3. Load into Ollama
Create a `Modelfile` with the following contents:
```dockerfile
FROM ./gemma-industrial-31b.gguf
```
Then run:
```bash
ollama create gemma-ie
ollama run gemma-ie
```

You now have a custom-tailored, private, and highly capable industrial engineering assistant running locally at zero cost!
