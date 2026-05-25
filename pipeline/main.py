import os
import re
import sys
import json
import sqlite3
import asyncio
import httpx
from datetime import datetime
from typing import List, Dict, Any, Optional

# --- Configuration & Environment Loading ---
try:
    from dotenv import load_dotenv
    dotenv_path = os.path.join(os.path.dirname(__file__), "..", ".env")
    if os.path.exists(dotenv_path):
        load_dotenv(dotenv_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Integration endpoints & access credentials
PAPERLESS_URL = os.environ.get("PAPERLESS_URL", "http://webserver:8000")
PAPERLESS_TOKEN = os.environ.get("PAPERLESS_TOKEN", "")
ANYTHINGLLM_URL = os.environ.get("ANYTHINGLLM_URL", "https://useanything.com/api/v1")
ANYTHINGLLM_KEY = os.environ.get("ANYTHINGLLM_KEY", "")
WORKSPACE_SLUG = os.environ.get("WORKSPACE_SLUG", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Google Cloud Storage Settings
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "")
GCS_BLOB_PATH = os.environ.get("GCS_BLOB_PATH", "datasets/dataset_gold_standards.jsonl")

# Engine Selection on OpenRouter (Gemini 2.5/3.5 models provide high reasoning and massive context capability)
STUDENT_MODEL = os.environ.get("STUDENT_MODEL", "google/gemini-2.5-pro")
TEACHER_MODEL = os.environ.get("TEACHER_MODEL", "google/gemini-2.5-pro")
PROFESSOR_MODEL = os.environ.get("PROFESSOR_MODEL", "google/gemini-2.5-pro")

# Hyperparameters for Triumvirate Loop
TARGET_POOL_SIZE = int(os.environ.get("TARGET_POOL_SIZE", "10000"))
CONCURRENCY_LIMIT = int(os.environ.get("CONCURRENCY_LIMIT", "3"))  # Safe rate limit threshold
SESSION_TURN_LIMIT = int(os.environ.get("SESSION_TURN_LIMIT", "3")) # In-depth back-and-forth turns
MIN_PROFESSOR_SCORE = float(os.environ.get("MIN_PROFESSOR_SCORE", "4.0")) # Approval threshold

# Filesystem Persistence Mounts (Docker Persistent Storage)
DB_PATH = os.environ.get("DB_PATH", "/app/data/curated_documents.db")
EXPORT_PATH = os.environ.get("EXPORT_PATH", "/app/data/dataset_gold_standards.jsonl")

# Target Engineering Domain Keywords for Relevance Matching
CORE_KEYWORDS = [
    "operations research", "linear programming", "integer programming", "queuing theory",
    "stochastic models", "supply chain", "logistics", "ergonomics", "human factors",
    "thermodynamics", "metallurgy", "corrosion", "material selection", "structural analysis",
    "finite element", "fluid dynamics", "vessel design", "industrial engineering",
    "manufacturing process", "quality control", "six sigma", "simulation modeling",
    "stochastic process", "facility layout", "materials handling", "production scheduling"
]

# --- Console Log Utility ---
def log(message: str, level: str = "INFO"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}", flush=True)

# --- SQLite Connection & Table Setup ---
def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    log("Initializing local staging SQLite database...")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Curated documents staged for learning sessions
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS curated_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paperless_id INTEGER UNIQUE,
            source TEXT NOT NULL,         -- 'anythingllm' or 'paperless'
            title TEXT,
            content TEXT,
            quality_score REAL DEFAULT 0.0,
            relevance_score REAL DEFAULT 0.0,
            status TEXT DEFAULT 'pending'  -- 'pending', 'processing', 'completed', 'failed'
        )
    """)
    
    # Tracks completed dialogue sessions and their overall pedagogical scores
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dialogue_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            average_score REAL DEFAULT 0.0,
            status TEXT DEFAULT 'pending',  -- 'pending', 'approved', 'rejected'
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(doc_id) REFERENCES curated_docs(id) ON DELETE CASCADE
        )
    """)
    
    # Turn-by-turn chat history records
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS session_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            turn_index INTEGER NOT NULL,
            role TEXT NOT NULL,            -- 'user' (student) or 'assistant' (teacher)
            content TEXT NOT NULL,
            FOREIGN KEY(session_id) REFERENCES dialogue_sessions(id) ON DELETE CASCADE
        )
    """)
    
    # Professor audit logs and feedback adjustments
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS professor_audits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER NOT NULL,
            turn_index INTEGER NOT NULL,
            score REAL NOT NULL,
            critique TEXT,
            instruction_modification TEXT,
            FOREIGN KEY(session_id) REFERENCES dialogue_sessions(id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()
    log("Staging database and auditing schemas successfully initialized.")

# --- Text Preprocessing Heuristics ---
def get_word_count(text: str) -> int:
    return len(re.findall(r'\w+', text)) if text else 0

def calculate_ocr_quality(text: str) -> float:
    """
    Measures the ratio of standard alphanumeric words to total strings.
    Weeds out documents corrupted by layout artifacts or bad scans.
    """
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    alphabetic_words = sum(1 for w in words if re.match(r'^[a-zA-Z\-]{2,20}$', w))
    return alphabetic_words / len(words)

def calculate_relevance(text: str) -> float:
    """
    Counts matches against central industrial engineering keywords.
    """
    if not text:
        return 0.0
    text_lower = text.lower()
    score = 0.0
    for keyword in CORE_KEYWORDS:
        matches = len(re.findall(re.escape(keyword), text_lower))
        score += matches * 1.5
    return score

# --- Stage 0: Curation & Preprocessing Ingestion Pipeline ---
def run_stage_0_ingestion():
    init_db()
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM curated_docs")
    existing_count = cursor.fetchone()[0]
    log(f"Staging database currently contains {existing_count}/{TARGET_POOL_SIZE} curated documents.")
    
    if existing_count >= TARGET_POOL_SIZE:
        log("10,000 document pool target already satisfied. Skipping Stage 0.")
        conn.close()
        return

    with httpx.Client(timeout=30.0) as client:
        # 1. Pull Hand-Filtered MD Documents from AnythingLLM Cloud
        anything_count = 0
        if ANYTHINGLLM_KEY and WORKSPACE_SLUG:
            try:
                log(f"Connecting to AnythingLLM space '{WORKSPACE_SLUG}' for hand-curated MD files...")
                headers = {"Authorization": f"Bearer {ANYTHINGLLM_KEY}"}
                response = client.get(f"{ANYTHINGLLM_URL}/workspace/{WORKSPACE_SLUG}/documents", headers=headers)
                
                if response.status_code == 200:
                    workspace_data = response.json()
                    docs_list = []
                    if "workspace" in workspace_data and "documents" in workspace_data["workspace"]:
                        docs_list = workspace_data["workspace"]["documents"]
                    elif "documents" in workspace_data:
                        docs_list = workspace_data["documents"]
                        
                    for doc in docs_list:
                        content = doc.get("content", "")
                        title = doc.get("title", f"AnythingLLM Doc {doc.get('id')}")
                        if not content and "text" in doc:
                            content = doc["text"]
                        if not content:
                            continue
                            
                        # Insert manual filtered docs as maximum quality & relevance (Gold standard)
                        cursor.execute("""
                            INSERT OR IGNORE INTO curated_docs (source, title, content, quality_score, relevance_score)
                            VALUES ('anythingllm', ?, ?, 1.0, 10.0)
                        """, (title, content))
                        anything_count += 1
                    conn.commit()
                    log(f"Loaded {anything_count} hand-filtered Markdown files from AnythingLLM workspace.")
                else:
                    log(f"AnythingLLM API returned status {response.status_code}. Skipping AnythingLLM pull.", "WARNING")
            except Exception as e:
                log(f"Error pulling from AnythingLLM: {e}", "ERROR")
        else:
            log("AnythingLLM workspace parameters omitted. Skipping AnythingLLM pull.")

        # Recalculate remaining expansion count
        cursor.execute("SELECT COUNT(*) FROM curated_docs")
        current_count = cursor.fetchone()[0]
        needed_docs = TARGET_POOL_SIZE - current_count
        
        if needed_docs <= 0:
            log(f"Staging database loaded to capacity ({current_count}/{TARGET_POOL_SIZE}) with pre-filtered MD files.")
            conn.close()
            return
            
        # 2. Extract & Filter remaining quota from Paperless-ngx
        if not PAPERLESS_TOKEN:
            log("No PAPERLESS_TOKEN provided. Cannot perform Paperless-ngx document expansion.", "ERROR")
            conn.close()
            sys.exit(1)
            
        log(f"Expanding staging database with {needed_docs} documents from Paperless-ngx...")
        headers = {
            "Authorization": f"Token {PAPERLESS_TOKEN}",
            "Accept": "application/json; version=6"
        }
        
        url = f"{PAPERLESS_URL}/api/documents/"
        page = 1
        paperless_count = 0
        skipped_short_long = 0
        skipped_poor_ocr = 0
        skipped_no_relevance = 0
        
        while url and needed_docs > 0:
            try:
                response = client.get(url, headers=headers)
                if response.status_code != 200:
                    log(f"Paperless API error on page {page}: Status {response.status_code}", "ERROR")
                    break
                    
                data = response.json()
                results = data.get("results", [])
                if not results:
                    break
                    
                for doc in results:
                    doc_id = doc.get("id")
                    title = doc.get("title", f"Paperless Doc {doc_id}")
                    content = doc.get("content", "")
                    
                    if not content:
                        continue
                        
                    # Filter: Length Check
                    word_count = get_word_count(content)
                    if word_count < 200 or word_count > 15000:
                        skipped_short_long += 1
                        continue
                        
                    # Filter: OCR gibberish Check
                    ocr_quality = calculate_ocr_quality(content)
                    if ocr_quality < 0.65:
                        skipped_poor_ocr += 1
                        continue
                        
                    # Filter: Topic Relevance Match
                    relevance_score = calculate_relevance(content)
                    if relevance_score < 1.0:
                        skipped_no_relevance += 1
                        continue
                        
                    # Insert validated document
                    cursor.execute("""
                        INSERT OR IGNORE INTO curated_docs (paperless_id, source, title, content, quality_score, relevance_score)
                        VALUES (?, 'paperless', ?, ?, ?, ?)
                    """, (doc_id, title, content, ocr_quality, relevance_score))
                    
                    paperless_count += 1
                    needed_docs -= 1
                    if needed_docs == 0:
                        break
                        
                conn.commit()
                url = data.get("next")
                page += 1
                
                log(f"Staged page {page}. Current DB count: {TARGET_POOL_SIZE - needed_docs}/{TARGET_POOL_SIZE}")
                log(f"Pre-filter stats -> Word-count skips: {skipped_short_long} | OCR-quality skips: {skipped_poor_ocr} | Relevance skips: {skipped_no_relevance}")
                
            except Exception as e:
                log(f"Error querying Paperless-ngx on page {page}: {e}", "ERROR")
                break
                
    conn.close()
    log("Stage 0 (Preprocessing & Curation) successfully executed.")

# --- OpenRouter Connection Helper ---
async def call_openrouter(client: httpx.AsyncClient, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Submits an async request to OpenRouter with resilient retries, backoff, and rate-limit safety.
    """
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/google-deepmind/antigravity",
        "X-Title": "IE Automatic Triumvirate Generation Loop"
    }
    
    max_retries = 5
    backoff = 2.0
    
    for attempt in range(max_retries):
        try:
            response = await client.post(url, headers=headers, json=payload, timeout=90.0)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                log(f"Rate limited by OpenRouter (429). Throttling for {backoff} seconds...", "WARNING")
                await asyncio.sleep(backoff)
                backoff *= 2.0
            else:
                log(f"OpenRouter API returned error status {response.status_code}: {response.text}", "ERROR")
                await asyncio.sleep(backoff)
                backoff *= 1.5
        except Exception as e:
            log(f"Network exception on OpenRouter call (Attempt {attempt+1}/{max_retries}): {e}", "WARNING")
            await asyncio.sleep(backoff)
            backoff *= 1.5
            
    return None

# --- Stage 1: The Automatic Learning Triumvirate Loop Execution ---

STUDENT_PROMPT = """You are a junior operations researcher and field technician working at the industrial engineering center.
You have been handed a raw technical report: "{document_title}".
Your job is to read this document and try to understand how to implement its findings in a live production environment.

You are communicating with our Center's Senior Consulting AI (The Teacher).
Since you are a junior engineer, you do not have access to our central RAG vector knowledge base—you only have this raw paper in front of you.

Follow these strict conversational constraints:
1. DO NOT say "Hello", "Thank you", "I would be happy to", or use any pleasantries. Be professional, direct, and slightly overwhelmed.
2. Read the document text and formulate a highly specific, challenging question about a design constraint, safety threshold, or mathematical model mentioned in the text.
3. Ask the Teacher to explain how this works or how to calculate it.
4. Once the Teacher answers, review it against your raw text. If the Teacher's answer is generic, point it out! Say: "The report mentions equation X, but you didn't explain how variable Y scales in that scenario. Can you clarify?"
5. Limit the conversation to 3 turns, then output "[STUDY_SESSION_COMPLETE]"."""

TEACHER_PROMPT = """You are the Distinguished Principal Consulting AI of the Industrial Engineering Research Center.
You are teaching a junior field technician (The Student) who is trying to implement a technical report.
You have full access to our AnythingLLM RAG Database, which retrieves pristine, contextually complete chunks of our center's 40 years of wisdom.

Your goal is to explain concepts clearly, provide step-by-step mathematical reasoning, and ensure our safety standards are strictly communicated.

CRITICAL Pedagogy Instructions:
1. Always outline your step-by-step logical reasoning inside a "<thought>" block before writing your final response.
2. Never skip mathematical formulas, material tolerances, or safety factors. Give concrete engineering answers, not summaries.
3. Keep your tone authoritative, direct, and professional. Avoid fluffy, enthusiastic AI filler words.

Current Professor's Instruction Modifications:
{dynamic_feedback_modifier}"""

PROFESSOR_PROMPT = """You are the Emeritus Professor of Industrial Engineering and chief Quality Auditor of our Center's training curriculum.
You are reviewing a dialogue between a Junior Student and our Consulting Teacher.

Your task is to review the latest dialogue exchange and evaluate the Teacher on:
1. **Factual Grounding**: Did the Teacher's answer accurately represent the formulas and tolerances in the reference context?
2. **Pedagogical Clarity**: Did the Teacher break down the math clearly or did it give a generic hand-waving summary?
3. **Safety Compliance**: Did the Teacher omit any critical threshold guidelines?

You must output your evaluation strictly in JSON format.
If the Teacher was vague or skipped math, generate an "instruction_modification" for the Teacher. This modification will be injected into the Teacher's system prompt for the next turn, instructing it exactly how to improve.

Output JSON format:
{{
  "score": 4.5, // 1.0 to 5.0
  "critique": "The Teacher correctly explained queuing theory but failed to outline the transition state equation.",
  "instruction_modification": "For your next response: You must write out the explicit differential equation for the transition states and define every variable."
}}"""

async def run_triumvirate_orchestrator(client: httpx.AsyncClient, doc_id: int, title: str, content: str, conn_lock: asyncio.Lock):
    """
    Manages the multi-agent state machine (Student, Teacher, Professor) 
    over a 3-turn interactive study dialogue session.
    """
    log(f"Launching Triumvirate study session for Doc {doc_id}: '{title}'")
    
    # 1. Initialize Dialogue Session record in database
    async with conn_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO dialogue_sessions (doc_id, status) VALUES (?, 'pending')", (doc_id,))
        session_id = cursor.lastrowid
        conn.commit()
        conn.close()

    # Initial state configurations
    dynamic_feedback_modifier = "None. Maintain standard professional consulting guidelines."
    session_messages = []  # In-memory tracking of conversation
    scores = []
    
    # We run exactly 3 rounds of back-and-forth tutoring
    for turn in range(SESSION_TURN_LIMIT):
        # --- ROUND A: Student (Agent 1) asks a question ---
        student_system = STUDENT_PROMPT.format(document_title=title)
        student_payload = {
            "model": STUDENT_MODEL,
            "messages": [
                {"role": "system", "content": student_system},
                {"role": "user", "content": f"RAW TECHNICAL REPORT:\n{content}\n\nDIALOGUE CONVERSATION SO FAR:\n{json.dumps(session_messages)}\n\nFormulate your next technical question."}
            ]
        }
        
        student_res = await call_openrouter(client, student_payload)
        if not student_res:
            log(f"Student Agent failed to respond on Doc {doc_id} Turn {turn}. Skipping round.", "WARNING")
            break
            
        student_question = student_res["choices"][0]["message"]["content"].strip()
        
        if "[STUDY_SESSION_COMPLETE]" in student_question or not student_question:
            log(f"Student signaled session completion on Doc {doc_id} at Turn {turn}.")
            break
            
        session_messages.append({"role": "user", "content": student_question})
        
        # --- ROUND B: Teacher (Agent 2) queries RAG & Explains ---
        teacher_system = TEACHER_PROMPT.format(dynamic_feedback_modifier=dynamic_feedback_modifier)
        
        # We query the generator model acting as the Teacher RAG.
        # Since AnythingLLM API handles context retrieval, we simulate the RAG loop:
        # We query the OpenRouter endpoint providing BOTH the teacher system instructions and the document's raw content
        # (This simulates the RAG document injection perfectly, grounding the response).
        teacher_payload = {
            "model": TEACHER_MODEL,
            "messages": [
                {"role": "system", "content": teacher_system},
                {"role": "user", "content": f"CONTEXT RECORD:\n{content}\n\nSTUDENT INQUIRY:\n{student_question}\n\nFormulate your comprehensive teaching response now."}
            ]
        }
        
        teacher_res = await call_openrouter(client, teacher_payload)
        if not teacher_res:
            log(f"Teacher Agent failed to respond on Doc {doc_id} Turn {turn}. Skipping round.", "WARNING")
            break
            
        teacher_explanation = teacher_res["choices"][0]["message"]["content"].strip()
        session_messages.append({"role": "assistant", "content": teacher_explanation})
        
        # --- ROUND C: Professor (Agent 3) Evaluates & Dynamic feedback loop ---
        professor_system = PROFESSOR_PROMPT
        professor_payload = {
            "model": PROFESSOR_MODEL,
            "messages": [
                {"role": "system", "content": professor_system},
                {"role": "user", "content": f"REFERENCE DOCUMENT TEXT:\n{content}\n\nLATEST DIALOGUE EXCHANGE:\nStudent Question: {student_question}\nTeacher Explanation: {teacher_explanation}\n\nGrade the Teacher's performance."}
            ],
            "response_format": {"type": "json_object"}
        }
        
        prof_res = await call_openrouter(client, professor_payload)
        prof_score = 0.0
        prof_critique = "Failed to parse Professor evaluation."
        
        if prof_res:
            try:
                prof_raw = prof_res["choices"][0]["message"]["content"]
                parsed_prof = json.loads(prof_raw)
                prof_score = float(parsed_prof.get("score", 0.0))
                prof_critique = parsed_prof.get("critique", "No critique provided.")
                
                # Dynamic Feedback Injection: Adjust teacher's system instructions for next turn!
                dynamic_feedback_modifier = parsed_prof.get("instruction_modification", "None. Continue standard consulting.")
            except Exception as pe:
                log(f"Professor JSON parse error on Doc {doc_id} Turn {turn}: {pe}", "WARNING")
                
        scores.append(prof_score)
        
        # Write Round to Database
        async with conn_lock:
            conn = get_db_connection()
            cursor = conn.cursor()
            # Save messages
            cursor.execute("""
                INSERT INTO session_messages (session_id, turn_index, role, content)
                VALUES (?, ?, 'user', ?)
            """, (session_id, turn, student_question))
            cursor.execute("""
                INSERT INTO session_messages (session_id, turn_index, role, content)
                VALUES (?, ?, 'assistant', ?)
            """, (session_id, turn, teacher_explanation))
            # Save Professor's Audit
            cursor.execute("""
                INSERT INTO professor_audits (session_id, turn_index, score, critique, instruction_modification)
                VALUES (?, ?, ?, ?, ?)
            """, (session_id, turn, prof_score, prof_critique, dynamic_feedback_modifier))
            conn.commit()
            conn.close()
            
        log(f"Doc {doc_id} Turn {turn} Complete. Prof Score: {prof_score}/5.0 | Critique: {prof_critique[:80]}...")

    # Calculate final session metrics and update status
    avg_score = sum(scores) / len(scores) if scores else 0.0
    status = "approved" if avg_score >= MIN_PROFESSOR_SCORE else "rejected"
    
    async with conn_lock:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE dialogue_sessions 
            SET average_score = ?, status = ?
            WHERE id = ?
        """, (avg_score, status, session_id))
        cursor.execute("""
            UPDATE curated_docs 
            SET status = 'completed'
            WHERE id = ?
        """, (doc_id,))
        conn.commit()
        conn.close()
        
    log(f"Completed Study Session {session_id} for Doc {doc_id}: Pedagogical Average {avg_score:.2f}/5.0. Status: {status.upper()}")

# --- Async Worker Pool Orchestration ---
async def triumvirate_worker(worker_id: int, queue: asyncio.Queue, client: httpx.AsyncClient, conn_lock: asyncio.Lock):
    log(f"Triumvirate Worker {worker_id} initialized.")
    while True:
        doc = await queue.get()
        if doc is None:
            queue.task_done()
            break
            
        doc_id = doc["id"]
        title = doc["title"]
        content = doc["content"]
        
        try:
            await run_triumvirate_orchestrator(client, doc_id, title, content, conn_lock)
        except Exception as e:
            log(f"Worker {worker_id} encountered critical exception on Doc {doc_id}: {e}", "ERROR")
            async with conn_lock:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("UPDATE curated_docs SET status = 'failed' WHERE id = ?", (doc_id,))
                conn.commit()
                conn.close()
        finally:
            queue.task_done()
            
    log(f"Worker {worker_id} gracefully shutdown.")

async def run_stage_1_generation_async():
    if not OPENROUTER_API_KEY:
        log("No OPENROUTER_API_KEY environment variable provided. Cannot execute Triumvirate loop.", "ERROR")
        sys.exit(1)
        
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Extract pending documents
    cursor.execute("SELECT id, title, content FROM curated_docs WHERE status IN ('pending', 'failed')")
    pending_docs = cursor.fetchall()
    conn.close()
    
    total = len(pending_docs)
    log(f"Staged Queue contains {total} documents ready for the Automatic Triumvirate Loop.")
    
    if total == 0:
        log("No pending documents in queue. Stage 1 Triumvirate loop complete.")
        return
        
    # Queue population
    queue = asyncio.Queue()
    for doc in pending_docs:
        await queue.put(doc)
        
    conn_lock = asyncio.Lock()
    
    # Configure low concurrency to respect OpenRouter TPM/RPM limit envelopes
    limits = httpx.Limits(max_keepalive_connections=CONCURRENCY_LIMIT, max_connections=CONCURRENCY_LIMIT * 2)
    async with httpx.AsyncClient(limits=limits, timeout=120.0) as client:
        workers = []
        for w_id in range(CONCURRENCY_LIMIT):
            task = asyncio.create_task(triumvirate_worker(w_id, queue, client, conn_lock))
            workers.append(task)
            
            # Sentinel stops
            await queue.put(None)
            
        await asyncio.gather(*workers)
        
    log("Stage 1 (Automatic Learning Triumvirate Loop) completed successfully.")

# --- Stage 2: Scribe Extractor (Agent 4) ---
def run_stage_2_scribe_export():
    log("Agent 4 (The Scribe) starting extraction process...")
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Select all messages from approved sessions (average score >= 4.0)
    cursor.execute("""
        SELECT s.id as session_id, m.role, m.content 
        FROM dialogue_sessions s
        JOIN session_messages m ON s.id = m.session_id
        WHERE s.status = 'approved'
        ORDER BY s.id, m.id
    """)
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        log("Scribe found zero approved sessions in the staging database. Skip export.", "WARNING")
        return
        
    # Group messages by session
    sessions_dict = {}
    for row in rows:
        s_id = row["session_id"]
        if s_id not in sessions_dict:
            sessions_dict[s_id] = []
            
        # Post-processing: clean output strings and strip administrative bracket codes
        cleaned_content = row["content"]
        cleaned_content = re.sub(r'\[STUDY_SESSION_COMPLETE\]', '', cleaned_content)
        cleaned_content = cleaned_content.strip()
        
        sessions_dict[s_id].append({
            "role": row["role"],
            "content": cleaned_content
        })
        
    os.makedirs(os.path.dirname(EXPORT_PATH), exist_ok=True)
    
    exported_count = 0
    with open(EXPORT_PATH, "w", encoding="utf-8") as f:
        for s_id, messages in sessions_dict.items():
            if len(messages) < 2:
                continue  # Exclude single message sessions if any
                
            record = {"messages": messages}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            exported_count += 1
            
    log(f"The Scribe successfully exported {exported_count} approved multi-turn study sessions to: {EXPORT_PATH}")
    log("Dialogue dataset is formatted in standard, fine-tuning ready ChatML JSONL.")

# --- Cloud Upload Step ---
def run_gcs_upload():
    """
    Saves the compiled training JSONL file directly to the user's GCS bucket.
    If running inside GCP Compute VM, storage.Client() automatically inherits
    the VM's project IAM service credentials.
    """
    if not GCS_BUCKET_NAME:
        log("GCS_BUCKET_NAME not specified in environment variables. Skipping automatic upload.")
        return
        
    log(f"Agent 4 (The Scribe) initiating GCS upload to bucket: gs://{GCS_BUCKET_NAME}/{GCS_BLOB_PATH}...")
    try:
        from google.cloud import storage
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(GCS_BLOB_PATH)
        
        blob.upload_from_filename(EXPORT_PATH)
        log("The Scribe successfully committed and uploaded the fine-tuning dataset to Google Cloud Storage!")
    except Exception as e:
        log(f"The Scribe failed GCS upload: {e}. You can manually upload standard file: {EXPORT_PATH}", "ERROR")

# --- Main Runtime Script ---
if __name__ == "__main__":
    log("--- AUTOMATIC LEARNING TRIUMVIRATE LOOP PROCESSOR STARTING ---")
    
    # 1. stage 0 Ingestion
    log("=== Stage 0: Executing Document Ingestion, Quality Checks, & Relevance Sorting ===")
    run_stage_0_ingestion()
    
    # 2. stage 1 Multi-Agent Loop
    log("=== Stage 1: Launching Student-Teacher-Professor Multi-Agent Training Loop ===")
    asyncio.run(run_stage_1_generation_async())
    
    # 3. stage 2 Scribe Compiler
    log("=== Stage 2: Deploying Agent 4 (The Scribe) Dataset Compilation ===")
    run_stage_2_scribe_export()
    
    # 4. GCS Upload Step
    log("=== Stage 3: Executing Scribe Auto-Upload to Google Cloud Storage ===")
    run_gcs_upload()
    
    log("--- TRIUMVIRATE LOOP PROCESSOR TERMINATED SUCCESSFULLY ---")
