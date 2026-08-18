# Code Review: paperless-gemma-fine-tune

Date: 2026-08-18 · Reviewer: salamander (kanban t_8f7c7049)
Rev: 3d1b6b8 (+ fix commit removing the stale hardcoded model default)

## 1. Architecture assessment

Three loosely coupled components:

A. Ingestion + Triumvirate loop (`pipeline/main.py`, `pipeline/Dockerfile.pipeline`)
   Stage 0: pull hand-filtered docs from AnythingLLM, top up from Paperless-ngx
   with heuristic filters (word count 200–15000, OCR-quality >= 0.65, keyword
   relevance >= 1.0). Stage 1: async Student/Teacher/Professor loop over
   OpenRouter, 3 turns/doc, results in SQLite. Stage 2: Scribe exports approved
   sessions (avg >= 4.0) to ChatML JSONL, optional GCS upload.
   Assessment: staged design with a resumable SQLite checkpoint is sound for a
   batch pipeline. The Professor's dynamic feedback injection into the Teacher
   prompt is a genuinely good pattern. Main structural weaknesses: doc content
   (full text up to ~15k words) is echoed into EVERY prompt for every turn —
   token waste and a context-overflow risk on long docs; the "RAG" described in
   the Teacher prompt is simulated by pasting raw content, not by calling the
   AnythingLLM RAG API, so the prompt text misrepresents what actually happens
   (cosmetic, but confusing); SQLite is accessed synchronously under an asyncio
   lock — fine at CONCURRENCY_LIMIT=3, becomes a bottleneck if raised.

B. Vertex AI fine-tune (`pipeline/vertex_train_launcher.py`, `pipeline/gemma_train_task.py`)
   Launcher submits a CustomJob from a local script with a Vertex PyTorch GPU
   container; task script downloads JSONL from GCS, runs LoRA SFT (TRL), uploads
   adapters back to GCS. Assessment: correct shape. Issues: launcher passes
   `--hf_token` even when empty (empty-string arg) and never passes
   `--max_seq_length`, so the 4096 default is silently used; `torch_dtype=`
   is deprecated in transformers>=4.56 (use `dtype=`); SFTTrainer API varies a
   lot across TRL versions — pin `trl` exactly, `>=0.8.1` invites breakage.

C. Document quality assessment (`paperless_quality_assessment.py`)
   Standalone script: Paperless API -> LM Studio chat-completions quality
   scoring with a heuristic fallback, optional continuous loop.
   Assessment: good JSON-repair/parsing fallbacks and heuristic degradation.
   The dead custom-field update path and several smaller issues below.

## 2. Fix applied in this change

- Removed the hardcoded legacy Gemma model default for GEMMA_MODEL in
  paperless_quality_assessment.py. LM Studio was unreachable from the VPS
  during review (GET http://100.119.61.113:1234/v1/models timed out), so no
  live model name could be substituted. GEMMA_MODEL now defaults to empty and
  the script exits with a clear error if unset.
- `.env.example`: `GEMMA_MODEL=` left blank with a comment requiring explicit
  setting; removed the duplicate PAPERLESS_TOKEN entry.

## 3. Issues found

### Critical

C1. Hardcoded legacy model name for GEMMA_MODEL (paperless_quality_assessment.py:62,
    .env.example:53). WRONG vs the model actually loaded in LM Studio — FIXED
    in this commit.

C2. plaintext secrets committed to the repo: docker-compose.local-test.yml ships
    a Paperless API token (line 80) and an AnythingLLM API key (line 85). Even
    for a sandbox these end up in git history and get reused by muscle memory.
    Rotate the AnythingLLM key (it may be a non-local key) and move both to
    .env. NOT fixed here (requires key rotation decision).

C3. No input validation on GEMMA_MODEL — an empty/wrong model name previously
    produced silent per-document heuristic-fallback runs (LM Studio error
    swallowed -> assess_document returns None/it retries forever every
    interval). FAIL-FAST check added in this commit.

### High

H1. main.py process pool starvation: `await queue.put(None)` sentinels are
    enqueued INSIDE the worker-spawn loop *after* create_task; workers start
    consuming immediately and the FIFO means early workers drain real docs
    while later workers can hit a sentinel as their first item and exit, then
    the queue refills docs that only one worker processes. Still correct (put()
    of remaining sentinels proceeds) but concurrency degenerates. Fix: spawn N
    workers, then put N sentinels.

H2. run_stage_1_generation_async re-processes 'failed' docs on every restart
    with no retry counter — a doc that deterministically crashes a worker is
    retried forever. Add attempts column.

H3. Word-count filter (200 min) silently drops short-but-valid documents
    (known issue, confirmed at main.py:267). Make MIN_WORDS/MAX_WORDS env- or
    CLI-configurable.

H4. Local-model call patterns: paperless_quality_assessment assess_document
    sets temperature/max_tokens (good), but there is no retry/backoff around
    LM Studio calls; a transient timeout permanently skips that document's
    LLM score for the run. OpenRouter calls in main.py have retries; LM Studio
    (the flaky one) has none. Add 3x exponential backoff.

H5. httpx client hangs: main.py uses one 120s timeout for the AsyncClient;
    an Ollama/LM-Studio-style stall on a single stream can park a worker for
    2 minutes with no watchdog. Set distinct connect/read timeouts and a
    per-document soft time budget.

H6. paperless_quality_assessment: `update_document_custom_fields` PATCHes
    {'custom_fields': {name: value}} but Paperless expects
    [{'field': id, 'value': v}] — even if wired up, it would 400. The call
    site is currently dead code (see D1).

### Medium / Low (report only)

M1. Dead code: custom-field update block in assess_all_documents is a
    `pass` with comment "Update logic would go here" (line ~347);
    update_document_custom_fields and get_custom_field_ids are therefore
    unused end-to-end. Either implement or delete.

M2. main.py unused imports: none severe, but `Optional` used without issue;
    verify with ruff/flake8 (not run here). `tqdm` is pinned in
    pipeline/requirements.txt but never imported in main.py.

M3. teacher/student calls don't set response_format, professor does; also
    `{{`/`}}` in PROFESSOR_PROMPT is escaped for .format() but the prompt is
    used unformatted — the Professor sees doubled braces in its system prompt.
    Harmless for JSON mode but sloppy; unescape.

M4. Full document content injected once per agent per turn (9x per doc
    upper bound). Cache/truncate (e.g. first 8k chars) and pass a shared
    condensed context; will cut OpenRouter cost materially.

M5. SQLite: no PRAGMA journal_mode=WAL; concurrent-ish access via lock is
    safe but writer stalls. Also no index on curated_docs.status.

M6. vertex_train_launcher: `training_args` includes `--hf_token ""` when
    HF_TOKEN unset; harmless but noisy. Missing --max_seq_length passthrough.
    Container `pytorch-gpu.2-1:latest` floats on `latest` — pin a digest.

M7. .env.example had PAPERLESS_TOKEN declared twice (fixed). README Step 2
    says "Gemma 4 31B" but launcher defaults to gemma-2-27b-it — update docs.

M8. docker-compose.override.yml relies on network `default` with a comment
    suggesting external network config; on a fresh VM this will just create an
    isolated bridge, NOT join the existing Paperless network, and the pipeline
    will fail to resolve `webserver`. README should state the merge-into-
    paperless-stack requirement explicitly.

M9. Logging: pipeline uses print-based `log()`; quality script logs to
    ~/.cache — divergent observability, no log rotation. Minor.

M10. AnythingLLM doc paging: run_stage_0_ingestion assumes a single
    /documents response; AnythingLLM paginates large workspaces. Also no
    dedupe between AnythingLLM and Paperless sources (different PKs) — same
    doc can be ingested twice.

## 4. Recommended fix order

1. Rotate/move the committed sandbox credentials (C2).
2. Sentinel placement + failed-retry cap (H1, H2).
3. Configurable word-count/OCR thresholds (H3).
4. Retry/backoff + split timeouts on the LM Studio client (H4, H5).
5. Either implement or remove Paperless custom-field writing (M1/H6).
6. Cost work: context truncation, exact version pins (M4, M2, M6).

## 5. Acceptance status for kanban t_8f7c7049

- [x] Legacy model string fully purged from repo (grep clean, excludes .git)
- [x] GEMMA_MODEL default removed (LM Studio unreachable from VPS; fail-fast added)
- [x] .env.example: GEMMA_MODEL= with explicit-set comment
- [x] Fix committed and pushed to origin/main
- [x] This report produced, incl. known issues (timeouts, word-count filter,
      max_tokens, httpx hangs)
- [x] Critical adjacent fix pushed (duplicate PAPERLESS_TOKEN in .env.example)
