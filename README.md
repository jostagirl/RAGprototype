# RAG Prototype — Industrial Telemetry Q&A

A from-scratch, intentionally minimal implementation of **Retrieval-Augmented Generation (RAG)** applied to a small set of industrial equipment incident reports. This is a learning tool — the goal is clarity over completeness. Every design decision is visible and tweakable so the pipeline behavior is easy to observe and understand as it grows.

---

## What Is RAG?

Large language models (LLMs) like GPT-4 are trained on general internet text. They have no knowledge of your pump logs, your sensor history, or your facility. RAG is the standard pattern for grounding an LLM in *your* data without retraining it.

The pipeline has three stages — if you think in terms of firmware, it maps pretty cleanly:

| RAG Stage | Firmware Analogy |
|---|---|
| **Embed documents** — convert text into high-dimensional numeric vectors | Computing a semantic "fingerprint" for each record; like a CRC but over meaning, not bytes |
| **Retrieve** — find which stored vectors are closest to the query vector | Content-addressable lookup; nearest-neighbor search using cosine similarity as the distance metric |
| **Generate** — pass only the retrieved excerpts to the LLM as context | The LLM is the inference engine; you're writing its input register before the call |

The key insight: the LLM never sees the full document corpus. It only sees what the retrieval stage decided was relevant. This keeps responses grounded and token usage bounded.

---

## Repo Structure

```
mini_rag.py          # Main pipeline: embed, retrieve, generate, guard
incident_reports.py  # The document corpus (knowledge base / "flash" data)
```

---

## Pipeline Walkthrough

```
[incident_reports.py]
       │
       ▼
  Embed all documents → vector_store[]      (ingestion, runs once at startup)
       │
       ▼
  Query specificity gate                    (input validation — rejects vague queries early)
       │
       ▼
  Embed query → compare against vector_store via cosine similarity
       │
       ▼
  Deterministic score boost (Pump 7)        (hand-tuned priority modifier)
       │
       ▼
  Filter by MIN_SCORE_THRESHOLD             (comparator — drops low-confidence chunks)
       │
       ▼
  Assemble context buffer (MAX_CONTEXT_CHARS)
       │
       ▼
  Truncation check                          (overflow detection on the context buffer)
       │
       ▼
  LLM call (GPT-4o-mini)                   (generative inference)
       │
       ▼
  Response confidence scan                  (output watchdog — flags model uncertainty)
```

---

## Tunable Parameters

All hot parameters are marked with `#****#` blocks in the code for easy scanning.

| Parameter | Location | Effect |
|---|---|---|
| `REQUIRED_TERMS` | query gate | Domain vocabulary that must appear in the query |
| `ASSET_IDENTIFIERS` | query gate | Specific assets the system knows about; expand as you add documents |
| `MIN_QUERY_WORDS` | query gate | Minimum query length; rejects one-word or vague inputs |
| `MIN_SCORE_THRESHOLD` | retrieval filter | Cosine similarity cutoff; raise to be more selective, lower to cast wider |
| `k` | retrieval | How many top candidates to consider before filtering |
| `MAX_CONTEXT_CHARS` | context assembly | Hard cap on context passed to the LLM (proxy for token budget) |
| `TRUNCATION_WARNING_RATIO` | truncation check | Fraction of `MAX_CONTEXT_CHARS` that triggers the overflow warning |
| `UNCERTAINTY_PHRASES` | confidence check | Strings in the model response that flag low-confidence output |

---

## Setup

**Requirements:** Python 3.9+, an OpenAI API key.

```bash
pip install openai numpy
export OPENAI_API_KEY=your_key_here
python mini_rag.py
```

The script embeds the documents in `incident_reports.py` at startup (one API call per document), then runs the full pipeline against the `query` variable defined near the top of `mini_rag.py`.

To explore behavior, swap the active `query = ...` line. Several example queries are commented out covering: the happy path, an off-topic question, an ambiguous/vague query, and a query that is on-topic but not specific enough. Watching how the pipeline responds to each is the main learning exercise.

---

## Learning Notes

These are the deliberate simplifications in this prototype and what they're stand-ins for:

- **No vector database.** The vector store is a plain Python list in memory. This makes the data structure fully transparent — you can print it, inspect individual embeddings, step through the similarity loop. A real system would use a vector DB (Pinecone, pgvector, Chroma, etc.), but that's an infrastructure concern, not a conceptual one.
- **Embeddings via OpenAI.** `text-embedding-3-small` converts text into 1536-dimensional float vectors. The embedding space is what makes semantic similarity possible — "freeze event" and "ice formation" land geometrically close even though they share no words. Worth printing a raw embedding once just to see what you're working with.
- **Deterministic score boost.** The `+0.05` bump for Pump 7 documents is a hand-coded priority rule layered on top of the similarity score. It shows that you can inject domain knowledge into the retrieval step without changing the model — useful for understanding where the system's "opinions" come from.
- **Strict system prompt (commented out).** There's an alternate system prompt that tells the model to answer *only* from context and refuse to extrapolate. The active prompt is more permissive. Toggling between them is a good way to see how much the LLM "adds" beyond what the retrieved context actually says.
- **`MAX_CONTEXT_CHARS` is intentionally tiny.** At 200 characters the context buffer will almost always truncate. This is for observability — set it to something large (e.g. 4000) once you've seen the truncation warning fire, and see how the response quality changes.
