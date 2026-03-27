import numpy as np
from openai import OpenAI

# OpenAI client — picks up OPENAI_API_KEY from the environment automatically.
# All embedding and chat calls go through this object.
client = OpenAI()

# ---------------------------------------------------------------------------
# DOCUMENT CORPUS
# These are the raw text records the system knows about — the "knowledge base."
# In a real system this would be loaded from a database, log files, or a vector
# DB. Here they're hard-coded so the data is fully visible while learning.
# See also: incident_reports.py (same strings, stored as named variables for
# use in other scripts).
# ---------------------------------------------------------------------------
documents = [
    "Pump 7 experienced failure during a freeze event on 2026-02-10. Temperature dropped below 28°F. Ice formation blocked intake valve. Manual thaw required.",
    "Pump 3 failed in summer due to overheating. Cooling fan malfunctioned. Internal temperature exceeded threshold.",
    "Pump 7 maintenance log. Winter inspection completed. Anti-freeze system tested and functional."
]


# ---------------------------------------------------------------------------
# EMBEDDING FUNCTION
# Converts a string into a high-dimensional float vector (1536 dims for this
# model). Think of it as a semantic fingerprint: strings with similar *meaning*
# produce vectors that are geometrically close, even if they share no words.
# This is what makes "freeze event" and "ice formation" retrievable by the
# query "cold weather failure."
#
# Cost note: each call makes one API round-trip. In this prototype that's once
# per document at startup, plus once per query at runtime.
# ---------------------------------------------------------------------------
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# QUERY SPECIFICITY GATE — tunable parameters
# These act as an input validation layer before any API calls are made.
# Analogous to a sanity-check block at the top of an ISR: reject bad input
# early rather than letting garbage propagate through the pipeline.
#
# Expand ASSET_IDENTIFIERS and REQUIRED_TERMS as the document corpus grows.
# ---------------------------------------------------------------------------
#****#
REQUIRED_TERMS = ["pump", "valve", "sensor", "motor", "failure", "maintenance"]
MIN_QUERY_WORDS = 4
ASSET_IDENTIFIERS = ["pump 7", "pump 3", "valve 2"]  # expand as your system grows
#****#

# ---------------------------------------------------------------------------
# RESPONSE CONFIDENCE CHECK — tunable parameters
# Phrases that indicate the model is hedging or guessing rather than answering
# from the retrieved context. Used as a post-generation output watchdog.
# ---------------------------------------------------------------------------
#****#
UNCERTAINTY_PHRASES = [
    "i don't have", "i'm not sure", "it's unclear",
    "insufficient information", "cannot determine",
    "it may be", "possibly", "i cannot confirm"
]
#****#


def check_response_confidence(response_text):
    """
    Scan the model's response for uncertainty language.
    Returns a list of matched phrases (empty list = no flags raised).
    This runs *after* generation — it's a read-only check, not a filter.
    """
    lowered = response_text.lower()
    flagged = [p for p in UNCERTAINTY_PHRASES if p in lowered]
    return flagged


def check_query_specificity(query):
    """
    Three-stage input gate. All three must pass or the query is rejected
    before any embedding calls are made:
      1. Minimum word count — blocks single-word and near-empty queries.
      2. Domain term presence — ensures the query is about something the
         system tracks (pumps, valves, sensors, etc.).
      3. Asset identifier — requires a specific named asset (e.g. "Pump 7").
         This prevents "why did the pump fail?" from silently returning
         results that may not apply to the right piece of equipment.
    """
    words = query.strip().split()
    if len(words) < MIN_QUERY_WORDS:
        return False, "Query too short — add more detail."
    has_domain_term = any(term in query.lower() for term in REQUIRED_TERMS)
    if not has_domain_term:
        return False, "Query lacks a domain-specific term (asset, component, or event type)."
    has_asset_id = any(asset in query.lower() for asset in ASSET_IDENTIFIERS)
    if not has_asset_id:
        return False, "Query must reference a specific asset (e.g. Pump 7, Valve 2)."
    return True, None


# ---------------------------------------------------------------------------
# QUERY — swap the active line to explore pipeline behavior with different inputs
#
# Happy path:  "Why did Pump 7 fail during the freeze?"
# Off-topic:   "What is the weather in Kona, Hawaii?"
# Too vague:   "Why did the pump fail?"    (no asset ID — gate rejects)
# Too short:   "What happened?"            (below MIN_QUERY_WORDS — gate rejects)
# ---------------------------------------------------------------------------
#****#
query = "Why did Pump 7 fail during the freeze?"
#query = "What is the weather in Kona, Hawaii?"
#query = "Why did the pump fail?"
#query = "What happened?"
#****#


# ---------------------------------------------------------------------------
# VECTOR STORE — INGESTION (runs at startup)
# Embed every document and store the (text, vector) pair in memory.
# This is the "write" side of what a vector database normally handles.
# In production this step would be pre-computed and persisted; here it runs
# every time so you can inspect the store directly (print vector_store[0]).
# ---------------------------------------------------------------------------
vector_store = []

for doc in documents:
    embedding = get_embedding(doc)
    vector_store.append({
        "text": doc,
        "embedding": embedding
    })

print(f"Query: {query}")
print("documents embedded:", len(vector_store))


# ---------------------------------------------------------------------------
# COSINE SIMILARITY
# Measures the angle between two vectors in high-dimensional space.
# Returns a value in [-1, 1]; 1.0 = identical direction, 0 = orthogonal.
# For embeddings from the same model, semantically similar texts reliably
# score above ~0.75; unrelated texts typically score below 0.5.
#
# Formula: cos(θ) = (A · B) / (||A|| * ||B||)
# This normalizes out vector magnitude so only direction (meaning) matters.
# ---------------------------------------------------------------------------
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# ---------------------------------------------------------------------------
# RETRIEVAL PIPELINE
# Runs only if the query passes the specificity gate above.
# ---------------------------------------------------------------------------
is_specific, reason = check_query_specificity(query)
if not is_specific:
    print(f"\n⚠ AMBIGUOUS QUERY: {reason}")
    print("System response: Please refine your query with a specific asset, component, or event type.")
else:
    # Embed the query using the same model used for documents.
    # The resulting vector lives in the same space as the stored embeddings,
    # making cosine similarity a valid distance metric between them.
    query_embedding = get_embedding(query)

    scored_docs = []

    for item in vector_store:
        score = cosine_similarity(query_embedding, item["embedding"])

        # DETERMINISTIC SCORE BOOST
        # A hand-coded priority modifier layered on top of semantic similarity.
        # Useful when you know certain assets are higher priority or have more
        # reliable documentation. This is an explicit policy decision — it shows
        # up in the scores and can be removed or expanded without touching the
        # embedding logic.
        if "Pump 7" in item["text"]:
#****#
            score += 0.05
#****#
        scored_docs.append((score, item["text"]))

    # Sort highest similarity first — descending by score.
    scored_docs.sort(reverse=True)

    # ---------------------------------------------------------------------------
    # RETRIEVAL FILTER & CONTEXT ASSEMBLY — tunable parameters
    # k:                    how many top candidates to evaluate before filtering
    # MIN_SCORE_THRESHOLD:  comparator threshold — chunks below this are dropped
    # MAX_CONTEXT_CHARS:    hard cap on the context string passed to the LLM
    #                       (proxy for token budget; intentionally small here
    #                        so truncation behavior is easy to observe)
    # ---------------------------------------------------------------------------
#****#
    MAX_CONTEXT_CHARS = 200
    k = 2
    MIN_SCORE_THRESHOLD = 0.60
#****#

    print("\nCandidates retrieved:")
    for score, text in scored_docs[:k]:
        status = "✓ PASSED" if score >= MIN_SCORE_THRESHOLD else "✗ FILTERED"
        print(f"  [{status}] Score: {score:.4f} — {text[:60]}...")

    # Apply threshold filter — only chunks above MIN_SCORE_THRESHOLD proceed.
    top_chunks = [doc for doc in scored_docs[:k] if doc[0] >= MIN_SCORE_THRESHOLD]

    if not top_chunks:
        # No chunk was confident enough — abstain rather than hallucinate.
        # This is preferable to passing low-quality context to the model,
        # which tends to produce confident-sounding but unreliable answers.
        print("\n⚠ ABSTAIN: No retrieved chunks met the minimum confidence threshold.")
        print("System response: Insufficient evidence to answer this query reliably.")
    else:
        # Join the passing chunks into a single context string.
        context_text = "\n\n".join([doc[1] for doc in top_chunks])

        # Hard truncation at MAX_CONTEXT_CHARS — simulates a fixed input buffer.
        # In a real system this would be a token count limit, not a character limit.
        context_text = context_text[:MAX_CONTEXT_CHARS]

        print("\nRetrieved Context:\n")
        print(context_text)

        # ---------------------------------------------------------------------------
        # TRUNCATION WARNING
        # If the context string is at or near the hard cap, the LLM may be working
        # with a mid-sentence cutoff. Flag this so the caller knows the response
        # reliability is reduced. Analogous to detecting a buffer-near-full condition
        # before it becomes a buffer-overflow problem.
        # ---------------------------------------------------------------------------
#****#
        TRUNCATION_WARNING_RATIO = 0.95
#****#
        context_was_truncated = (
            MAX_CONTEXT_CHARS is not None and
            len(context_text) >= MAX_CONTEXT_CHARS * TRUNCATION_WARNING_RATIO
        )

        if context_was_truncated:
            print("\n⚠ TRUNCATION WARNING: Context may have been cut. Response reliability reduced.")

        # ---------------------------------------------------------------------------
        # GENERATIVE STEP — LLM call
        # The model receives only the retrieved context, not the full document corpus.
        # Two system prompts are available (one commented out):
        #   Active:     Permissive — model can draw on general knowledge.
        #   Commented:  Strict — model must answer only from the provided context
        #               and explicitly refuse to extrapolate. Toggleing between them
        #               shows how much the LLM "adds" beyond the retrieved text.
        # ---------------------------------------------------------------------------
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
            #****#
            #   {"role": "system", "content": "Answer strictly using only the provided context. Do not infer beyond it. If the answer is not fully supported by the context, say that the information is insufficient."},
            #****#
                {"role": "system", "content": "You are an industrial telemetry assistant."},
                {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
            ]
        )

        response_text = response.choices[0].message.content
        print("\nModel Response:\n")
        print(response_text)

        # ---------------------------------------------------------------------------
        # OUTPUT WATCHDOG — response confidence check
        # Scan the model's reply for hedging language. If flagged, the response
        # should go to a human for review rather than being acted on directly.
        # This runs after generation and does not modify the response — it's
        # observability, not a filter.
        # ---------------------------------------------------------------------------
        flags = check_response_confidence(response_text)
        if flags:
            print(f"\n⚠ CONFIDENCE FLAG: Model expressed uncertainty — {flags}")
            print("Recommend: human review before acting on this response.")
