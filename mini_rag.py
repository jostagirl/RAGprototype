import numpy as np
from openai import OpenAI


client = OpenAI()

documents = [
    "Pump 7 experienced failure during a freeze event on 2026-02-10. Temperature dropped below 28°F. Ice formation blocked intake valve. Manual thaw required.",
    "Pump 3 failed in summer due to overheating. Cooling fan malfunctioned. Internal temperature exceeded threshold.",
    "Pump 7 maintenance log. Winter inspection completed. Anti-freeze system tested and functional."
]

# EMBEDDING FUNCTION- 
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

# QUERY + RETRIEVAL LOGIC
#****#
#query = "Why did Pump 7 fail during the freeze?"
query = "What is the weather in Kona, Hawaii?"
#****#

# VECTOR STORE (INGESTION)
vector_store = []

for doc in documents:
    embedding = get_embedding(doc)
    vector_store.append({
        "text": doc,
        "embedding": embedding
    })

print(f"Query: {query}")
print ("documents embedded:", len(vector_store))

# COSINE SIMILARITY
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

query_embedding = get_embedding(query)

scored_docs = []

for item in vector_store:
    score = cosine_similarity(query_embedding, item["embedding"])

    #Deterministic boost- score modifier
    if "Pump 7" in item["text"]:
#****#
        score += 0.05
#****#    
    scored_docs.append((score, item["text"]))

# Sort highest similarity first
scored_docs.sort(reverse=True)

print("\nTop Matches:")
for score, text in scored_docs:
    print(f"\nScore: {score:.4f}")
    print(text)

# ASSEMBLE CONTEXT AND CALL GENERATIVE MODEL
# Simulate Token Headroom Awareness with maxchars variable
#****#
MAX_CONTEXT_CHARS = 500
#****#
# Select top 2 chunks with minimum confidence threshold
#****#
k = 2
MIN_SCORE_THRESHOLD = 0.60
#****#
top_chunks = [doc for doc in scored_docs[:k] if doc[0] >= MIN_SCORE_THRESHOLD]

if not top_chunks:
    print("\n⚠ ABSTAIN: No retrieved chunks met the minimum confidence threshold.")
    print("System response: Insufficient evidence to answer this query reliably.")
else:
    context_text = "\n\n".join([doc[1] for doc in top_chunks])
    context_text = context_text[:MAX_CONTEXT_CHARS]

    print("\nRetrieved Context:\n")
    print(context_text)

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

    print("\nModel Response:\n")
    print(response.choices[0].message.content)