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

# VECTOR STORE (INGESTION)
vector_store = []

for doc in documents:
    embedding = get_embedding(doc)
    vector_store.append({
        "text": doc,
        "embedding": embedding
    })

print ("documents embedded:", len(vector_store))

# COSINE SIMILARITY
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# QUERY + RETRIEVAL LOGIC
query = "Why did Pump 7 fail during the freeze?"

query_embedding = get_embedding(query)

scored_docs = []

for item in vector_store:
    score = cosine_similarity(query_embedding, item["embedding"])
    scored_docs.append((score, item["text"]))

# Sort highest similarity first
scored_docs.sort(reverse=True)

print("\nTop Matches:")
for score, text in scored_docs:
    print(f"\nScore: {score:.4f}")
    print(text)

# ASSEMBLE CONTEXT AND CALL GENERATIVE MODEL

# Select top 2 chunks
top_context = [doc for _, doc in scored_docs[:2]]

context_text = "\n\n".join(top_context)

print("\nRetrieved Context:\n")
print(context_text)

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are an industrial telemetry assistant."},
        {"role": "user", "content": query}
    ]
)

print("\nModel Response WITHOUT RAG:\n")
print(response.choices[0].message.content)