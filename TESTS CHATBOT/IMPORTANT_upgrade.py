import os
import json
import faiss
import numpy as np
import gradio as gr
from sklearn.preprocessing import normalize
from openai import OpenAI

# =========================
# CONFIGURATION
# =========================

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBED_MODEL = "text-embedding-ada-002"
CHAT_MODEL = "gpt-3.5-turbo"

TOP_K = 4
SUMMARY_TRIGGER_TURNS = 8

# =========================
# LOAD DATA
# =========================

with open('json/library_data.json', 'r', encoding='utf-8') as f:
    library_data = json.load(f)

entries = [entry for entry in library_data]
texts = [entry['text'] for entry in entries]
metadata_mapping = {i: entry for i, entry in enumerate(entries)}

# =========================
# EMBEDDINGS
# =========================

def generate_embeddings(texts):
    try:
        resp = client.embeddings.create(
            model=EMBED_MODEL,
            input=texts
        )
        return [d.embedding for d in resp.data]
    except Exception as e:
        print("Embedding error:", e)
        return np.zeros((len(texts), 1536)).tolist()

combined_embeddings = generate_embeddings(texts)
combined_embeddings = normalize(combined_embeddings)
embedding_array = np.array(combined_embeddings, dtype=np.float32)

dimension = embedding_array.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embedding_array)

# =========================
# RETRIEVAL
# =========================

def retrieve_information(query, k=TOP_K):

    query_emb = generate_embeddings([query])
    query_emb = normalize(query_emb)
    query_emb = np.array(query_emb, dtype=np.float32)

    _, indices = faiss_index.search(query_emb, k=k)

    results = []

    for idx in indices[0]:
        item = metadata_mapping.get(idx)
        if not item:
            continue

        if item['type'] == 'faq':
            results.append(f"FAQ: {item['answer']}")

        elif item['type'] == 'book':
            results.append(
                f"BOOK: {item['title']} — Status: {item['status']}"
            )

    return "\n".join(results)

# =========================
# QUERY REWRITING
# =========================

def rewrite_query(query, chat_history):

    messages = [
        {
            "role": "system",
            "content": (
                "Rewrite the latest user question into a standalone "
                "library search query. Resolve references and pronouns."
            )
        },
        *chat_history,
        {"role": "user", "content": query}
    ]

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0,
        max_tokens=80
    )

    return resp.choices[0].message.content.strip()

# =========================
# MEMORY SUMMARIZATION
# =========================

def summarize_history(chat_history):

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content":
                "Summarize factual conversation state for memory retention."
            },
            *chat_history
        ],
        temperature=0,
        max_tokens=150
    )

    return resp.choices[0].message.content.strip()

# =========================
# MESSAGE BUILDING
# =========================

SYSTEM_PROMPT = """
You are Librito, an academic library assistant chatbot.

Rules:
- Use retrieved library data as primary grounding source.
- Do not invent book availability or policy details.
- If data is missing, state that explicitly.
- Prefer precise, factual answers.
- Keep answers concise but complete.
"""

def build_messages(query, chat_history, retrieved_block, memory_summary=None):

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if memory_summary:
        messages.append({
            "role": "system",
            "content": f"Conversation memory summary:\n{memory_summary}"
        })

    messages.append({
        "role": "system",
        "content": f"Retrieved library data:\n{retrieved_block}"
    })

    messages.extend(chat_history)
    messages.append({"role": "user", "content": query})

    return messages

# =========================
# MAIN RESPONSE PIPELINE
# =========================

def get_gpt_response(query, chat_history):

    # Step 1 — rewrite query using conversation context
    standalone_query = rewrite_query(query, chat_history)

    # Step 2 — retrieve context with rewritten query
    retrieved_block = retrieve_information(standalone_query, TOP_K)

    # Step 3 — optional rolling memory compression
    memory_summary = None
    if len(chat_history) >= SUMMARY_TRIGGER_TURNS:
        memory_summary = summarize_history(chat_history[-SUMMARY_TRIGGER_TURNS:])

    # Step 4 — build grounded prompt
    messages = build_messages(
        query,
        chat_history,
        retrieved_block,
        memory_summary
    )

    # Step 5 — grounded generation
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=300
    )

    return resp.choices[0].message.content.strip()

# =========================
# GRADIO INTERFACE
# =========================

chatbot = gr.ChatInterface(
    fn=get_gpt_response,
    title="Vistula University Library — Conversational RAG Chatbot",
    description=(
        "Context-aware academic library assistant using FAISS retrieval "
        "and grounded language generation."
    ),
    chatbot=gr.Chatbot(type="messages"),
    textbox=gr.Textbox(placeholder="Ask about books, availability, or library services"),
    submit_btn="Ask",
    stop_btn="Stop",
    theme="default"
)

# =========================
# LAUNCH
# =========================

if __name__ == "__main__":
    chatbot.launch(share=True)