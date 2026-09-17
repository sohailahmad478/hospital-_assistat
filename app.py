import streamlit as st

st.set_page_config(page_title="Hospital Knowledge Assistant", page_icon="🏥", layout="centered")

try:
    import json
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from groq import Groq
except Exception as e:
    st.error("A required package failed to import.")
    st.exception(e)
    st.stop()

# ---- Config ----
EMBEDDINGS_FILE = "embeddings.npy"
METADATA_FILE = "chunks_metadata.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in secrets. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_data():
    embeddings = np.load(EMBEDDINGS_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL)
    return embeddings, chunks, model

try:
    embeddings, chunks, embed_model = load_data()
except Exception as e:
    st.error("Failed to load embeddings or metadata.")
    st.exception(e)
    st.stop()

def get_relevant_chunks(query, k=TOP_K):
    query_vec = embed_model.encode([query], convert_to_numpy=True)[0]
    query_vec = query_vec / np.clip(np.linalg.norm(query_vec), 1e-10, None)
    scores = embeddings @ query_vec
    top_idx = np.argsort(scores)[::-1][:k]
    return [chunks[i] for i in top_idx]

def build_context(selected_chunks):
    blocks = []
    for i, c in enumerate(selected_chunks):
        blocks.append(f"[Source {i+1}: {c['source_file']} | Department: {c['department']}]\n{c['text']}")
    return "\n\n".join(blocks)

def ask_groq(question, context):
    system_prompt = (
        "You are a hospital knowledge assistant. Answer the user's question using ONLY "
        "the information in the provided context. If the answer is not in the context, "
        "say you don't have that information in the knowledge base. Be concise and clear."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content

# ---- UI ----
st.title("🏥 Hospital Knowledge Assistant")
st.caption("Ask a question about hospital policies. Answers are grounded in your knowledge base.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("📄 Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- **{s['source_file']}** ({s['department']})")

question = st.chat_input("Ask a question about hospital policy...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base..."):
            selected = get_relevant_chunks(question)
            context = build_context(selected)

        with st.spinner("Generating answer..."):
            answer = ask_groq(question, context)

        st.markdown(answer)

        sources = []
        seen = set()
        for c in selected:
            key = (c["source_file"], c["department"])
            if key not in seen:
                seen.add(key)
                sources.append({"source_file": c["source_file"], "department": c["department"]})

        with st.expander("📄 Sources"):
            for s in sources:
                st.markdown(f"- **{s['source_file']}** ({s['department']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
