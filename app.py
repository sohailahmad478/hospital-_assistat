import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---- Config ----
INDEX_DIR = "faiss_index"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 4

st.set_page_config(page_title="Hospital Knowledge Assistant", page_icon="🏥", layout="centered")

# ---- Load API key from Streamlit secrets ----
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found in secrets. Add it in Streamlit Cloud → Settings → Secrets.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_index_and_model():
    index = faiss.read_index(f"{INDEX_DIR}/index.faiss")
    with open(f"{INDEX_DIR}/index.pkl", "rb") as f:
        docstore, index_to_docstore_id = pickle.load(f)
    model = SentenceTransformer(EMBEDDING_MODEL)
    return index, docstore, index_to_docstore_id, model

index, docstore, index_to_docstore_id, embed_model = load_index_and_model()

def get_relevant_chunks(query, k=TOP_K):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        doc_id = index_to_docstore_id[idx]
        doc = docstore.search(doc_id)
        results.append(doc)
    return results

def build_context(chunks):
    context_blocks = []
    for i, c in enumerate(chunks):
        dept = c.metadata.get("department", "unknown")
        src = c.metadata.get("source_file", "unknown")
        context_blocks.append(f"[Source {i+1}: {src} | Department: {dept}]\n{c.page_content}")
    return "\n\n".join(context_blocks)

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
            chunks = get_relevant_chunks(question)
            context = build_context(chunks)

        with st.spinner("Generating answer..."):
            answer = ask_groq(question, context)

        st.markdown(answer)

        sources = []
        seen = set()
        for c in chunks:
            src = c.metadata.get("source_file", "unknown")
            dept = c.metadata.get("department", "unknown")
            key = (src, dept)
            if key not in seen:
                seen.add(key)
                sources.append({"source_file": src, "department": dept})

        with st.expander("📄 Sources"):
            for s in sources:
                st.markdown(f"- **{s['source_file']}** ({s['department']})")

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
