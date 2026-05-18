import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import chromadb
import requests
import os
from dotenv import load_dotenv
import uuid

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f172a, #020617);
    color: white;
}

[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}

h1 {
    color: white;
    text-align: center;
    font-size: 70px !important;
    margin-bottom: 30px;
}

.stTextInput input {
    background-color: #111827;
    color: white !important;
    border-radius: 15px;
    border: 2px solid #7c3aed;
    padding: 14px;
}

.stTextInput input:focus {
    border: 2px solid #a855f7 !important;
    box-shadow: 0 0 15px #9333ea;
}

.answer-box {
    background: rgba(17, 24, 39, 0.9);
    padding: 25px;
    border-radius: 20px;
    border: 1px solid #7c3aed;
    color: white;
    font-size: 18px;
    line-height: 1.8;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

# ChromaDB — persist so reruns don't re-add chunks
chroma_client = chromadb.Client()

vectorizer = TfidfVectorizer()

st.markdown("<h1>📚 AI Study Assistant (RAG)</h1>", unsafe_allow_html=True)

def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

def chunk_text(text, chunk_size=500):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def generate_answer(prompt):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1024
    }
    response = requests.post(url, headers=headers, json=payload)
    try:
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {e} | Raw: {response.text}"

# Use session state to avoid reprocessing on every Streamlit rerun
if "collection" not in st.session_state:
    st.session_state.collection = chroma_client.get_or_create_collection("documents")
if "vectorizer_fitted" not in st.session_state:
    st.session_state.vectorizer_fitted = False
if "chunks" not in st.session_state:
    st.session_state.chunks = []

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file and not st.session_state.vectorizer_fitted:
    with st.spinner("Processing PDF..."):
        text = extract_text(uploaded_file)
        chunks = chunk_text(text)
        st.session_state.chunks = chunks

        embeddings = vectorizer.fit_transform(chunks).toarray()
        st.session_state.vectorizer = vectorizer
        st.session_state.vectorizer_fitted = True

        for i, chunk in enumerate(chunks):
            st.session_state.collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embeddings[i].tolist()],
                documents=[chunk]
            )
    st.success("PDF Processed Successfully!")

if st.session_state.vectorizer_fitted:
    question = st.text_input("Ask a question from the PDF")

    if question:
        q_embedding = st.session_state.vectorizer.transform([question]).toarray()[0]
        results = st.session_state.collection.query(
            query_embeddings=[q_embedding.tolist()],
            n_results=3
        )
        context = "\n".join(results["documents"][0])
        prompt = f"""You are an AI Study Assistant.
Answer the question only using the context below.

Context:
{context}

Question: {question}

Answer:"""

        answer = generate_answer(prompt)
        st.subheader("✨ Answer")

        st.markdown(
            f"""
            <div class="answer-box">
                {answer}
            </div>
            """,
            unsafe_allow_html=True
        )