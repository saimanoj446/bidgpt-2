from flask import Flask, request, jsonify, send_from_directory
import os

# Vector store and embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Groq LLM integration
from langchain_groq import ChatGroq

from langchain.chains import RetrievalQA
from flask_cors import CORS

# Set your Groq API key
os.environ["GROQ_API_KEY"] = "skefv96KlAEOMVf2ur59Uez5H6gXoSjNQzH7uN9G3ZRcQ7IL1vwtPsurckAKIVQlgbIxq0R8f7R8KVDIOcOxHo2sGAsdaDbHqSLoXx5pDFlOaMCXbjJGSdT0p3iyuBSooQJoEUlEej5DIiy13jd3qY0AEj0V89ABbJo5felKyeHP9b6Lngor"
  # <-- Replace with your real Groq API key

# Chroma vector store directory
persist_directory = "chroma"

# Embedding and Vector Store - Using a lighter model
embedding = HuggingFaceEmbeddings(model_name="paraphrase-MiniLM-L3-v2")
vectordb = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

# LLM (Groq Llama 3.1) and Retriever
llm = ChatGroq(
    model="llama-3.1-8b-instant",  # or "llama3-8b-8192" if available
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    groq_api_key=os.environ["GROQ_API_KEY"]
)
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Flask app setup
app = Flask(__name__, static_folder='.')
CORS(app)

@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        user_message = request.json.get("message", "")
        print(f"🔹 User input: {user_message}")
        if not user_message:
            return jsonify({"message": "Empty message"}), 400
        response = qa_chain.run(user_message)
        print(f"✅ RAG response: {response}")
        return jsonify({"message": response})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"message": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
