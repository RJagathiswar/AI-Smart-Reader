from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import fitz, io, os, re, uuid
from docx import Document
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
load_dotenv()

try:
    import openai
    OPENAI_AVAILABLE = True
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
    else:
        OPENAI_AVAILABLE = False
except Exception:
    OPENAI_AVAILABLE = False

app = Flask(__name__, static_folder='frontend/static', template_folder='frontend')
CORS(app)

DOCS = {}

def extract_text_from_pdf_bytes(b: bytes) -> str:
    text = []
    with fitz.open(stream=b, filetype="pdf") as doc:
        for page in doc:
            text.append(page.get_text())
    return "\n".join(text)

def extract_text_from_docx_bytes(b: bytes) -> str:
    with io.BytesIO(b) as bio:
        doc = Document(bio)
        paras = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(paras)

def extract_text_from_txt_bytes(b: bytes) -> str:
    try:
        return b.decode('utf-8', errors='ignore')
    except:
        return b.decode('latin-1', errors='ignore')

def clean_text(t: str) -> str:
    return re.sub(r'\s+', ' ', t).strip()

def split_sentences(text: str):
    sents = re.split(r'(?<=[.!?])\s+', text)
    sents = [s.strip() for s in sents if s.strip()]
    return sents

def chunk_text(text: str, approx_words=150):
    sentences = split_sentences(text)
    chunks = []
    curr = []
    curr_words = 0
    for s in sentences:
        w = len(s.split())
        if curr_words + w <= approx_words:
            curr.append(s); curr_words += w
        else:
            if curr:
                chunks.append(" ".join(curr))
            curr = [s]; curr_words = w
    if curr:
        chunks.append(" ".join(curr))
    chunks = [clean_text(c) for c in chunks if len(c.strip())>20]
    return chunks

def build_index(chunks):
    vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
    tfidf = vect.fit_transform(chunks)
    return vect, tfidf

def retrieve_top_chunks(question, vect, tfidf, chunks, top_k=4):
    q_vec = vect.transform([question])
    sims = cosine_similarity(q_vec, tfidf).flatten()
    idxs = np.argsort(-sims)[:top_k]
    results = []
    for i in idxs:
        results.append({"index": int(i), "score": float(sims[i]), "chunk": chunks[i]})
    return results

def extract_answer_from_chunks(question, top_chunks):
    q_tokens = set(re.findall(r'\w+', question.lower()))
    sentences = []
    for item in top_chunks:
        sents = split_sentences(item["chunk"])
        for s in sents:
            tokens = set(re.findall(r'\w+', s.lower()))
            overlap = len(q_tokens & tokens)
            sentences.append((overlap, s))
    sentences.sort(key=lambda x: -x[0])
    best = [s for score,s in sentences if score>0]
    if not best:
        return top_chunks[0]["chunk"]
    return " ".join(best[:2])

def openai_answer(question, context_chunks):
    if not OPENAI_AVAILABLE:
        return None
    prompt = (
        "You are an assistant. Answer the QUESTION using ONLY the CONTEXT. If the answer is not in the context, say you cannot find it.\n\n"
        "CONTEXT:\n" + "\n---\n".join(context_chunks) + "\n\nQUESTION: " + question + "\n\nAnswer concisely."
    )
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.0
        )
        if isinstance(resp, dict) and 'choices' in resp:
            text = resp["choices"][0]["message"]["content"].strip()
        else:
            text = resp.choices[0].message.content.strip()
        return text
    except Exception:
        return None

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/static/<path:p>')
def static_files(p):
    return send_from_directory('frontend/static', p)

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"ok": False, "message": "No file field"}), 400
    f = request.files['file']; fname = f.filename or "uploaded"; b = f.read(); text = ""
    lower = fname.lower()
    try:
        if lower.endswith('.pdf'):
            text = extract_text_from_pdf_bytes(b)
        elif lower.endswith('.docx'):
            text = extract_text_from_docx_bytes(b)
        elif lower.endswith('.txt'):
            text = extract_text_from_txt_bytes(b)
        else:
            return jsonify({"ok": False, "message": "Unsupported file type. Use PDF, DOCX, or TXT."}), 400
    except Exception as e:
        return jsonify({"ok": False, "message": f"Failed to parse file: {str(e)}"}), 500
    text = clean_text(text)
    if len(text) < 20:
        return jsonify({"ok": False, "message": "Document appears empty or couldn't extract text."}), 400
    chunks = chunk_text(text, approx_words=160)
    if not chunks:
        chunks = [p for p in text.split('\n') if len(p.strip())>30]
    vect, tfidf = build_index(chunks)
    doc_id = str(uuid.uuid4())
    DOCS[doc_id] = {"filename": fname, "chunks": chunks, "vectorizer": vect, "tfidf": tfidf}
    return jsonify({"ok": True, "doc_id": doc_id, "filename": fname, "chunks": len(chunks)}), 200

@app.route('/api/docs', methods=['GET'])
def list_docs():
    items = [{"doc_id": k, "filename": v["filename"], "chunks": len(v["chunks"])} for k,v in DOCS.items()]
    return jsonify({"ok": True, "docs": items})

@app.route('/api/query', methods=['POST'])
def query():
    data = request.get_json() or {}
    doc_id = data.get('doc_id'); question = data.get('question','').strip(); use_ai = bool(data.get('use_ai', False))
    if not doc_id or doc_id not in DOCS:
        return jsonify({"ok": False, "message": "Invalid doc"}), 400
    if not question:
        return jsonify({"ok": False, "message": "Empty question"}), 400
    doc = DOCS[doc_id]; vect = doc['vectorizer']; tfidf = doc['tfidf']; chunks = doc['chunks']
    top = retrieve_top_chunks(question, vect, tfidf, chunks, top_k=5)
    top_texts = [t['chunk'] for t in top]
    answer = None
    if use_ai and OPENAI_AVAILABLE:
        ai = openai_answer(question, top_texts)
        if ai:
            answer = ai
    if not answer:
        answer = extract_answer_from_chunks(question, top)
    return jsonify({"ok": True, "answer": answer, "top_chunks": top}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000)); host = os.environ.get('HOST', '0.0.0.0')
    print(f"Starting app on http://{host}:{port} (OpenAI available: {OPENAI_AVAILABLE})")
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting app on http://{host}:{port} (OpenAI available: {OPENAI_AVAILABLE})")
    app.run(host=host, port=port)


