"""
Intelligent Query Assistant - simple base version
-------------------------------------------------
1. Keeps a small knowledge base of question/answer pairs.
2. Finds the most relevant entry for a user's query using TF-IDF + cosine similarity.
3. (Optional) Asks Claude to phrase a natural answer from the matched entries,
   if the ANTHROPIC_API_KEY environment variable is set.

Run:  python app.py   ->  open http://127.0.0.1:5000
"""
import os

from flask import Flask, jsonify, render_template_string, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 1. Knowledge base  (replace / extend with your own data)
# ---------------------------------------------------------------------------
KNOWLEDGE_BASE = [
    {"q": "What is machine learning?",
     "a": "Machine learning is a branch of AI where systems learn patterns from data instead of being explicitly programmed."},
    {"q": "What is natural language processing?",
     "a": "NLP is the field of AI that helps computers understand, interpret and generate human language."},
    {"q": "What is a database?",
     "a": "A database is an organized collection of data that can be easily stored, accessed and managed."},
    {"q": "What is SQL?",
     "a": "SQL (Structured Query Language) is used to create, read, update and delete data in relational databases."},
    {"q": "What is Python?",
     "a": "Python is a high-level, beginner-friendly programming language widely used in web, data science and AI."},
    {"q": "What is an API?",
     "a": "An API (Application Programming Interface) lets different software programs communicate with each other."},
    {"q": "What is cloud computing?",
     "a": "Cloud computing delivers computing resources such as servers, storage and databases over the internet."},
    {"q": "What is cybersecurity?",
     "a": "Cybersecurity is the practice of protecting systems, networks and data from digital attacks."},
]

# ---------------------------------------------------------------------------
# 2. Retrieval engine
# ---------------------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
# Index question + answer text so queries can match either one
doc_matrix = vectorizer.fit_transform(f'{d["q"]} {d["a"]}' for d in KNOWLEDGE_BASE)

SIMILARITY_THRESHOLD = 0.10  # below this we say "I don't know"


def search(query, top_k=3):
    """Return the top_k knowledge-base entries most similar to the query."""
    scores = cosine_similarity(vectorizer.transform([query]), doc_matrix).ravel()
    ranked = scores.argsort()[::-1][:top_k]
    return [(KNOWLEDGE_BASE[i], float(scores[i])) for i in ranked if scores[i] >= SIMILARITY_THRESHOLD]


def ask_claude(query, matches):
    """Optional: let Claude write a natural answer using only the matched context."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic  # imported lazily so the app runs without it
        client = anthropic.Anthropic(api_key=api_key)
        context = "\n".join(f"- {m['q']} {m['a']}" for m, _ in matches)
        reply = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=400,
            system="Answer the user's question using ONLY the provided context. Be concise.",
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}],
        )
        return reply.content[0].text
    except Exception as exc:  # fall back to plain retrieval on any failure
        print("Claude call failed:", exc)
        return None


# ---------------------------------------------------------------------------
# 3. Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template_string(PAGE)


@app.route("/ask", methods=["POST"])
def ask():
    query = (request.get_json(silent=True) or {}).get("query", "").strip()
    if not query:
        return jsonify(answer="Please type a question.", confidence=0)

    matches = search(query)
    if not matches:
        return jsonify(answer="Sorry, I couldn't find anything relevant. Try rephrasing your question.",
                       confidence=0)

    best, score = matches[0]
    answer = ask_claude(query, matches) or best["a"]
    return jsonify(answer=answer, matched_question=best["q"], confidence=round(score * 100))


# ---------------------------------------------------------------------------
# 4. Frontend (single page)
# ---------------------------------------------------------------------------
PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intelligent Query Assistant</title>
<style>
  body{font-family:system-ui,sans-serif;background:#f3f4f6;margin:0;display:flex;justify-content:center}
  .box{width:100%;max-width:640px;margin:24px;background:#fff;border-radius:12px;box-shadow:0 2px 10px #0002;display:flex;flex-direction:column;height:85vh}
  h1{font-size:1.2rem;margin:0;padding:16px;border-bottom:1px solid #e5e7eb}
  #chat{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
  .msg{padding:10px 14px;border-radius:12px;max-width:80%;line-height:1.4}
  .user{background:#2563eb;color:#fff;align-self:flex-end}
  .bot{background:#e5e7eb;align-self:flex-start}
  .meta{font-size:.75rem;color:#6b7280;margin-top:4px}
  form{display:flex;gap:8px;padding:12px;border-top:1px solid #e5e7eb}
  input{flex:1;padding:10px;border:1px solid #d1d5db;border-radius:8px;font-size:1rem}
  button{padding:10px 18px;border:0;border-radius:8px;background:#2563eb;color:#fff;font-size:1rem;cursor:pointer}
</style>
</head>
<body>
<div class="box">
  <h1>🤖 Intelligent Query Assistant</h1>
  <div id="chat"><div class="msg bot">Hi! Ask me about machine learning, NLP, databases, SQL, Python, APIs, cloud or cybersecurity.</div></div>
  <form id="f">
    <input id="q" placeholder="Type your question..." autocomplete="off" autofocus>
    <button>Send</button>
  </form>
</div>
<script>
const chat = document.getElementById('chat'), f = document.getElementById('f'), q = document.getElementById('q');
function add(text, cls, meta){
  const d = document.createElement('div'); d.className = 'msg ' + cls; d.textContent = text;
  if(meta){ const m = document.createElement('div'); m.className='meta'; m.textContent = meta; d.appendChild(m); }
  chat.appendChild(d); chat.scrollTop = chat.scrollHeight;
}
f.addEventListener('submit', async e => {
  e.preventDefault();
  const text = q.value.trim(); if(!text) return;
  add(text, 'user'); q.value = '';
  try {
    const r = await fetch('/ask', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query:text})});
    const d = await r.json();
    add(d.answer, 'bot', d.matched_question ? `Matched: "${d.matched_question}" · Confidence ${d.confidence}%` : '');
  } catch { add('Server error. Is the app running?', 'bot'); }
});
</script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(debug=True)
