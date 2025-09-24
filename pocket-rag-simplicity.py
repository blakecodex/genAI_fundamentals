# genai fundamentals — single file, tiny rag + prompt eng + eval + viz
# minimal comments, all lowercase, tiny human typos left in :)

import numpy as np
import matplotlib.pyplot as plt
import random, os

# ---------- data ----------
docs = [
    {"id":"a1","text":"genai uses embeddings, retrieval, and prompts to ground generation."},
    {"id":"b2","text":"retrieval augmented generation (rag) mixes search with a generator."},
    {"id":"c3","text":"calibration and evaluation are important to avoid hallucinations."},
    {"id":"d4","text":"vector spaces let us compare text by cosine similarity."},
    {"id":"e5","text":"prompt engineering with few shot examples steers output quality."},
]

# ---------- tiny embed ----------
def tokenize(t):
    return [w.strip(".,()").lower() for w in t.split()]

def vocab_build(texts):
    v = {}
    for t in texts:
        for w in tokenize(t):
            v.setdefault(w,0)
    return {w:i for i,w in enumerate(sorted(v.keys()))}

def vec(text, vocab):
    x = np.zeros(len(vocab))
    for w in tokenize(text):
        if w in vocab: x[vocab[w]] += 1.0
    n = np.linalg.norm(x)
    return x / n if n>0 else x

def cosine(a,b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na==0 or nb==0: return 0.0
    return float(np.dot(a,b)/(na*nb))

# ---------- rag bits ----------
def build_index(docs):
    vocab = vocab_build([d["text"] for d in docs])
    mat = [(d["id"], vec(d["text"], vocab)) for d in docs]
    return vocab, mat

def retrieve(q, vocab, mat, k=3):
    qv = vec(q, vocab)
    scored = [(i, cosine(qv, dv)) for i, dv in mat]
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]

def stitch_prompt(q, top):
    ex1 = "q: what is rag?\na: it combines retrieval with generation to ground answers."
    ex2 = "q: why eval?\na: to measure quality and reduce hallucinations."
    ctx = "\n".join([f"{cid}:{round(s,3)}" for cid,s in top])
    return (
        "you are a blunt assistant. use the docs by id. if unsure say 'not sure'.\n"
        + ex1 + "\n" + ex2 + f"\nctx ids: {ctx}\nq: {q}\na:"
    )

def generate(prompt, doc_map):
    # fake generator: use top ctx doc if it overlaps the question, else generic
    q = [l for l in prompt.splitlines() if l.startswith("q: ")][-1][3:]
    ids_line = [l for l in prompt.splitlines() if l.startswith("ctx ids: ")][-1][9:]
    top_id = ids_line.split(",")[0].split(":")[0].strip()
    base = doc_map.get(top_id, "")
    if any(w in base.lower() for w in q.lower().split()):
        ans = base
    else:
        ans = "rag uses retrieval + prompts to ground answers; eval helps avoid hallucinations."
    return ans.strip()

# ---------- eval (faithfulness proxy) ----------
def faithfulness(answer, sources):
    texts = [s["text"] for s in sources] + [answer]
    vocab = vocab_build(texts)
    av = vec(answer, vocab)
    sims = [cosine(av, vec(t, vocab)) for t in texts[:-1]]
    score = sum(sims)/len(sims) if sims else 0.0
    return score, sims

# ---------- viz ----------
def bar_scores(ids, sims, out_path):
    plt.figure()
    plt.bar(ids, sims)
    plt.title("similarity to sources")
    plt.xlabel("doc id")
    plt.ylabel("cosine")
    plt.tight_layout()
    plt.savefig(out_path)
    return out_path

# ---------- paig (printed super simple) ----------
def paig():
    return (
        "problem: want quick grounded answers without heavy deps.\n"
        "approach: tiny embeddings -> retrieve top-k -> few-shot prompt -> template generation.\n"
        "impact: fast, reproducible, explainable demo.\n"
        "governance: version files, fixed seeds, show ctx ids, log a plot.\n"
    )

# ---------- main ----------
if __name__ == "__main__":
    random.seed(7)

    q = "explain rag and why eval matters"
    vocab, index = build_index(docs)
    top = retrieve(q, vocab, index, k=3)
    doc_map = {d["id"]: d["text"] for d in docs}
    prompt = stitch_prompt(q, top)
    answer = generate(prompt, doc_map)
    chosen = [d for d in docs if d["id"] in [i for i,_ in top]]
    score, sims = faithfulness(answer, chosen)
    ids = [d["id"] for d in chosen]
    plot_path = bar_scores(ids, sims, "scores.png")

    print(paig())
    print("q:", q)
    print("answer:", answer)
    print("score:", round(score,3))
    print("ctx ids:", ids)
    print("plot:", plot_path)