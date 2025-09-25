# main_live.py
# pocket rag with live Yahoo Finance headlines for NVIDIA and Intel
# why these? because it's topical — NVIDIA invested billions in Intel this week (third week of September 2025)

import numpy as np
import matplotlib.pyplot as plt
import random
from pocket_rag_simplicity_nvidia.rss_utils import get_combined_headlines

# -vectorization
def tokenize(t): return [w.strip(".,()").lower() for w in t.split()]
def vocab_build(texts):
    v = {}
    for t in texts:
        for w in tokenize(t): v.setdefault(w, 0)
    return {w:i for i,w in enumerate(sorted(v.keys()))}

def vec(text, vocab):
    x = np.zeros(len(vocab))
    for w in tokenize(text):
        if w in vocab: x[vocab[w]] += 1.0
    n = np.linalg.norm(x)
    return x / n if n > 0 else x

def cosine(a,b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na > 0 and nb > 0 else 0.0

# retrieval
def build_index(docs):
    vocab = vocab_build([d['text'] for d in docs])
    matrix = [(d['id'], vec(d['text'], vocab)) for d in docs]
    return vocab, matrix

def retrieve(query, vocab, matrix, k=3):
    qv = vec(query, vocab)
    scored = [(doc_id, cosine(qv, dv)) for doc_id, dv in matrix]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]

#  prompt gen
def stitch_prompt(query, retrieved_docs):
    ex = """q: what’s the outlook on chipmakers?
a: recent investment activity and demand shifts suggest strategic positioning.
"""

    ctx_line = "\n".join([f"{cid}:{round(sim,3)}" for cid,sim in retrieved_docs])
    return f"{ex}ctx ids:\n{ctx_line}\nq: {query}\na:"


# fake generation
def generate(prompt, doc_map):
    q = [l for l in prompt.splitlines() if l.startswith("q: ")][-1][3:]
    top_id = prompt.split("ctx ids:\n")[1].split(":")[0].strip()
    base = doc_map.get(top_id, "")
    if any(w in base.lower() for w in q.lower().split()):
        return base
    return "not enough context to answer confidently."

# faithfulness score
def faithfulness(answer, sources):
    texts = [s['text'] for s in sources] + [answer]
    vocab = vocab_build(texts)
    av = vec(answer, vocab)
    sims = [cosine(av, vec(t, vocab)) for t in texts[:-1]]
    return sum(sims)/len(sims) if sims else 0.0, sims

# viz
def plot_sims(ids, sims, out_path="scores_live.png"):
    plt.figure()
    plt.bar(ids, sims)
    plt.title("retrieval similarity")
    plt.ylabel("cosine")
    plt.tight_layout()
    plt.savefig(out_path)
    return out_path

# main
if __name__ == "__main__":
    random.seed(42)
    docs = get_combined_headlines()
    if not docs:
        print("no headlines pulled.")
        exit()

    query = "what should a client know about nvidia’s investment in intel?"

    vocab, index = build_index(docs)
    top = retrieve(query, vocab, index, k=3)
    doc_map = {d['id']: d['text'] for d in docs}
    prompt = stitch_prompt(query, top)
    answer = generate(prompt, doc_map)
    chosen = [d for d in docs if d['id'] in [i for i,_ in top]]
    score, sims = faithfulness(answer, chosen)
    ids = [d['id'] for d in chosen]
    plot_path = plot_sims(ids, sims)

    print("---")
    print("query:", query)
    print("answer:", answer)
    print("faithfulness score:", round(score,3))
    print("ctx ids:", ids)
    print("plot:", plot_path)
