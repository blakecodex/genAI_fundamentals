# genAI_fundamentals

**Personal projects to sharpen GenAI fundamentals — applied to consulting and finance systems. NOT ACADEMIC. NO PROFESSIONAL CROSSOVER.**

Each folder explores a core GenAI concept through code. Projects use Python (Go coming later), with a bias toward utility, minimalism, and decision-support thinking.

---

### LangChain Finance Agent

A tool-using LangChain agent that summarizes stock trends and business headlines.

- Uses `yfinance` for 7-day stock data
- Pulls news via NewsAPI
- Chains tools using LangChain agent framework
- Optional CUDA note included (for local LLM extensions)

**Key files:** `main.py`, `news_utils.py`  
**Run:**  
```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
python main.py
```

*Example prompt:*  
> “Summarize Microsoft’s stock movement and current news. What might a client want to do next?”

---

### Pocket RAG Simplicity

A minimal retrieval-augmented generation loop — no LLMs, no embeddings. Just NumPy, cosine similarity, and prompt templates.

- Manual tokenization + BoW vectorizer
- Top-k doc retrieval
- Prompt stitching + fake generation
- Faithfulness score using cosine
- Outputs a bar chart of doc similarity

**Key file:** `main.py`  
**Run:** `python main.py`

---

### Pocket RAG: NVIDIA Edition

Topical variant of Pocket RAG.  
Simulates a market briefing using **live RSS headlines from Yahoo Finance** (NVDA + INTC).

- Pulls live news using RSS (fallback included)
- Vectorizes headline text
- Retrieves relevant items
- Generates a consulting-style response
- Visualizes document faithfulness

**Key files:** `main_live.py`, `rss_utils.py`  
**Run:**  
```bash
python main_live.py
```

*Prompt used:*  
> “What should a client know about NVIDIA’s investment in Intel?”

---

### Example Topics Covered

- LangChain agents + tools
- Math + finance + retrieval chaining
- Custom prompt engineering
- Cosine-based ranking + scoring
- Streamlit UI (optional)
- CUDA readiness for local LLMs

---

### Future Goals

- Add Go-based clones of each project
- Add local LLM support via HuggingFace + CUDA
- Expand into multimodal + embedding-based retrieval
- Feel free to contribute via pull requests
