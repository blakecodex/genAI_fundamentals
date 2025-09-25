# genAI_fundamentals

This repository is dedicated to breaking down the fundamental concepts of Generative AI (genAI) into small, manageable mini-projects. The goal is to provide a hands-on approach to understanding the core techniques, algorithms, and models that power generative AI, with implementations in **Python** (and later **Go**). Each concept is explored through a simple project to reinforce learning and provide practical experience.

## Objectives

- **Learn by Doing**: Each mini-project focuses on a specific genAI concept, making it easier to understand and implement.
- **Progressive Complexity**: Start with simple concepts and build up to more complex models and techniques.
- **Language Flexibility**: Initial projects are built using **Python**, with plans to extend the exploration into **Go**.
- **Conceptual Understanding**: Projects will not only focus on code but also explain the theory and intuition behind each generative model or technique.

## Project Structure

Each folder in this repository corresponds to a mini-project focused on a different fundamental concept of generative AI. Each project includes:

- A description of the concept
- **Python (and eventually Go) code implementation**
- Step-by-step breakdowns and comments in the code
- An explanation of the results and takeaways

### Example Topics

- **Basic Neural Networks for Generation**: Exploring simple feedforward neural networks as a foundation for genAI.
- **Markov Chains**: Modeling sequences with simple probabilistic methods.
- **Variational Autoencoders (VAEs)**: Understanding how VAEs are used for generating new data points similar to a training set.
- **Generative Adversarial Networks (GANs)**: Implementing GANs to learn how two networks (generator and discriminator) can compete to produce realistic data.
- **Reinforcement Learning for Generation**: How agent-based learning methods contribute to generative tasks.
- **Transformers**: Breaking down the transformer architecture for text generation tasks like chatbots or story generation.


## Mini-Project: LangChain Finance Demo

**Personal project to sharpen GenAI fundamentals using LangChain.**

### Problem

How can we use GenAI tooling to summarize real-time financial signals—stock price trends and business headlines—and generate consulting-style recommendations?

### Approach

This project uses:

- Yahoo Finance (via `yfinance`) for stock price trends.
- Real business news from NewsAPI.
- LangChain tools including:
    - Custom tool chaining
    - `LLMMathChain` for calculation
    - Simple memory (buffered)
- Streamlit front-end for user interaction.

### Core Components

- `main.py`: LangChain agent with tool chaining and logic
- `news_utils.py`: Integrates real-time news via NewsAPI
- `app.py`: Streamlit UI to ask questions
- `requirements.txt`: Setup dependencies
- `README.md`: Full framework and run instructions

### Example Prompt

> “Summarize recent movement in MSFT and link to current business news. What should a client look at next?”

### To Run

```bash
pip install -r requirements.txt
export NEWSAPI_KEY=your_newsapi_key_here
streamlit run app.py
```

This project demonstrates how LLMs can interact with structured financial data and live context to support real-world business decisions.


## Installation & Setup

1. Clone the repository:
    ```bash
    git clone https://github.com/your_username/genAI_fundamentals.git
    cd genAI_fundamentals
    ```

2. Install the necessary dependencies (Python version):
    ```bash
    pip install -r requirements.txt
    ```

3. To explore the Go version (coming soon), ensure you have Go installed on your system.

## How to Use

1. Navigate to any project folder to find the code and explanation of the concept.
2. Run the Python scripts to see the concept in action.
3. Read through the explanation to understand the theory behind the code.

## Contributing

Feel free to contribute by:

- Suggesting new concepts to explore.
- Reporting issues or bugs.
- Sharing ideas for improving the implementations.

## Future Goals

- **Expand into Go**: After the initial Python exploration, we'll add Go implementations of the same concepts.
- **Advanced Topics**: Explore more sophisticated generative AI techniques such as diffusion models, neural architecture search, and multimodal generation.

---


## Mini-Project: Pocket RAG Simplicity

**A minimal Retrieval-Augmented Generation (RAG) implementation—built for clarity, not completeness.**

### Problem

How do we demystify RAG by building it from scratch using only core Python and NumPy, without relying on external libraries or embedding APIs?

### Approach

This project focuses on:
- Manual tokenization and bag-of-words vectorization
- Cosine similarity retrieval over a tiny in-memory doc set
- Simple prompt stitching with two few-shot examples
- Template-based response generation
- Lightweight faithfulness scoring using cosine similarity
- Crude visualization using `matplotlib`

### Core Components

- `data.py`: Sample docs
- `embed.py`: Tokenizer and bag-of-words vectorizer
- `rag.py`: Retriever and stitched prompt generator
- `eval.py`: Similarity-based faithfulness score
- `viz.py`: Matplotlib bar chart of similarity
- `main.py`: Full example chaining components

### Example Prompt

> “What is RAG and why does evaluation matter?”

### To Run

```bash
python main.py
```

Outputs:
- Prompt trace
- Generated response
- Faithfulness score
- `scores.png` visualization

This project is intentionally imperfect. Variable names are lowercase, comments are sparse, and structure is modular but minimal—designed to expose the skeleton of RAG for learning.



## Mini-Project: Pocket RAG Simplicity (Live Version)

**This version adds real-time financial context via Yahoo Finance RSS feeds.**

### Why NVIDIA and Intel?

Because it's topical — in the third week of September 2025, NVIDIA invested billions into Intel. This version demonstrates how a basic RAG system can incorporate live, structured text data into a minimal GenAI pipeline.

### Features

- Pulls live news headlines from Yahoo Finance RSS feeds (NVIDIA and Intel)
- Performs cosine-based document retrieval over headline vectors
- Stitches a prompt and simulates response generation
- Calculates a faithfulness score and plots similarity to top documents
- Encapsulated in `main_live.py` and `rss_utils.py`

### Example Prompt

> “What should a client know about NVIDIA’s investment in Intel?”

### Output

- Answer (simulated, grounded)
- Retrieval trace (IDs and similarity scores)
- `scores_live.png` visualization


## Mini-Project: Pocket RAG Simplicity — NVIDIA Edition

**Script:** `pocket-rag_simplicity_nvidia.py`  
**Objective:** Demonstrate how a minimal retrieval-augmented generation (RAG) loop can be applied to real financial headlines to support decision-making in a consulting context.

### Context

In the third week of September 2025, NVIDIA invested billions in Intel. This mini-project simulates how a client-facing analyst might combine real-time news with retrieval and summarization logic to produce fast, grounded insights using only NumPy and standard Python.

### Problem

Can we extract relevant signals from live business headlines and return a consulting-style response about a market event—without full LLM infrastructure?

### Approach

- Headlines are pulled from Yahoo Finance RSS feeds (NVIDIA and Intel)
- Text is vectorized using bag-of-words + cosine similarity
- Top-k context headlines are retrieved and fed into a prompt template
- A basic response is generated based on context match
- Faithfulness score is computed and visualized with `matplotlib`

### Key Components

- `rss_utils.py`: Pulls and parses RSS headlines for NVDA and INTC
- `pocket-rag_simplicity_nvidia.py`: Full RAG script with prompt stitching, response generation, and evaluation
- `scores_live.png`: Visual output showing context similarity

### Example Prompt

> “What should a client know about NVIDIA’s investment in Intel?”

### Output

- Simulated grounded response (from retrieved context)
- Cosine similarity plot showing relevance of source docs
- Average faithfulness score (proxy for grounding)

This project demonstrates how retrieval logic and systems thinking can be applied even with minimal tooling—supporting fast decision workflows in finance and consulting environments.

