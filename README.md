# Medical-RAG-Agent-Side-Effect-Extraction-and-Summarization
An end-to-end Retrieval-Augmented Generation (RAG) system for extracting, validating, and synthesizing medication side-effects from real patient reviews using hybrid retrieval and agentic workflows.

Built using LangGraph, hybrid BM25 + dense retrieval, and instruction-tuned LLMs running fully locally in Google Colab.

---

## 🚀 Overview

This project implements a modular, agent-orchestrated RAG pipeline that transforms unstructured patient reviews into clinically grounded, structured side-effect insights with evidence citations.

The workflow includes:

* Semantic chunking of patient reviews
* Hybrid retrieval (BM25 + dense embeddings)
* Structured side-effect extraction
* Evidence validation
* Citation-grounded response synthesis

The system is designed for medical review analysis, pharmacovigilance research, and adverse event monitoring.

---

## 🧠 Architecture

Pipeline flow:

```
Patient Reviews
      ↓
Chunking + BM25 Index
      ↓
Dense Embeddings
      ↓
Hybrid Retrieval
      ↓
LangGraph Agent Workflow
      ├── Extract
      ├── Validate
      └── Answer
```

---

## 📂 Project Structure

```
medical-rag-agent/
│
├── chunking_module.ipynb
├── embeddings_module.ipynb
├── langgraph_workflow_module.ipynb
│
├── pipeline_runner.ipynb / .py
├── README.md
│
├── artifacts/
│   ├── chunks.parquet
│   ├── bm25_tokens.pkl
│   └── dense_embeddings.npy
│
└── executed_notebooks/
```

---

## 📊 Data Sources

Primary dataset:

**UCI Drug Review Dataset (Drugs.com)**
~215k real patient reviews with:

* Drug name
* Condition
* Review text
* Rating
* Usefulness votes

Optional evaluation dataset:

**CADEC Corpus** — Annotated adverse drug events.

---

## 🧩 Notebook Modules

### 1️⃣ `chunking_module.ipynb`

* Cleans and standardizes patient reviews
* Performs semantic chunking
* Generates BM25 token corpus
* Saves chunk metadata

Outputs:

```
chunks.parquet
bm25_tokens.pkl
```

---

### 2️⃣ `embeddings_module.ipynb`

* Loads chunked corpus
* Generates dense embeddings
* Uses MiniLM / sentence-transformers encoders
* Saves dense embedding matrix

Outputs:

```
dense_embeddings.npy
embedding_meta.json
```

---

### 3️⃣ `langgraph_workflow_module.ipynb`

Implements the agent pipeline:

Nodes:

* Intent routing (optional)
* Hybrid retrieval
* Side-effect extraction (JSON)
* Evidence validation
* Citation-grounded answer synthesis

Supports instruction-tuned local LLMs such as:

* Phi-3.5-mini-instruct
* TinyLlama
* Qwen2.5

---

## 🔎 Hybrid Retrieval

Retrieval combines:

* BM25 lexical relevance
* Dense semantic similarity

Score fusion:

```
Hybrid Score = α * Dense + (1−α) * BM25
```

Benefits:

* Handles noisy patient language
* Captures rare clinical terms
* Improves recall + precision

---

## 🧾 Structured Extraction Schema

Each extracted side-effect includes:

```json
{
  "side_effect": "nausea",
  "severity": "mild",
  "onset": "first week",
  "duration": "2 days",
  "negated": false,
  "evidence": "I felt nauseous after starting",
  "source_id": 18293
}
```

---

## ⚙️ Execution — Google Colab

Run the full pipeline using the runner notebook/script.

### Step 1 — Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

### Step 2 — Set notebook folder path

```python
BASE = "/content/drive/MyDrive/medical-rag-agent"
```

### Step 3 — Set prompt

```python
PROMPT = "Does sertraline cause insomnia and how severe is it?"
```

### Step 4 — Execute runner

The runner will:

1. Run chunking notebook
2. Run embeddings notebook
3. Inject prompt into LangGraph notebook
4. Execute workflow
5. Save executed notebooks

---

## 🧪 Example Queries

* “What side effects do people report for sertraline?”
* “Does metformin cause dizziness?”
* “Summarize patient experiences with Contrave.”
* “List severe adverse effects mentioned.”

---

## 🛠️ Tech Stack

* LangGraph
* LangChain Core
* PyTorch
* Transformers
* Sentence-Transformers
* Rank-BM25
* NumPy / Pandas
* Google Colab
* HuggingFace Models

---

## 📈 Use Cases

* Pharmacovigilance research
* Adverse event monitoring
* Drug safety signal detection
* Clinical review mining
* Healthcare RAG systems

---

## 🔐 Notes

* Runs fully locally (no external APIs required)
* Embeddings + retrieval cached
* LLM interchangeable
* Supports prompt injection via runner

---

## ✨ Future Work

* Multi-query retrieval
* Knowledge graph grounding
* Temporal side-effect tracking
* Frequency estimation
* Clinical ontology normalization

---

## 📜 License

MIT License
