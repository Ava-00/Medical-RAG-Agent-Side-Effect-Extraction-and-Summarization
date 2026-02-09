# 🧠 Medical RAG Agent — LangGraph Workflow on Patient Reviews

Agentic Retrieval-Augmented Generation system for extracting, validating, and synthesizing medication side-effects from real patient reviews using hybrid retrieval and LangGraph orchestration.

---

# 📌 Project Description

This project implements an end-to-end Medical RAG (Retrieval-Augmented Generation) pipeline that transforms unstructured patient drug reviews into clinically grounded, structured side-effect insights with evidence citations.

The system combines:

* Hybrid retrieval (BM25 + dense embeddings)
* Agentic workflow orchestration (LangGraph)
* Structured medical extraction
* Evidence validation
* Citation-grounded answer synthesis

Built to support pharmacovigilance research, adverse event monitoring, and clinical review mining.

---

# 🏗️ Architecture

```
Patient Reviews
      ↓
Semantic Chunking
      ↓
BM25 Index
      ↓
Dense Embeddings
      ↓
Hybrid Retrieval
      ↓
LangGraph Agents
   ├── Extract
   ├── Validate
   └── Answer
```

---

# 📂 Repository Structure

```
medical-rag-agent/
│
├── chunking_module.ipynb
├── embeddings_module.ipynb
├── langgraph_workflow_module.ipynb
│
├── workflow_execution_script.py
├── README.md
├── requirements.txt
│
└── executed_notebooks/
```

---

# 📊 Dataset

Primary dataset:

**UCI Drug Review Dataset (Drugs.com)**
~215k patient reviews containing:

* Drug name
* Condition
* Review text
* Rating
* Usefulness score

Optional evaluation dataset:

**CADEC Corpus** — Annotated adverse drug events.

---

# 🧩 Notebook Modules

## 1️⃣ Chunking Module

`chunking_module.ipynb`

Functions:

* Clean patient reviews
* Semantic chunking
* BM25 corpus creation
* Metadata indexing

Outputs:

```
chunks.parquet
bm25_tokens.pkl
```

---

## 2️⃣ Embeddings Module

`embeddings_module.ipynb`

Functions:

* Load chunked corpus
* Generate dense embeddings
* Normalize vectors
* Save embedding matrix

Outputs:

```
dense_embeddings.npy
embedding_meta.json
```

---

## 3️⃣ LangGraph Workflow Module

`langgraph_workflow_module.ipynb`

Implements agent pipeline:

Nodes:

* Intent routing
* Hybrid retrieval
* Side-effect extraction
* Evidence validation
* Final synthesis

Supports local LLMs such as:

* Phi-3.5-mini-instruct
* TinyLlama
* Qwen2.5

---

# 🔎 Hybrid Retrieval

Combines lexical + semantic signals:

```
Hybrid Score = α · Dense + (1−α) · BM25
```

Benefits:

* Handles noisy patient language
* Captures rare clinical terms
* Improves recall and precision

---

# 🧾 Extraction Schema

```json
{
  "side_effect": "nausea",
  "severity": "moderate",
  "onset": "first week",
  "duration": "3 days",
  "negated": false,
  "evidence": "I felt nauseous after starting",
  "source_id": 18293
}
```

---

# ⚙️ Execution — Google Colab

## Step 1 — Mount Drive

```python
from google.colab import drive
drive.mount("/content/drive")
```

---

## Step 2 — Run Pipeline

```python
!python "/content/drive/MyDrive/<FOLDER>/workflow_execution_script.py" \
  --base "/content/drive/MyDrive/<FOLDER>" \
  --prompt "Does sertraline cause insomnia and how severe is it?"
```

Runner executes:

1. Chunking notebook
2. Embeddings notebook
3. LangGraph workflow (with prompt injection)

Outputs saved in:

```
executed_notebooks/
```

---


Invoke 
```python
DEFAULT_QUERY = "What side effects do people report?"

try:
    QUERY = EXTERNAL_QUERY
except NameError:
    QUERY = DEFAULT_QUERY
```
with:

```python
result = app.invoke({"query": QUERY})
```

# 🧪 Example Queries

* What side effects are reported for sertraline?
* Does metformin cause dizziness?
* Summarize patient experience with Contrave.
* List severe adverse effects mentioned.

---

# 🛠️ Tech Stack

* LangGraph
* LangChain Core
* PyTorch
* Transformers
* Sentence-Transformers
* Rank-BM25
* NumPy / Pandas
* Google Colab

---

# 📈 Use Cases

* Pharmacovigilance research
* Drug safety monitoring
* Clinical review mining
* Adverse event detection
* Healthcare RAG systems

---

# 🚧 Future Work

* Multi-query retrieval
* Ontology normalization
* Frequency estimation
* Temporal side-effect tracking
* Knowledge graph grounding

---

# ✨ Acknowledgements

* UCI Machine Learning Repository
* HuggingFace Transformers
* LangGraph / LangChain
