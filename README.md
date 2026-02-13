# Dense Retrieval on BEIR SciFact: FAISS Flat vs IVF vs HNSW

A small, reproducible benchmark for **dense retrieval** using:

- **Sentence-Transformers** embeddings (default: `sentence-transformers/all-MiniLM-L6-v2`)
- **FAISS** vector search: **Flat (exact)**, **IVF (inverted index)**, **HNSW (graph ANN)**
- BEIR **SciFact** dataset (real queries + qrels)
- Metrics: **nDCG@10**, **MRR@10**, **Recall@10**
- Sweeps + plots for **speed vs quality** tradeoffs

Runs in **Colab** (notebook) or as a **CLI** (Python module).

---

## What this project does

End-to-end dense retrieval pipeline:

1. Download BEIR SciFact (corpus / queries / qrels)
2. Embed documents + queries (normalized vectors)
3. Build FAISS index (Flat / IVF / HNSW)
4. Retrieve top-k docs for each query
5. Evaluate using qrels: nDCG@10, MRR@10, Recall@10
6. Sweep ANN knobs and visualize tradeoffs

---

## Repository layout
```
dense-retrieval-faiss-beir/
├── dense_retrieval_scifact_faiss.ipynb
├── src/
│ ├── benchmark.py
│ ├── indexing.py
│ └── metrics.py
├── assets/
├── requirements.txt
└── README.md
```

---

## Cosine similarity via inner product

Embeddings are **L2-normalized**, so:
```
cosine(a, b) == dot(a, b)
```
That’s why the FAISS indices use **inner product** (IP):

- `IndexFlatIP`
- `METRIC_INNER_PRODUCT`

---

## Quickstart (local)

### Install
```py
pip install -r requirements.txt
```
### Run once (exact vs ANN)
```py
python -m src.benchmark --index flat --k 10
python -m src.benchmark --index ivf --k 10 --nlist 256 --nprobe 16
python -m src.benchmark --index hnsw --k 10 --M 32 --efSearch 64
```
Output prints a metrics dict including:

- ms/query
- recall@k
- mrr@k
- ndcg@k
- n_valid

---

## Sweeps (recommended)

### IVF sweep: nprobe
```py
python -m src.benchmark --sweep ivf --k 10 --nlist 256 --sweep_vals 1 2 4 8 16 32 64 \
 --out assets/ivf_sweep.csv \
 --plot assets/ivf_sweep.png
```
### HNSW sweep: efSearch
```py
python -m src.benchmark --sweep hnsw --k 10 --sweep_vals 8 16 32 64 128 256 \
 --out assets/hnsw_sweep.csv \
 --plot assets/hnsw_sweep.png
```
---

## Parameters you’ll tune

### IVF

- nlist: number of coarse clusters (inverted lists)
- nprobe: number of clusters searched per query

Higher nprobe => better quality, slower queries.

### HNSW

- M: graph connectivity / degree
- efSearch: search breadth at query time

Higher efSearch => better quality, slower queries.

---

## Colab

Open:
```
notebooks/dense_retrieval_scifact_faiss.ipynb
```

It runs end-to-end: download -> embed -> index -> eval -> plots.

If Colab throws import errors around `datasets`, it’s often a local file/folder named `datasets` shadowing Hugging Face. Restarting the runtime after installs also helps.

---

## Extensions

Ways to make this more realistic:

- Use a bigger BEIR dataset (e.g., TREC-COVID / FiQA) to amplify ANN benefits
- Switch to a stronger retrieval model (E5) and add query/passage prefixes
- Add compression (IVF-PQ) for large corpora

---

## License

MIT.
