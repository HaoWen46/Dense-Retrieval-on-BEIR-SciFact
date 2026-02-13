import argparse
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from beir import util
from beir.datasets.data_loader import GenericDataLoader

from .indexing import IndexConfig, build_faiss_index
from .metrics import ndcg_at_k, mrr_at_k, recall_at_k

def load_beir_scifact() -> Tuple[Dict, Dict, Dict, str]:
    dataset = "scifact"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
    data_path = util.download_and_unzip(url, dataset)
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")
    return corpus, queries, qrels, data_path

def prepare_texts(corpus: Dict, queries: Dict):
    doc_ids = list(corpus.keys())
    docs = [(corpus[d].get("title", "") + " " + corpus[d]["text"]).strip() for d in doc_ids]
    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]
    return doc_ids, docs, query_ids, query_texts

def embed(model: SentenceTransformer, texts: List[str], batch_size: int = 128) -> np.ndarray:
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    return emb.astype("float32")

def evaluate_index(index, q_emb: np.ndarray, query_ids: List[str], doc_ids: List[str], qrels: Dict, k: int):
    # Search timing
    t0 = time.perf_counter()
    scores, ids = index.search(q_emb, k)
    t1 = time.perf_counter()
    ms_per_query = (t1 - t0) * 1000 / len(query_ids)

    # Metrics
    ndcg = 0.0
    mrr = 0.0
    rec = 0.0
    valid = 0

    for row, qid in enumerate(query_ids):
        rel = qrels.get(qid, {})
        if not rel:
            continue

        retrieved_doc_ids = [doc_ids[i] for i in ids[row].tolist()]
        valid += 1

        ndcg += ndcg_at_k(retrieved_doc_ids, rel, k=k)
        mrr += mrr_at_k(retrieved_doc_ids, rel, k=k)
        rec += recall_at_k(retrieved_doc_ids, rel, k=k)

    if valid == 0:
        return {
            "n_valid": 0,
            "ms/query": ms_per_query,
            "recall@k": 0.0,
            "mrr@k": 0.0,
            "ndcg@k": 0.0
        }

    return {
        "n_valid": valid,
        "ms/query": ms_per_query,
        "recall@k": rec / valid,
        "mrr@k": mrr / valid,
        "ndcg@k": ndcg / valid
    }

def run_once(args):
    corpus, queries, qrels, data_path = load_beir_scifact()
    doc_ids, docs, query_ids, query_texts = prepare_texts(corpus, queries)

    model = SentenceTransformer(args.model)

    print(f"Dataset path: {data_path}")
    print(f"Docs={len(docs)} Queries={len(query_texts)}")
    print(f"Model={args.model}")

    t0 = time.perf_counter()
    doc_emb = embed(model, docs, batch_size=args.batch_size)
    t1 = time.perf_counter()
    q_emb = embed(model, query_texts, batch_size=args.batch_size)
    t2 = time.perf_counter()

    print(f"Embed docs: {doc_emb.shape} in {t1-t0:.2f}s")
    print(f"Embed queries: {q_emb.shape} in {t2-t1:.2f}s")

    cfg = IndexConfig(
        kind=args.index,
        nlist=args.nlist,
        nprobe=args.nprobe,
        M=args.M,
        efConstruction=args.efConstruction,
        efSearch=args.efSearch
    )

    t3 = time.perf_counter()
    index = build_faiss_index(doc_emb, cfg)
    t4 = time.perf_counter()
    print(f"Index build ({cfg.kind}): {t4-t3:.2f}s")

    metrics = evaluate_index(index, q_emb, query_ids, doc_ids, qrels, k=args.k)
    print("Metrics:", metrics)
    return metrics

def sweep_ivf(args):
    corpus, queries, qrels, data_path = load_beir_scifact()
    doc_ids, docs, query_ids, query_texts = prepare_texts(corpus, queries)

    model = SentenceTransformer(args.model)
    doc_emb = embed(model, docs, batch_size=args.batch_size)
    q_emb = embed(model, query_texts, batch_size=args.batch_size)

    cfg = IndexConfig(kind="ivf", nlist=args.nlist, nprobe=1)
    index = build_faiss_index(doc_emb, cfg)

    rows = []
    for nprobe in args.sweep_vals:
        index.nprobe = int(nprobe)
        m = evaluate_index(index, q_emb, query_ids, doc_ids, qrels, k=args.k)
        rows.append({"nprobe": int(nprobe), **m})

    df = pd.DataFrame(rows)
    print(df)

    if args.out:
        df.to_csv(args.out, index=False)
        print("Wrote:", args.out)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(df["nprobe"], df["ndcg@k"], marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("nprobe (log2)")
        plt.ylabel(f"nDCG@{args.k}")
        plt.title("IVF sweep: quality vs nprobe")
        plt.grid(True, alpha=0.3)
        plt.savefig(args.plot, dpi=200, bbox_inches="tight")
        print("Saved plot:", args.plot)

def sweep_hnsw(args):
    corpus, queries, qrels, data_path = load_beir_scifact()
    doc_ids, docs, query_ids, query_texts = prepare_texts(corpus, queries)

    model = SentenceTransformer(args.model)
    doc_emb = embed(model, docs, batch_size=args.batch_size)
    q_emb = embed(model, query_texts, batch_size=args.batch_size)

    cfg = IndexConfig(kind="hnsw", M=args.M, efConstruction=args.efConstruction, efSearch=8)
    index = build_faiss_index(doc_emb, cfg)

    rows = []
    for ef in args.sweep_vals:
        index.hnsw.efSearch = int(ef)
        m = evaluate_index(index, q_emb, query_ids, doc_ids, qrels, k=args.k)
        rows.append({"efSearch": int(ef), **m})

    df = pd.DataFrame(rows)
    print(df)

    if args.out:
        df.to_csv(args.out, index=False)
        print("Wrote:", args.out)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(df["efSearch"], df["ndcg@k"], marker="o")
        plt.xscale("log", base=2)
        plt.xlabel("efSearch (log2)")
        plt.ylabel(f"nDCG@{args.k}")
        plt.title("HNSW sweep: quality vs efSearch")
        plt.grid(True, alpha=0.3)
        plt.savefig(args.plot, dpi=200, bbox_inches="tight")
        print("Saved plot:", args.plot)

def parse_args():
    p = argparse.ArgumentParser(description="Dense retrieval benchmark on BEIR SciFact using FAISS.")
    p.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--k", type=int, default=10)

    # one-shot index config
    p.add_argument("--index", type=str, choices=["flat", "ivf", "hnsw"], default="flat")
    p.add_argument("--nlist", type=int, default=256)
    p.add_argument("--nprobe", type=int, default=16)
    p.add_argument("--M", type=int, default=32)
    p.add_argument("--efConstruction", type=int, default=200)
    p.add_argument("--efSearch", type=int, default=64)

    # sweep mode
    p.add_argument("--sweep", type=str, choices=["ivf", "hnsw"], default=None,
                   help="Run a sweep over IVF nprobe or HNSW efSearch.")
    p.add_argument("--sweep_vals", type=int, nargs="+",
                   default=[1,2,4,8,16,32,64],
                   help="Values to sweep (nprobe or efSearch depending on mode).")
    p.add_argument("--out", type=str, default=None, help="CSV output path.")
    p.add_argument("--plot", type=str, default=None, help="PNG plot output path.")

    return p.parse_args()

def main():
    args = parse_args()

    if args.sweep == "ivf":
        sweep_ivf(args)
        return
    if args.sweep == "hnsw":
        sweep_hnsw(args)
        return

    run_once(args)

if __name__ == "__main__":
    main()
