import math
from typing import Dict, List

def dcg(rels: List[float]) -> float:
    return sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels))

def ndcg_at_k(retrieved_doc_ids: List[str], qrel_dict: Dict[str, float], k: int = 10) -> float:
    rels = [qrel_dict.get(doc_id, 0.0) for doc_id in retrieved_doc_ids[:k]]
    ideal = sorted(qrel_dict.values(), reverse=True)[:k]
    denom = dcg(ideal)
    return 0.0 if denom == 0.0 else dcg(rels) / denom

def mrr_at_k(retrieved_doc_ids: List[str], qrel_dict: Dict[str, float], k: int = 10) -> float:
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in qrel_dict:
            return 1.0 / rank
    return 0.0

def recall_at_k(retrieved_doc_ids: List[str], qrel_dict: Dict[str, float], k: int = 10) -> float:
    # Binary recall: did we retrieve at least one relevant doc?
    return 1.0 if any(doc_id in qrel_dict for doc_id in retrieved_doc_ids[:k]) else 0.0
