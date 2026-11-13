"""CPU reranker: a transparent linear score over retrieval signals.

score = w_sim * cosine(question, record)
      + w_path * 1/(1 + depth)         if the record is reached through documented links
      + w_type * [record type can carry the requested answer field]
Weights live in config (retrieval.w_*) and are not tuned on test data.
"""
from __future__ import annotations


def rerank(items: list, w_sim: float, w_path: float, w_type: float, answer_types: tuple = ()) -> list:
    for it in items:
        it.score = (w_sim * it.similarity
                    + (w_path / (1 + it.depth) if it.depth is not None else 0.0)
                    + (w_type if it.event.event_type in answer_types else 0.0))
    return sorted(items, key=lambda it: (-it.score, it.event.timestamp or "", it.event.event_id))
