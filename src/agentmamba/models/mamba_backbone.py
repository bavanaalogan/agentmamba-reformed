"""Frozen ``state-spaces/mamba-130m-hf`` backbone (CPU, reference PyTorch scan path).

Inference-only uses:
1. ``record_features``: per-record surprisal (mean / max token NLL) and mean-pooled final
   hidden state, used as OPTIONAL gate features (gate ablation G vs H).
2. ``score_candidates``: length-normalized log-likelihood of each candidate answer string
   continuing a records+question prompt. This is the candidate-ranking answer mechanism used
   by the staged benchmark.
3. ``generate``: optional greedy free-form generation from a records+question prompt, used by
   the interactive raw-baseline comparison.

Measured limitation (artifacts/models/mamba_probe.json): the base model is not instruction-tuned
and its free generation does not reliably answer record questions. Generation is exposed for
the requested raw baseline, not treated as an answer-quality guarantee.

Efficiency: the prompt is processed ONCE; each candidate is then decoded token by token from
a copy of the recurrent cache. (Feeding a multi-token chunk into the cached path is not
supported by transformers 5.20.2's Mamba decode step and gave wrong log-probs; token-by-token
decoding matched a full forward pass to within 3e-5 per token, and a model test checks this
against ``score_candidates_full``.)

Scores are memoized in a SQLite key-value cache so repeated experiment runs are cheap.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np


class ScoreCache:
    def __init__(self, path: Path | None):
        self._lock = threading.Lock()
        self._conn = None
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            # several experiment processes may share this cache file
            self._conn = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
            self._conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")
            self._conn.commit()

    def get(self, key: str):
        if self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute("SELECT v FROM kv WHERE k=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value) -> None:
        if self._conn is None:
            return
        with self._lock:
            self._conn.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))
            self._conn.commit()


class MambaBackbone:
    def __init__(self, model_id: str = "state-spaces/mamba-130m-hf", threads: int = 8,
                 max_tokens: int = 768, cache_path: Path | None = None):
        self.model_id = model_id
        self.threads = threads
        self.max_tokens = max_tokens
        self._model = None
        self._tok = None
        self._load_lock = threading.Lock()
        self.cache = ScoreCache(cache_path)
        self.load_error: str | None = None
        self.load_seconds: float | None = None
        self.last_call: dict = {}

    # ---------------------------------------------------------------- loading
    def load(self) -> "MambaBackbone":
        with self._load_lock:
            if self._model is not None:
                return self
            import torch
            from transformers import AutoTokenizer, MambaForCausalLM

            torch.set_num_threads(self.threads)
            t = time.perf_counter()
            try:
                if self._tok is None:
                    self._tok = AutoTokenizer.from_pretrained(self.model_id)
                    self._tok.padding_side = "right"
                    if self._tok.pad_token is None:
                        self._tok.pad_token = self._tok.eos_token
                self._model = MambaForCausalLM.from_pretrained(self.model_id).eval()
            except Exception as exc:  # record the exact blocker; callers decide what to do
                self.load_error = f"{type(exc).__name__}: {exc}"
                raise
            self.load_seconds = time.perf_counter() - t
        return self

    @property
    def loaded(self) -> bool:
        return self._model is not None

    # ---------------------------------------------------------------- features
    def record_features(self, texts: list[str], max_tokens: int = 96, batch_size: int = 64) -> np.ndarray:
        """Returns array [n, 2 + hidden]: mean NLL, max NLL, mean-pooled last hidden state."""
        import torch

        self.load()
        hidden = self._model.config.hidden_size
        out = np.zeros((len(texts), 2 + hidden), dtype="float32")
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            enc = self._tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                            max_length=max_tokens)
            with torch.no_grad():
                res = self._model(input_ids=enc["input_ids"], output_hidden_states=True)
            mask = enc["attention_mask"].float()
            logp = torch.log_softmax(res.logits[:, :-1].float(), -1)
            tok_lp = logp.gather(2, enc["input_ids"][:, 1:, None])[..., 0]
            m = mask[:, 1:]
            nll = -(tok_lp * m)
            mean_nll = nll.sum(1) / m.sum(1).clamp(min=1)
            max_nll = nll.max(1).values
            h = res.hidden_states[-1]
            pooled = (h * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            block = torch.cat([mean_nll[:, None], max_nll[:, None], pooled], 1).numpy()
            out[idx] = block
        return out

    # ---------------------------------------------------------------- answer scoring
    def tokenizer(self):
        """The model's tokenizer alone (cheap; used to count tokens without loading weights)."""
        if self._tok is None:
            from transformers import AutoTokenizer

            with self._load_lock:
                if self._tok is None:
                    tok = AutoTokenizer.from_pretrained(self.model_id)
                    tok.padding_side = "right"
                    if tok.pad_token is None:
                        tok.pad_token = tok.eos_token
                    self._tok = tok
        return self._tok

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer()(text, add_special_tokens=False)["input_ids"])

    def _encode(self, prompt: str, candidates: list[str], max_tokens: int | None = None):
        mt = max_tokens or self.max_tokens
        p_ids = self._tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = [self._tok(" " + c, add_special_tokens=False)["input_ids"] or [self._tok.eos_token_id] for c in candidates]
        room = mt - max(len(c) for c in c_ids)
        return (p_ids[-room:] if room > 0 else p_ids[-1:]), c_ids

    def score_candidates(self, prompt: str, candidates: list[str], max_tokens: int | None = None) -> list[float]:
        """Mean log-probability of each candidate's tokens given the prompt (higher = preferred).
        The prompt is truncated from the left to ``max_tokens`` (the question is at the end).
        ``self.last_call`` records whether the scores came from the cache (latency honesty)."""
        if not candidates:
            return []
        mt = max_tokens or self.max_tokens
        key = hashlib.sha256(json.dumps([self.model_id, mt, prompt, candidates]).encode()).hexdigest()
        hit = self.cache.get(key)
        if hit is not None:
            self.last_call = {"cache_hit": True, "n_candidates": len(candidates)}
            return hit
        import copy

        import torch

        self.load()
        p_ids, c_ids = self._encode(prompt, candidates, mt)
        self.last_call = {"cache_hit": False, "n_candidates": len(candidates), "prompt_tokens": len(p_ids),
                          "candidate_tokens": sum(len(c) for c in c_ids)}
        L = len(p_ids)
        scores = []
        with torch.no_grad():
            # logits_to_keep=1: only the last position's logits are materialized (a 2048-token prompt
            # would otherwise allocate 2048 x 50280 floats per call; same values for the last position)
            out = self._model(input_ids=torch.tensor([p_ids]), use_cache=True, logits_to_keep=1)
            first = torch.log_softmax(out.logits[0, -1].float(), -1)
            prefix_cache = out.cache_params
            del out
            for c in c_ids:
                lps = [float(first[c[0]])]
                cache = copy.deepcopy(prefix_cache)
                for j in range(len(c) - 1):
                    o = self._model(input_ids=torch.tensor([[c[j]]]), cache_params=cache, use_cache=True,
                                    cache_position=torch.tensor([L + j]), logits_to_keep=1)
                    cache = o.cache_params
                    lps.append(float(torch.log_softmax(o.logits[0, -1].float(), -1)[c[j + 1]]))
                    del o
                del cache
                scores.append(sum(lps) / len(lps))
            del prefix_cache, first
        gc.collect()
        self.cache.put(key, scores)
        return scores

    def score_candidates_full(self, prompt: str, candidates: list[str]) -> list[float]:
        """Reference implementation (batched full forward passes); used to validate the cached path."""
        import torch

        self.load()
        p_ids, c_ids = self._encode(prompt, candidates)
        seqs = [p_ids + c for c in c_ids]
        L = max(len(s) for s in seqs)
        pad = self._tok.pad_token_id
        ids = torch.tensor([s + [pad] * (L - len(s)) for s in seqs])
        with torch.no_grad():
            logits = self._model(input_ids=ids).logits.float()
        logp = torch.log_softmax(logits[:, :-1], -1)
        scores = []
        for b, c in enumerate(c_ids):
            pos = torch.arange(len(p_ids) - 1, len(p_ids) - 1 + len(c))
            tgt = torch.tensor(c)
            scores.append(float(logp[b, pos, tgt].mean()))
        return scores

    def generate(self, prompt: str, max_new_tokens: int = 24, max_prompt_tokens: int | None = None) -> str:
        """Greedily generate text, optionally truncating the prompt to a token budget."""
        import torch

        self.load()
        ids = self._tok(prompt, return_tensors="pt")
        if max_prompt_tokens is not None and ids["input_ids"].shape[1] > max_prompt_tokens:
            ids = {key: value[:, -max_prompt_tokens:] for key, value in ids.items()}
        input_tokens = ids["input_ids"].shape[1]
        with torch.no_grad():
            g = self._model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
        generated = g[0, input_tokens:]
        self.last_call = {"cache_hit": False, "prompt_tokens": input_tokens,
                          "generated_tokens": int(generated.shape[0]), "max_new_tokens": max_new_tokens}
        return self._tok.decode(generated, skip_special_tokens=True)
