"""Phase-0 probe: can state-spaces/mamba-130m-hf load and run on this CPU, and how does
it behave on a record-reading question? Writes artifacts/models/mamba_probe.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import psutil
import torch
from transformers import AutoTokenizer, MambaForCausalLM

from agentmamba.config import PROJECT_ROOT
from agentmamba.reproducibility import environment_info

MODEL = "state-spaces/mamba-130m-hf"


def main() -> None:
    torch.set_num_threads(8)
    proc = psutil.Process()
    rss0 = proc.memory_info().rss
    t = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = MambaForCausalLM.from_pretrained(MODEL).eval()
    load_s = time.perf_counter() - t
    report = {"model": MODEL, "env": environment_info(), "load_seconds": round(load_s, 2),
              "rss_mb_after_load": round((proc.memory_info().rss - rss0) / 2**20, 1),
              "n_params": sum(p.numel() for p in model.parameters()),
              "architecture": type(model).__name__, "config": {k: getattr(model.config, k) for k in
                                                              ("hidden_size", "num_hidden_layers", "state_size", "vocab_size")}}
    # forward throughput
    texts = ["Refractory inspection of ladle C01-LD03: hairline crack at slag line; lining thickness 142 mm."] * 32
    enc = tok(texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        model(**enc)  # warm-up
        t = time.perf_counter()
        out = model(**enc, output_hidden_states=True)
        report["forward_batch32_seconds"] = round(time.perf_counter() - t, 3)
        report["tokens_per_record"] = int(enc["input_ids"].shape[1])
        report["hidden_state_shape"] = list(out.hidden_states[-1].shape)
    # free-generation QA probe on records
    prompt = ("Plant records:\n"
              "[1] Heat C01-H0012 tapped into ladle C01-LD04.\n"
              "[2] Heat C01-H0013 tapped into ladle C01-LD02.\n"
              "[3] Caster C01-CC1 cast heat C01-H0012 in sequence C01-CS03 producing slab C01-SL0012.\n"
              "Question: Which ladle carried heat C01-H0012?\nAnswer:")
    ids = tok(prompt, return_tensors="pt")
    gens = {}
    with torch.no_grad():
        t = time.perf_counter()
        g = model.generate(**ids, max_new_tokens=16, do_sample=False)
        gens["greedy"] = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)
        report["generate_16_tokens_seconds"] = round(time.perf_counter() - t, 3)
        # likelihood scoring of candidates (the answer-extraction mechanism used by AgentMamba)
        scores = {}
        for cand in ["C01-LD04", "C01-LD02", "C01-CS03"]:
            full = tok(prompt + " " + cand, return_tensors="pt")["input_ids"]
            n_prompt = ids["input_ids"].shape[1]
            logits = model(full).logits
            lp = torch.log_softmax(logits[0, :-1], -1).gather(1, full[0, 1:, None])[:, 0]
            scores[cand] = round(float(lp[n_prompt - 1:].mean()), 4)
        gens["candidate_mean_logprob"] = scores
    report["qa_probe"] = gens
    report["rss_mb_peak_delta"] = round((proc.memory_info().rss - rss0) / 2**20, 1)
    out_path = PROJECT_ROOT / "artifacts" / "models" / "mamba_probe.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
