"""TreeSight-Live :: interactive query shell (Ollama + interpretability).

A single interactive script: you type a query, it is answered by your LOCAL Ollama model
(e.g. ``gemma4:12b``), and TreeSight explains WHY that answer landed in the region it did
-- the decision path, the feature values, the confidence (Wilson interval), the collected
evidence, and the region hint. The sidecar keeps learning from every query and the learned
tree can be saved/loaded so built-up knowledge survives restarts.

The base model is never modified or fine-tuned; TreeSight only observes its outputs
(logprobs + self-consistency between ``--samples`` generations) and learns its OWN small
tree, which it also uses to steer context back into the prompt (astronomically auditable).

Requirements: Ollama running locally. Pure stdlib + the bundled ``treesight.py``.

Examples
--------
    python treesight_chat.py                              # interactive session
    python treesight_chat.py -l treesight_gemma4.pt       # resume learned knowledge
    python treesight_chat.py --samples 3 --num-predict 128
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import urllib.request

from treesight import (
    FeatureBridge,
    FeedbackAbstraction,
    TreeSight,
    TSTree,
)

_CHANNEL_MARKER = "channel"


# --------------------------------------------------------------------------------------
# Ollama HTTP adapter
# --------------------------------------------------------------------------------------

class OllamaBase:
    """Callable stand-in for a model served by local Ollama.

    ``__call__(query, hint=None) -> (response, meta)`` matches TreeSight's base contract.
    ``hint`` is prompt-level context only -- the model's weights are never touched.
    """

    DEFAULT_SYSTEM = (
        "You are a precise, helpful assistant. Answer directly and concisely; "
        "when you are unsure, say so briefly."
    )

    def __init__(
        self,
        model: str = "gemma4:12b",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
        num_predict: int = 192,
        num_ctx: int = 2048,
        num_samples: int = 2,
        sample_temperature: float = 0.6,
        thinking: bool = False,
        ce_scale: float = 0.75,
        timeout: float = 300.0,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.num_samples = num_samples
        self.sample_temperature = sample_temperature
        self.thinking = thinking
        self.ce_scale = ce_scale
        self.timeout = timeout
        self.calls = 0
        self.last_duration_ms: Optional[float] = None

    # --- HTTP ----------------------------------------------------------------
    def _post(self, payload: Dict) -> Dict:
        url = f"{self.host}/api/chat"
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"Ollama HTTP {e.code}: {e.read().decode()[:300]}") from e
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"cannot reach Ollama at {self.host} -- is the server running?"
            ) from e

    # --- chat ----------------------------------------------------------------
    def chat(
        self,
        content: str,
        temperature: float,
        want_logprobs: bool = False,
    ) -> Tuple[str, Dict]:
        payload: Dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.DEFAULT_SYSTEM},
                {"role": "user", "content": content},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.num_predict,
                "num_ctx": self.num_ctx,
            },
        }
        if want_logprobs:
            payload["logprobs"] = True
        if not self.thinking:
            payload["think"] = False   # skip reasoning channel: ~10x faster

        t0 = time.time()
        data = self._post(payload)
        self.calls += 1
        self.last_duration_ms = (data.get("eval_duration") or 0) / 1e6

        message = data.get("message")
        if message is None:
            raise RuntimeError(
                f"Ollama returned no message (done_reason={data.get('done_reason')!r})"
            )
        text = message.get("content", "").strip()
        meta = {
            "eval_count": data.get("eval_count", 0),
            "eval_duration_ms": (data.get("eval_duration") or 0) / 1e6,
            "done_reason": data.get("done_reason"),
            "logprobs": data.get("logprobs") or [],
        }
        return text, meta

    # --- signals -------------------------------------------------------------
    @staticmethod
    def _non_channel_logprobs(logprobs: List[Dict]) -> List[float]:
        out = []
        for lp in logprobs:
            tok = str(lp.get("token", ""))
            if _CHANNEL_MARKER.lower() in tok.lower():
                continue
            v = lp.get("logprob")
            if isinstance(v, (int, float)):
                out.append(float(v))
        return out

    def entropy_from_logprobs(self, logprobs: List[Dict]) -> float:
        lps = self._non_channel_logprobs(logprobs)
        if not lps:
            return 1.0
        ce = -sum(lps) / len(lps)
        return max(0.0, min(1.0, ce / self.ce_scale))

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        cap = 4000
        return difflib.SequenceMatcher(None, a[:cap], b[:cap]).ratio()

    # --- base-model contract -------------------------------------------------
    def __call__(self, query: str, hint: Optional[str] = None) -> Tuple[str, Dict]:
        content = query
        if hint:
            content = f"{query}\n\n[TreeSight region context: {hint}]"

        main, main_meta = self.chat(content, self.temperature, want_logprobs=True)
        samples: List[str] = [main]
        for _ in range(max(0, self.num_samples - 1)):
            s, _ = self.chat(content, self.sample_temperature, want_logprobs=False)
            samples.append(s)

        consistency = 0.0
        if len(samples) > 1:
            consistency = sum(self._similarity(samples[0], s)
                              for s in samples[1:]) / (len(samples) - 1)
        entropy = self.entropy_from_logprobs(main_meta["logprobs"])

        meta = {
            "model": self.model,
            "entropy": round(entropy, 4),
            "consistency": round(consistency, 4),
            "eval_count": main_meta["eval_count"],
            "eval_duration_ms": main_meta["eval_duration_ms"],
            "done_reason": main_meta["done_reason"],
            "n_calls": self.num_samples,
        }
        return main, meta


# --------------------------------------------------------------------------------------
# Interpretability rendering
# --------------------------------------------------------------------------------------

_LOCAL_MAP = {
    "query_len": "query length",
    "has_code": "mentions code",
    "n_wh_words": "how/what/why words",
    "n_qmarks": "question marks",
    "resp_len": "answer length",
    "hedge_count": "hedging words",
    "uncertainty": "model uncertainty",
}


def _steps(ex: Dict) -> List[str]:
    steps: List[str] = []
    for s in ex["path"]:
        if "split" in s:
            fname = _LOCAL_MAP.get(s["split"].split("<=")[0].split(">")[0].strip(),
                                  s["split"].split("<=")[0])
            steps.append(s["split"])
        elif "type" in s:
            steps.append(f"soft-router p_left={s.get('p_left')}")
    return steps or ["root"]


def _confidence_text(w: List[float]) -> str:
    lo, hi = w
    if lo > 0.2:
        return "reliable"
    if hi < 0.0:
        return "unreliable"
    return "hedging / uncertain"


def render_interpretability(ex: Dict, feats: Dict, meta: Dict) -> str:
    """Human-readable explanation of WHY this query landed in this region."""
    leaf = ex["leaf"]

    lines: List[str] = []
    lines.append("  where it went:")
    for s in ex["path"]:
        val = s.get("value") if "value" in s else "?"
        fname = s["split"].split("<=")[0].strip()
        human = _LOCAL_MAP.get(fname, fname)
        lines.append(f"    - {human} [feature = {val}] -> {s['split']}")
    if not ex["path"]:
        lines.append("    - root (tree still gathering evidence)")

    lines.append("")
    lines.append("  measured signals:")
    lines.append(f"    - uncertainty (logprob-proxy) : {meta.get('entropy', 0.0)}")
    lines.append(f"    - self-consistency (n={meta.get('n_calls', 1)} samples): "
                 f"{meta.get('consistency', 0.0)}")
    lines.append(f"    - reward R                    : "
                 f"{meta.get('_reward', 0.0):+.2f}")

    lines.append("")
    lines.append("  region knowledge:")
    lines.append(f"    - leaf node      : {leaf['uid']} (depth {leaf['depth']})")
    lines.append(f"    - evidence so far: n = {leaf['N_eff']} interactions")
    lines.append(f"    - region score   : {leaf['mean_r']:+.2f} "
                 f"(var {leaf['var_r']:.3f}) -> {_confidence_text(leaf['wilson'])}")
    lines.append(f"    - 95% interval   : [{leaf['wilson'][0]:+.2f}, "
                 f"{leaf['wilson'][1]:+.2f}]")
    lines.append(f"    - drifted        : {leaf['drifted']}")
    lines.append(f"    - hint           : {leaf['hint']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Save / load helpers
# --------------------------------------------------------------------------------------

def save_sidecar(sidecar: TreeSight, path: str) -> None:
    sidecar.save(path)


# --------------------------------------------------------------------------------------
# Interactive shell
# --------------------------------------------------------------------------------------

_BANNER = """
============================================================
  TreeSight-Live :: interactive interpretability shell
  (model: {model} | samples: {samples} | thinking: {thinking})
============================================================
  Type a question and press Enter.  Commands:
    /explain  last answer's decision path again
    /regions  show the learned region map (knowledge map)
    /save     <path>  persist learned knowledge
    /quit     exit
""".strip()


def run_shell(
    model: str,
    samples: int,
    num_predict: int,
    thinking: bool,
    resume: Optional[str] = None,
) -> None:
    base = OllamaBase(model=model, num_samples=samples,
                      num_predict=num_predict, thinking=thinking)

    tree = TSTree(in_features=len(FeatureBridge.FEATURE_NAMES),
                  min_samples=2.0, growth_threshold=0.30, lr=5e-2)
    sidecar = TreeSight(base, tree=tree)

    if resume and os.path.exists(resume):
        try:
            sidecar = TreeSight.load(resume, base)
            print(f"resumed learned tree from {resume} "
                  f"({len(sidecar.tree.all_nodes)} nodes)")
        except Exception as e:  # noqa: BLE001
            print(f"could not resume {resume}: {e}; starting fresh")

    print(_BANNER.format(model=model, samples=samples, thinking=thinking))

    while True:
        try:
            raw = input("\n❯ ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not raw:
            continue

        low = raw.lower()
        if low in ("/quit", "/exit", "/q"):
            print("bye")
            break
        if low in ("/explain", "/e"):
            print("  (each answer already shows its path; use /regions for the map)")
            continue
        if low in ("/regions", "/r"):
            print_regions(sidecar)
            continue
        if low.startswith("/save"):
            parts = raw.split()
            path = parts[1] if len(parts) > 1 else "treesight_state.pt"
            save_sidecar(sidecar, path)
            print(f"  saved learned tree -> {path}")
            continue
        if low.startswith("/"):
            print("  unknown command")
            continue

        # ---- answer + interpretability ----------------------------------
        segments = _segment_queries(raw)

        if len(segments) == 1:
            _run_one(sidecar, base, segments[0], samples)
        else:
            print(f"  (detected {len(segments)} queries in your input -- "
                  f"learning each separately)")
            for i, seg in enumerate(segments, 1):
                print(f"\n  ══ part {i}/{len(segments)}: {seg}")
                _run_one(sidecar, base, seg, samples)


def _run_one(sidecar: TreeSight, base: OllamaBase,
             query: str, samples: int) -> None:
    """Run a single query through TreeSight and print answer + interpretability."""
    print("  (model thinking...)")
    try:
        out = sidecar.run(query, user_rating=0.0, steer=True)
    except RuntimeError as e:
        print(f"  ERROR: {e}")
        return

    ex = out["explanation"]
    reward = out["trace"]["reward"]
    src = out["trace"].get("meta", {})
    rmeta = {
        "entropy": src.get("entropy", 0.0),
        "consistency": src.get("consistency", 0.0),
        "n_calls": src.get("n_calls", samples),
        "eval_ms": round(base.last_duration_ms or 0.0, 0),
    }
    feats = sidecar.bridge.build(query, out["response"], src).to_dict()
    meta = {"_reward": reward, **rmeta}

    print("\n" + "─" * 68)
    _print_answer(out["response"])
    print("─" * 68)
    print(render_interpretability(ex, feats, meta))


def _segment_queries(text: str) -> List[str]:
    """Split a pasted paragraph into separate query chunks.

    Splits on newlines and on sentence/question boundaries, keeping the query cohesive.
    Use heuristics: prioritize '?', then sentence-ending periods, then newlines.
    """
    text = text.strip()
    if not text:
        return []
    if "\n" in text:
        pieces = [p.strip() for p in text.split("\n") if p.strip()]
        if len(pieces) > 1:
            return pieces

    # Split on sentence/question boundaries.
    chunks: List[str] = []
    buf: List[str] = []
    for token in re.split(r"(?<=[?.!])\s+", text):
        if not token.strip():
            continue
        buf.append(token)
        # End a chunk at a question mark or a sentence-ending period followed by
        # a capital-start word (heuristic for a new query).
        if token.rstrip().endswith("?") or _looks_like_new_query(token, buf):
            chunks.append(" ".join(buf).strip())
            buf = []
    if buf:
        chunks.append(" ".join(buf).strip())
    return [c for c in chunks if c]


def _looks_like_new_query(main: str, buf: List[str]) -> bool:
    """True if 'main' ends a sentence AND the next token looks like a fresh query."""
    if not main.rstrip().endswith("."):
        return False
    if len(buf) == 1:
        return False
    return True


def _print_answer(response: str) -> None:
    for para in response.split("\n"):
        print(("  " + para).rstrip())


def print_regions(sidecar: TreeSight) -> None:
    km = json.loads(sidecar.knowledge_map())
    print("\n  learned regions (knowledge map):")
    print("  " + "-" * 64)
    for leaf in km["leaves"]:
        lo, hi = leaf["wilson"]
        conf = _confidence_text(leaf["wilson"])
        print(f"  {leaf['uid']:5s} {leaf['path'] or 'root':30s} "
              f"n={leaf['N_eff']:6.1f} r={leaf['mean_r']:+.2f} "
              f"[{lo:+.2f},{hi:+.2f}] {conf}")
    print(f"\n  nodes={km['nodes']} leaves={len(km['leaves'])} depth={km['depth']} "
          f"queries={km['total_queries']} drifted={len(km['drifted_nodes'])}")


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="TreeSight-Live interactive interpretability shell (Ollama).")
    ap.add_argument("--model", default="gemma4:12b")
    ap.add_argument("--samples", type=int, default=2, help="generations per query")
    ap.add_argument("--num-predict", type=int, default=192)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("-l", "--load", default=None, metavar="PATH",
                    help="resume learned tree state from PATH")
    args = ap.parse_args()

    run_shell(model=args.model, samples=args.samples,
              num_predict=args.num_predict, thinking=args.thinking,
              resume=args.load)


if __name__ == "__main__":
    main()