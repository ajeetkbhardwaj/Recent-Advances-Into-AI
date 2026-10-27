"""TreeSight-Live: Ollama attachment point.

Attaches the ``TreeSight`` sidecar (``treesight.py``) to a locally-running Ollama model
(e.g. ``gemma4:12b``) via the native HTTP API -- NO model modification, NO fine-tuning,
NO separate HF serving. The sidecar simply "watches" the model answer real queries and
learns a growing, human-readable decision tree of *where* the model is reliable.

Signals used by the feedback layer, sourced from Ollama's ``/api/chat``:

* ``entropy``     -- cross-entropy proxy from per-token ``logprobs`` (``logprobs: true``).
  Values are normalized to [0,1]; low => confident, high => hedging/drifty.
* ``consistency`` -- mean pairwise similarity of ``num_samples`` generations at a small
  sampling temperature (self-consistency).

Both are observations of the base model's own output -- the base model is never on the
autograd graph and its weights are never touched.

Usage
-----
    python treesight_ollama.py --queries 12 --samples 2 --min-samples 2

Or programmatically::

    from treesight import TreeSight
    from treesight_ollama import OllamaBase

    sidecar = TreeSight(OllamaBase(model="gemma4:12b"))
    out = sidecar.run("fix this error: TypeError: 'NoneType' object is not subscriptable")
    print(out["explanation"])
"""

from __future__ import annotations

import argparse
import difflib
import json
import time
from typing import Dict, List, Optional, Tuple

import urllib.request

from treesight import (
    FeatureBridge,
    FeedbackAbstraction,
    TreeSight,
    TSTree,
)

# Tokens that delimit Ollama's "thinking" channel (channel markers). They are metadata,
# not model text, so they are excluded from the entropy estimate.
_CHANNEL_MARKER = "channel"


class OllamaError(RuntimeError):
    pass


class OllamaBase:
    """Callable stand-in for a model served by a local Ollama instance.

    ``__call__(query, hint=None) -> (response, meta)`` matches TreeSight's base-model
    contract exactly. ``hint`` is injected as context-level (prompt-only) guidance, never
    into the weights.
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

    # ------------------------------------------------------------------ HTTP
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
            raise OllamaError(f"Ollama HTTP {e.code}: {e.read().decode()[:300]}") from e
        except urllib.error.URLError as e:
            raise OllamaError(
                f"cannot reach Ollama at {self.host} -- is the server running?"
            ) from e

    # ------------------------------------------------------------------ chat
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
            payload["think"] = False   # skip the reasoning channel: ~10x faster

        t0 = time.time()
        data = self._post(payload)
        self.calls += 1

        message = data.get("message")
        if message is None:
            raise OllamaError(
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

    # ------------------------------------------------------------------ signals
    @staticmethod
    def _non_channel_logprobs(logprobs: List[Dict]) -> List[float]:
        """Per-token logprobs with thinking-channel delimiters filtered out."""
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
        """Cross-entropy proxy -> [0,1]. 0 = perfectly confident, 1 = dithering."""
        lps = self._non_channel_logprobs(logprobs)
        if not lps:
            return 1.0
        ce = -sum(lps) / len(lps)          # mean negative-logloss in nats
        return max(0.0, min(1.0, ce / self.ce_scale))

    @staticmethod
    def _similarity(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        cap = 4000
        return difflib.SequenceMatcher(None, a[:cap], b[:cap]).ratio()

    # ------------------------------------------------------------------ base contract
    def __call__(self, query: str, hint: Optional[str] = None) -> Tuple[str, Dict]:
        content = query
        if hint:
            content = (
                f"{query}\n\n"
                f"[TreeSight region context: {hint}]"
            )

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
# Demo runner
# --------------------------------------------------------------------------------------

_DEMO_QUERIES = [
    "explain how attention works in a transformer",
    "fix this error: TypeError: 'NoneType' object is not subscriptable",
    "why did my training loss jump to NaN overnight?",
    "what is the capital of France?",
    "write a regex to extract email addresses from text",
    "is 2+2 really 4? are you absolutely sure?",
    "why does numpy broadcast (3,1) against (4,) fail?",
    "tell me a recipe for chocolate chip cookies",
]


def _fmt(item: Dict) -> str:
    e = item["trace"]
    steps = [s.get("split") or s.get("type") for s in e["path"]] or []
    return (
        f"R={e['reward']:+.2f}  path={steps or ['root']}  "
        f"leaf={e['leaf']['uid']} (n={e['leaf']['N_eff']}, r={e['leaf']['mean_r']:+.2f})"
    )


def demo(
    queries: List[str],
    models_samples: int,
    min_samples: float,
    growth_threshold: float,
    seed: int,
    steer_demo: bool,
    model: str = "gemma4:12b",
    num_predict: int = 192,
    thinking: bool = False,
    save_path: Optional[str] = None,
) -> None:
    import random

    random.seed(seed)

    base = OllamaBase(model=model, num_samples=models_samples,
                      num_predict=num_predict, thinking=thinking)
    tree = TSTree(
        in_features=len(FeatureBridge.FEATURE_NAMES),
        min_samples=min_samples,
        growth_threshold=growth_threshold,
        lr=5e-2,
    )
    sidecar = TreeSight(base, tree=tree)

    print(f"TreeSight-Live | Ollama attachment  (model={base.model}, samples={base.num_samples})")
    print(f"  entropy/consistency from base-model output; base weights untouched\n")

    for i, q in enumerate(queries, 1):
        t0 = time.time()
        out = sidecar.run(q, user_rating=0.0, steer=False)
        dt = time.time() - t0
        print(f"[{i:2d}] ({dt:5.1f}s) {_fmt(out)}")
        okay = True
        for u in out["grown_nodes"]:
            print(f"       SPLIT -> {u}")
        if not out["trace"]["path"] and i == 1:
            print("     (root not yet split; tree still collecting evidence)")
        print(f"      Q: {q[:70]}")

    print("\n" + "=" * 76)
    print("KNOWLEDGE MAP  (regions the sidecar learned from live inference)")
    print("=" * 76)
    km = json.loads(sidecar.knowledge_map())
    for leaf in km["leaves"]:
        print(f"  {leaf['uid']:5s} {leaf['path'] or 'root':38s} "
              f"n={leaf['N_eff']:6.1f} r={leaf['mean_r']:+.2f} "
              f"w=[{leaf['wilson'][0]:+.2f},{leaf['wilson'][1]:+.2f}] drifted={leaf['drifted']}")
    print(f"\n  nodes={km['nodes']} leaves={len(km['leaves'])} depth={km['depth']} "
          f"drifted_nodes={len(km['drifted_nodes'])}")

    if save_path:
        sidecar.save(save_path)
        print(f"  saved learned tree -> {save_path}")

    if steer_demo and km["leaves"]:
        q = "fix this error: TypeError: 'NoneType' object is not subscriptable"
        print("\n" + "-" * 76)
        print("STEER-BACK demo (prompt-level guidance injected, weights untouched)")
        print("-" * 76)
        out = sidecar.run(q, user_rating=0.0, steer=True)
        hint = out["explanation"]["leaf"]["hint"]
        print(f"  Q: {q}")
        print(f"  leaf: {out['explanation']['leaf']['uid']} "
              f"(n={out['explanation']['leaf']['N_eff']})")
        print(f"  injected hint: {hint}")
        print(f"  response[0:200]: {out['response'][:200]!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="TreeSight-Live attached to local Ollama.")
    ap.add_argument("--model", default="gemma4:12b")
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--samples", type=int, default=2, help="generations per query")
    ap.add_argument("--num-predict", type=int, default=192, help="max tokens per generation")
    ap.add_argument("--thinking", action="store_true", help="allow the reasoning channel (slow)")
    ap.add_argument("--min-samples", type=float, default=2.0, help="leaf grow threshold")
    ap.add_argument("--growth-threshold", type=float, default=0.30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-steer", action="store_true")
    ap.add_argument("--save", default=None, metavar="PATH",
                    help="persist the learned tree state to PATH")
    args = ap.parse_args()

    queries = [_DEMO_QUERIES[i % len(_DEMO_QUERIES)] for i in range(args.queries)]
    demo(queries, args.samples, args.min_samples, args.growth_threshold,
         args.seed, steer_demo=not args.no_steer,
         model=args.model, num_predict=args.num_predict, thinking=args.thinking,
         save_path=args.save)


if __name__ == "__main__":
    main()