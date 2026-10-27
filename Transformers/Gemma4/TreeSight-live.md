# TreeSight-Live

## A Model-Agnostic, Attach-on-Inference Continual Interpretability Sidecar

**Version:** 0.1 (design + implementation guide)
**Status:** Implemented (`treesight.py`) — attach to any existing model, no fine-tuning

---

## Abstract

Existing LLMs — pretrained or post-trained — are black boxes. GiST-4 (previous design)
showed how to *fine-tune* a readable tree onto Gemma. TreeSight-Live removes the fine-tuning
entirely: it is a **sidecar** that attaches to **any** model performing live inference and
**learns from that inference** — continuously — to produce human-readable explanations *and*
auditable, reusable knowledge, never touching the base model's weights.

Every user query becomes a training step for the sidecar:

```
query —> base model (untouched) —> response
                  |                    |
                  v                    v
          FeatureBridge            FeedbackAbstraction (reward R)
                  |                    |
                  v                    v
           DynamicSoftTree  —Phase A (no_grad): accumulate & grow
                  |           Phase B (autograd): weights learn R
                  v
          Knowledge Map  —->  context-level hints steered back to the base,
                              plus a fully interpretable decision path per query
```

The aspiration the user asked for — *"if it has weights they must learn as users use it, so
that its accumulated knowledge becomes available to the model with full interpretability"* —
is realized exactly: the tree's **routers and leaf readouts are weights**, updated online
from live traffic; leaves accumulate **decaying** sufficient statistics (like a sliding
attention window over time, in the spirit of Gemma's hybrid attention); and each leaf's
knowledge is exported and *steered back* into inference via context-level hints (prompt
injection, never weight updates).

---

## Part I — Theory

### 1. Problem formulation

The base model $\Phi$ is an unknown, fixed function producing a response $y$ for each query
$x$. We do not observe its internals (or we choose not to rely on them). We attach a sidecar
whose job is to learn a **region-conditioned explanation** of $\Phi$'s behavior.

Let:

- $\phi(x) \in \mathbb{R}^{K}$ — interpretable features derived from the interaction
  (query lexical stats, response stats, uncertainty proxies).
- $r(y \mid x) \in [-1, 1]$ — a **feedback reward** from the FeedbackAbstraction layer
  (self-consistency, entropy/uncertainty, downstream success, or explicit user rating).
- $\mathcal{T}$ — a soft (or hard) binary decision tree with leaves $\mathcal{L}$ and
  router weights at internal nodes.

**Learning goal (online, non-stationary):**

$$
\forall \ell \in \mathcal{L}: \quad \hat{r}_\ell(x) \approx \mathbb{E}[\, r(y \mid x) \mid x \in \text{region}_\ell\,]
$$

i.e., each leaf learns the *expected helpfulness/uncertainty of the model in its region*,
and routers learn the region boundaries — both updated incrementally as users interact.

### 2. A decision tree as a contextual bandit

Each leaf is an **arm**; the user query is **context**; the reward is $R$. The tree is a
hierarchical, adaptive policy:

- **Exploitation:** route $\phi(x)$ to the leaf and read its average reward + best historical
  strategy (the *hint*).
- **Exploration:** fleaves with low effective sample size $N_\text{eff}$ get a conservative
  bonus (UCB-style), so untested regions are not misreported as trustworthy.
- **Growth as exploration:** a leaf splits when its within-leaf reward variance is high and
  $N_\text{eff}$ is sufficient — the tree expands exactly where behavior is unstable.

This is a **contextual bandit over a dynamically-growing tree**, where
`ddtree.py`'s `evaluate_and_grow` is the structure-exploration operator.

### 3. Decaying sufficient statistics (recency-weighted memory)

Naive accumulation (`N += Σw`) treats year-old traffic as equal to yesterday's. We want a
**sliding memory**: statistics decay per observation, mirroring Gemma's "sliding window over
time" (485). Gill

For each node $v$, with decay factor $0<\gamma<1$ and per-batch flow $w$:

$$
N_v \leftarrow \gamma\, N_v + \sum_i w_i, \qquad
s^x_v \leftarrow \gamma\, s^x_v + \sum_i w_i\, x_i, \qquad
s^{x2}_v \leftarrow \gamma\, s^{x2}_v + \sum_i w_i\, x_i^{\otimes 2}
$$

and likewise for reward. This yields a **bounded, stale-forgetting** empirical measure:

$$
\text{eff. sample} = N_v, \qquad \hat\mu_x = s^x_v / N_v, \qquad
\mathrm{Cov} = s^{x2}_v / N_v - \hat\mu_x^{\otimes 2}.
$$

$\gamma \approx 0.99$ gives a half-life of ≈ 69 batches — if the model's behavior drifts
(e.g., the base model is upgraded, or the query distribution shifts), old regions
automatically fade and can be re-explored.

### 4. Growth criterion (measure-theoretic purity, online)

A leaf $v$ with $N_v \ge N_\min$ splits when

$$
G(v) = \alpha \underbrace{\frac{\mathrm{tr}\,\mathrm{Cov}_X(v)}{\mathrm{tr}\,\mathrm{Cov}_X(\mathrm{root})}}_{\text{feature dispersal}}
+ \beta \underbrace{\frac{\mathrm{Var}_R(v)}{\mathrm{Var}_R(\mathrm{root})}}_{\text{behavioral instability}} > \tau.
$$

For interpretability-first operation (the default) the split is **hard and axis-aligned**:
choose feature $j^*$ and threshold $t$ maximizing **reward-variance reduction**,

$$
\Delta_j = \mathrm{Var}_R(v) - \big[\, p_L \mathrm{Var}_R(v_L) + p_R \mathrm{Var}_R(v_R) \,\big],
\qquad
(j^*, t) = \arg\max \Delta_j \;\; \text{subject to } p_L, p_R \ge \epsilon.
$$

with $t = \hat\mu_{x_{j^*}}$ (the node's empirical mean on that axis) for determinism. Each
split emits a human sentence: *"region splits on feature entropy ≤ 3.2 bits"*.

Optionally, soft-sigmoid routing (`ddtree.py`-style) can be enabled; then routers are
optimized in Phase B with the same objective, at the cost of axis-interpretability.

### 5. The two-phase online update

Maps directly onto `ddtree.py`'s two-phase paradigm, at **inference time** (per query/batch):

| Phase                            | Mode                | Operation                                                         | Gradient target      |
| -------------------------------- | ------------------- | ----------------------------------------------------------------- | -------------------- |
| **A — accumulate & grow** | `torch.no_grad()` | route batch; decay + accumulate stats;`evaluate_and_grow`       | none                 |
| **B — learn weights**     | autograd            | readout regression to$R$, flow-weighted; optional router update | sidecar weights only |

Phase B's loss:

$$
\mathcal{L} = \sum_\ell \pi_\ell(x)\, \big( f_\ell(\phi(x)) - R \big)^2 + \lambda_\text{purity}\, H(\pi)
$$

Back-prop never crosses into $\Phi$ (the base model is never on the graph).

### 6. Drift detection

A hidden drift in the base model (or in user intent) changes the conditional reward
distribution inside a leaf. Each node keeps two EMA's:

$$
\hat r^\text{fast}_v \leftarrow \eta_f \hat r^\text{fast}_v + (1 - \eta_f) R, \quad
\hat r^\text{slow}_v \leftarrow \eta_s \hat r^\text{slow}_v + (1 - \eta_s) R,
\qquad \eta_f \ll \eta_s.
$$

A node is flagged **drifted** when

$$
| \hat r^\text{fast}_v - \hat r^\text{slow}_v | > z \cdot \sigma_{R,v}^{(\text{slow})}
$$

(σ from the slow decayed variance). Drifted leaves are reported, and their statistics can be
reset to re-explore (optional `reset` policy). This is the sidecar's own "model-update"
awareness.

### 7. Knowledge-map schema (machine + human readable)

The accumulated knowledge is exported as JSON (diffable, cacheable, auditable):

```json
{
  "tree_version": 42,
  "roots_stats": {"N_eff": 4132.0, "mean_r": 0.31},
  "leaves": [
    {
      "uid": "n3",
      "depth": 2,
      "path": "entropy<=3.20 -> is_code=1",
      "split": {"feature": "entropy", "threshold": 3.20},
      "N_eff": 821.0,
      "mean_r": 0.87,
      "var_r": 0.11,
      "wilson_lo": 0.83, "wilson_hi": 0.90,
      "centroid": {"query_len": 78, "hedge_count": 0.4},
      "drifted": false,
      "hint": "Region reliably helpful (n=821, R=0.87). Suggest: ask for stack trace first."
    }
  ],
  "drift": [{"uid": "n9", "fast": -0.2, "slow": 0.4, "z": 2.1}],
  "coverage": {"fract_queries": 0.71, "active_leaves": 12}
}
```

The **hint** is the "rich knowledge available to the model": at inference, the sidecar
retrieves the winning leaf's hint and injects it into the base model's *context* (prompt),
not its weights.

---

## Part II — Implementation

Repositories/files:

```
ddtree.py                 — the two-phase DynamicSoftTree core (reference design)
treesight.py              — THE NEW IMPLEMENTATION (self-contained, trainable sidecar)
TreeSight-live.md         — this document
```

### 8. Interface contract (model-agnostic attachment)

The sidecar attaches to anything callable:

```python
base: Callable[[str, dict], tuple[str, dict]]   # (query) -> (response, meta)
# meta may carry optional: logprobs, samples (for self-consistency), ...
```

No `output_hidden_states`, no tokenizer unmasking, no gradients into the base. Works for
HF pipelines, API wrappers, a local GGUF/llama.cpp process, or a mock for dry-run.

### 9. Component inventory (`treesight.py`)

| Class                    | Responsibility                                                                                 |
| ------------------------ | ---------------------------------------------------------------------------------------------- |
| `Features` (dataclass) | list of (name, value) interpretable features                                                   |
| `FeatureBridge`        | query/response →`Features`, with optional logprob-entropy proxy                             |
| `FeedbackAbstraction`  | reward from (self-consistency, entropy, user rating)                                           |
| `TSNode`               | node: decayed stats, readout weight, split state, drift EMA's, hint                            |
| `TSTree`               | dynamic growing tree: sparse forward, Phase A/Phase B, split search, drift scan, knowledge map |
| `TreeSight`            | the attachable sidecar: run → explain → observe → steer → export                           |
| `MockBase`             | dry-run "existing model" with topic-dependent uncertainty (no transformers needed)             |
| `demo`                 | end-to-end: attach, chat, watch growth + paths + knowledge map                                 |

### 10. Split search (Phase A, hard axis-aligned)

For each leaf eligible to split, on the current batch clip restricted to that leaf:

1. compute reward mean/var,
2. for each feature $j$: threshold at batch-mean, compute $\Delta_j$ (both children
   non-empty),
3. split on the argmax; build children inheriting the parent readout (the tree's "unified
   K/V" trick from `_bifurcate_node`),
4. register children (optimizer sees them next step).

### 11. Phase B weight learning

Each leaf carries `readout: Linear(K, 1)` + scalar (or just scalar EMA in minimal mode).
Online optimizer is `AdamW` over readouts (and optional soft routers); flow-weighted MSE to
$R$. Only sidecar parameters — the base model is never parameterized.

### 12. Steer-back loop

```python
hint = sidecar.best_hint(query)      # leaf lookup by path probability
prompt = base_prompt(query)
if hint: prompt += f"\n[TreeSight] Region knowledge: {hint}"
response = base(prompt)              # same model, richer context
```

The base model remains 100% unchanged; the hint is purely conversational context — exactly
"giving the model its accumulated knowledge with full interpretability."

---

## Part III — Usage

### 13. Quick start

```bash
python treesight.py --queries 64                 # mock model, auto feedback
python treesight.py --queries 64 --real-feedback # type +1/-1 per query
python treesight.py --soft-routing               # sigmoid routers learn too
```

Attach to a real model:

```python
from transformers import pipeline
from treesight import TreeSight, FeatureBridge, FeedbackAbstraction

base = pipeline("text-generation", model="HuggingFaceTB/SmolLM2-135M")
sidecar = TreeSight(base, bridge=FeatureBridge(), feedback=FeedbackAbstraction())

out = sidecar.run("fix this TypeError: NoneType has no attribute foo")
print(out["explanation"])
print(sidecar.knowledge_map(pretty=True))
```

### 14. Hardware

Trivial: pure CPU is fine. `treesight.py` needs only PyTorch; the demo mock needs nothing
extra. At research scale (real base model serving queries) the sidecar adds ~`O(active)×K`
per query + one tiny AdamW step.

### 15. Experiment plan (Milestones)

| M  | Check                                             | Measure                                              |
| -- | ------------------------------------------------- | ---------------------------------------------------- |
| M0 | Mock end-to-end runs (this PR)                    | leaves grow, paths readable, map exports             |
| M1 | Attach to SmolLM2-135M / Gemma E4B-4bit on Kaggle | confusion regions detected vs. entropy ground truth  |
| M2 | 26B A4B server (via API) — logits/text only      | region coverage + drift after server retrain         |
| M3 | Steer-back A/B on a downstream task               | helpfulness +R with hints on, no harm with hints off |

---

## Part IV — Limitations

1. Reward quality is the ceiling: a noisy `FeedbackAbstraction` trains a noisy tree. The
   Wilson intervals and conservative reporting mitigate but don't remove this.
2. Hard single-axis splits are very readable but can over-fragment; `min_samples` and
   `min_reduction` hyperparameters control it.
3. The base model must remain behaviorally stationary *within* a leaf's memory horizon for
   stable hints; drift detection reports, it doesn't cure.
4. The sidecar explains *observed behavior*, not hidden weights: two models that behave
   identically will be explained identically even if internally different.

---

*Related:* `README.md` (Gemma 4 architecture), `GiST-technical-report.md` (predecessor
design), `ddtree.py` (two-phase soft-tree reference: `evaluate_and_grow`,
`accumulate_stats`, `_bifurcate_node`, `EPS` pruning).
