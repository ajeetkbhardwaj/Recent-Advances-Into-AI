"""TreeSight-Live: model-agnostic, attach-on-inference continual interpretability sidecar.

Attaches to ANY pretrained/post-trained model performing live inference and LEARNS from
that inference — without modifying or fine-tuning the base model in any way.

The design is fully specified in ``TreeSight-live.md``. In short:

* every user query -> base model -> response (+ optional scores) becomes a training step;
* a FeatureBridge turns each interaction into a small set of interpretable features;
* a dynamically growing tree (two-phase paradigm from ``ddtree.py``) accumulates DECAYING
  sufficient statistics and grows new leaves when within-leaf behavioural variance is high;
* leaf readouts are real weights, updated online (Phase B, autograd) toward the feedback
  reward produced by the FeedbackAbstraction layer;
* the resulting knowledge map (per-leaf stats + hints + confidence) is exported as JSON and
  steered back to the base model as context-level hints (prompt-only, weights untouched).

Formally this is a contextual bandit over a dynamically growing decision tree.

This module is self-contained (PyTorch only); the mock base requires no transformers install.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Probability mass below this threshold is pruned (the "product trap" guard from ddtree.py).
EPS: float = 1e-5

# Feature indices used for explanations. Keep aligned with FeatureBridge.FEATURE_NAMES.
_DEBUG = False


# --------------------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------------------

@dataclass
class Features:
    """One interpretation unit of an interaction: named scalar features."""

    values: Dict[str, float] = field(default_factory=dict)

    def tensor(self) -> torch.Tensor:
        return torch.tensor([self.values[n] for n in FeatureBridge.FEATURE_NAMES], dtype=torch.float32)

    def names(self) -> List[str]:
        return list(FeatureBridge.FEATURE_NAMES)

    def to_dict(self) -> Dict[str, float]:
        return dict(self.values)


class FeatureBridge:
    """Map a (query, response, meta) interaction to interpretable axis-aligned features.

    These features are deliberately hand-crafted and human-readable so that tree splits
    read as sentences ("region splits on entropy <= 3.2"). Works on text alone; optional
    ``meta`` may carry an uncertainty score (e.g., logprob entropy from an API).
    """

    FEATURE_NAMES: Tuple[str, ...] = (
        "query_len",
        "has_code",
        "n_wh_words",
        "n_qmarks",
        "resp_len",
        "hedge_count",
        "uncertainty",   # from meta (logprob-derived) if provided, else a text proxy
    )

    HEDGES: Tuple[str, ...] = (
        "maybe", "perhaps", "probably", "not sure", "unclear", "might", "could",
        "i think", "seems", "appears", "possibly", "hard to say", "depends",
    )
    WH_WORDS: Tuple[str, ...] = ("why", "what", "how", "when", "where", "which")

    def __init__(self) -> None:
        self.count = 0

    def build(
        self,
        query: str,
        response: str,
        meta: Optional[Dict] = None,
    ) -> Features:
        meta = meta or {}
        q = query.lower()
        r = response.lower()

        hedge_count = sum(q.count(h) + r.count(h) for h in self.HEDGES)
        uncertainty = float(meta.get("entropy", 0.0))
        if uncertainty <= 0.0:  # text-only proxy when no score is available
            uncertainty = min(1.0, hedge_count / 4.0)

        feats: Dict[str, float] = {
            "query_len": float(len(query)),
            "has_code": float(1.0 if (("```" in q) or any(k in q for k in ("def ", "import ", "error:", "traceback", "foo(", "->", "="))) else 0.0),
            "n_wh_words": float(sum(q.count(w) for w in self.WH_WORDS)),
            "n_qmarks": float(q.count("?")),
            "resp_len": float(len(response)),
            "hedge_count": float(hedge_count),
            "uncertainty": float(uncertainty),
        }
        self.count += 1
        return Features(feats)


# --------------------------------------------------------------------------------------
# Feedback abstraction layer
# --------------------------------------------------------------------------------------

class FeedbackAbstraction:
    """Turn observable interaction signals into a reward R in [-1, 1].

    The reward is what the tree learns to predict, so its quality is the ceiling on
    explanation quality. Sources (combined as a weighted average):

    * self-consistency: agreement between N sampled generations of the same query;
    * uncertainty: logprob-entropy proxy (low entropy -> more confident -> higher reward);
    * user rating: explicit -1/0/+1, which overrides/anchors the automatic estimate.
    """

    def __init__(
        self,
        w_consistency: float = 0.6,
        w_uncertainty: float = 0.2,
        w_user: float = 0.2,
    ) -> None:
        self.w_c = w_consistency
        self.w_u = w_uncertainty
        self.w_r = w_user

    def reward(
        self,
        meta: Optional[Dict] = None,
        user_rating: float = 0.0,
    ) -> float:
        meta = meta or {}
        consistency = float(meta.get("consistency", 0.5))
        entropy = float(meta.get("entropy", 0.0))

        r_consistency = 2.0 * consistency - 1.0                       # 0..1 -> -1..1
        r_uncertainty = -min(entropy, 1.0)                           # more entropy -> -1

        r = (self.w_c * r_consistency
             + self.w_u * r_uncertainty
             + self.w_r * float(user_rating))
        return max(-1.0, min(1.0, r))


# --------------------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------------------

class TSNode(nn.Module):
    """A node in the growing TreeSight tree.

    Holds:

    * a trainable ``readout`` (Linear K->1) predicting the feedback reward for its region
      (Phase B weights that "learn as users use the model");
    * decaying sufficient statistics (recency-weighted memory),
    * (leaf) a hard axis-aligned split -- feature + threshold -- chosen greedily;
    * drift-detection EMA tracks (fast vs slow reward mean);
    * a ``hint``: the best practical advice for the region, derived from its stats.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.in_features = in_features

        # Phase-B weights (learned online against reward R).
        # Zero-initialized: the tree starts predicting a neutral value and learns the
        # region's reward purely from observed traffic (residual-stream-style init).
        self.readout = nn.Linear(in_features, 1)
        nn.init.zeros_(self.readout.weight)
        nn.init.zeros_(self.readout.bias)
        self.router: Optional[nn.Linear] = None      # enabled only in soft-routing mode

        # Topology.
        self.left: Optional[TSNode] = None
        self.right: Optional[TSNode] = None

        # Hard split state (leaf default). split_feature == -1 means "no split yet".
        self.split_feature: int = -1
        self.threshold: float = 0.0
        self.is_soft: bool = False

        # Decaying sufficient statistics (recency-weighted memory).
        self.register_buffer("N", torch.tensor(0.0))
        self.register_buffer("sumX", torch.zeros(in_features))
        self.register_buffer("sumX2", torch.zeros(in_features))
        self.register_buffer("sumR", torch.tensor(0.0))
        self.register_buffer("sumR2", torch.tensor(0.0))

        # Drift detection (fast vs slow reward EMA) + bookkeeping.
        self.register_buffer("r_fast", torch.tensor(0.0))
        self.register_buffer("r_slow", torch.tensor(0.0))
        self.drifted: bool = False

        # Recent-member ring buffer used for split search in streaming mode
        # (online tree can only look at the recent batch otherwise).
        self.bufX: Optional[torch.Tensor] = None
        self.bufR: Optional[torch.Tensor] = None
        self.bufW: Optional[torch.Tensor] = None
        self.buf_cap: int = 64

        self.uid: str = f"n{id(self)}"
        self.created_at: int = 0
        self.depth: int = 0

    # --- topology ---------------------------------------------------------------
    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def make_internal(self, feature: int, threshold: float, soft: bool) -> None:
        """Turn this leaf into an internal routing node with the given hard split."""
        self.split_feature = feature
        self.threshold = threshold
        self.is_soft = soft
        if soft:
            self.router = nn.Linear(self.in_features, 1)

    # --- decayed statistics -----------------------------------------------------
    def accumulate(self, X: torch.Tensor, R: torch.Tensor, weights: torch.Tensor, decay: float) -> None:
        """Accumulate recency-weighted stats: ``s <- gamma*s + sum(w)``."""
        w = weights.unsqueeze(-1)
        self.N = decay * self.N + weights.sum()
        self.sumX = decay * self.sumX + (X * w).sum(dim=0)
        self.sumX2 = decay * self.sumX2 + ((X ** 2) * w).sum(dim=0)
        self.sumR = decay * self.sumR + (R * weights).sum()
        self.sumR2 = decay * self.sumR2 + ((R ** 2) * weights).sum()

    def push(self, X: torch.Tensor, R: torch.Tensor, weights: torch.Tensor) -> None:
        """Append current-batch members to the recent-member ring buffer."""
        if X.numel() == 0:
            return
        x, r, w = X.detach(), R.detach(), weights.detach()
        if self.bufX is None:
            self.bufX, self.bufR, self.bufW = x, r, w
        else:
            self.bufX = torch.cat([self.bufX, x])
            self.bufR = torch.cat([self.bufR, r])
            self.bufW = torch.cat([self.bufW, w])
        if self.bufX.size(0) > self.buf_cap:
            self.bufX = self.bufX[-self.buf_cap:]
            self.bufR = self.bufR[-self.buf_cap:]
            self.bufW = self.bufW[-self.buf_cap:]

    def moments(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """(mean_X, var_X, mean_R, var_R) from the decayed sufficient statistics."""
        if self.N <= 0:
            mX = torch.zeros(self.in_features)
            vX = torch.zeros(self.in_features)
            mR = torch.zeros(())
            vR = torch.zeros(())
        else:
            mX = self.sumX / self.N
            vX = (self.sumX2 / self.N - mX ** 2).clamp_min(0.0)
            mR = self.sumR / self.N
            vR = (self.sumR2 / self.N - mR ** 2).clamp_min(0.0)
        return mX, vX, mR, vR

    def update_drift(
        self,
        R: torch.Tensor,
        eta_fast: float = 0.1,
        eta_slow: float = 0.01,
        z_thresh: float = 2.0,
    ) -> None:
        """Update drift EMAs and flag the node if fast/slow reward diverge."""
        self.r_fast = eta_fast * R + (1.0 - eta_fast) * self.r_fast
        self.r_slow = eta_slow * R + (1.0 - eta_slow) * self.r_slow
        _, _, _, vR = self.moments()
        sigma = (vR.sqrt() + 1e-6).item()
        diff = (self.r_fast - self.r_slow).item()
        if sigma > 1e-4 and abs(diff) > z_thresh * sigma:
            self.drifted = True

    # --- explanation helpers ----------------------------------------------------
    def wilson(self, z: float = 1.96) -> Tuple[float, float]:
        """Wilson score interval on the leaf's mean reward (honest confidence)."""
        n = max(float(self.N), 1e-9)
        mu = float(self.sumR / n)
        p = (mu + 1.0) / 2.0                                   # map [-1,1] -> [0,1]
        denom = 1 + z ** 2 / n
        centre = (p + z ** 2 / (2 * n)) / denom
        margin = z * math.sqrt((p * (1 - p) / n) + z ** 2 / (4 * n ** 2)) / denom
        lo, hi = centre - margin, centre + margin
        return 2.0 * lo - 1.0, 2.0 * hi - 1.0

    def centroid(self) -> Dict[str, float]:
        mX, _, _, _ = self.moments()
        return {FeatureBridge.FEATURE_NAMES[i]: round(float(mX[i]), 3)
                for i in range(min(len(FeatureBridge.FEATURE_NAMES), self.in_features))}

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"TSNode({self.uid}, leaf={self.is_leaf()}, "
                f"split={self.split_feature}, N={float(self.N):.0f})")


# --------------------------------------------------------------------------------------
# The dynamically growing tree
# --------------------------------------------------------------------------------------

class TSTree(nn.Module):
    """The online-growing interpretability tree.

    Core behaviour:

    * ``observe_and_grow(x, r)`` -- Phase A: route, decay-accumulate stats, then split any
      leaf whose behavioural variance (``G``) is still high and whose support is sufficient.
    * ``learn(x, r)`` -- Phase B: update each leaf's readout weight toward R (flow weighted).
    * ``forward(x)`` -- sparse masked forward returning expected reward per token.
    * ``explain(x)`` -- returns the decision path and leaf membership/confidence per token.
    * ``knowledge_map()`` -- exports the full interpretable state as JSON-ready dict.
    """

    def __init__(
        self,
        in_features: int,
        decay: float = 0.99,
        growth_threshold: float = 0.35,
        min_samples: float = 20.0,
        min_reduction: float = 1e-3,
        alpha: float = 0.0,
        beta: float = 1.0,
        max_leaves: int = 512,
        soft_routing: bool = False,
        lr: float = 1e-3,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.decay = decay
        self.growth_threshold = growth_threshold
        self.min_samples = min_samples
        self.min_reduction = min_reduction
        self.alpha = alpha
        self.beta = beta
        self.max_leaves = max_leaves
        self.soft_routing = soft_routing
        self.lr = lr

        self.root = TSNode(in_features)
        self.root.uid = "root"
        self.all_nodes: nn.ModuleList = nn.ModuleList([self.root])
        self.creation_counter: int = 0
        self.total_queries: int = 0
        self.leaves_created: int = 0

        # Online-optimizer bookkeeping. Params are REGISTERED INCREMENTALLY:
        # children born in ``_bifurcate`` register their readout (+ router) params so the
        # optimizer's stored references always cover the current leaves. The optimizer is
        # rebuilt whenever the registered param set grows (drops momentum for the newcomers).
        self._train_params: List[torch.nn.Parameter] = []
        self._opt: Optional[torch.optim.Optimizer] = None
        self._pcount: int = 0
        self._register(self.root)

    def _node_learnable_params(self, node: TSNode) -> List[torch.nn.Parameter]:
        params = [p for p in node.readout.parameters() if p.requires_grad]
        if self.soft_routing and node.router is not None:
            params += [p for p in node.router.parameters() if p.requires_grad]
        return params

    def _register(self, *nodes: TSNode) -> None:
        known = {id(p) for p in self._train_params}
        for node in nodes:
            for p in self._node_learnable_params(node):
                if id(p) not in known:
                    self._train_params.append(p)
                    known.add(id(p))
        if self._train_params and len(self._train_params) != self._pcount:
            self._pcount = len(self._train_params)
            self._opt = None  # rebuild optimizer to cover the new leaves

    def _optimizer(self) -> torch.optim.Optimizer:
        if self._opt is None:
            self._opt = torch.optim.AdamW(self._train_params, lr=self.lr)
        return self._opt

    # ------------------------------------------------------------------ forward
    def forward(self, X: torch.Tensor, flow: Optional[Dict[str, float]] = None) -> torch.Tensor:
        """Expected reward prediction, sparse masked recursion (mirrors ddtree)."""
        N = X.size(0)
        return self._forward_recursive(self.root, X, torch.ones(N, device=X.device), flow)

    def _forward_recursive(
        self,
        node: TSNode,
        X: torch.Tensor,
        weights: torch.Tensor,
        flow: Optional[Dict[str, float]],
    ) -> torch.Tensor:
        if flow is not None:
            flow[node.uid] = flow.get(node.uid, 0.0) + float(weights.sum().detach())

        mask = weights > EPS
        if not mask.any():
            return torch.zeros(X.size(0), 1, device=X.device)

        Xa, wa = X[mask], weights[mask]
        out = torch.zeros(X.size(0), 1, device=X.device)

        if node.is_leaf():
            pred = node.readout(Xa) * wa.unsqueeze(-1)
            out[mask] = pred
            return out

        if node.is_soft and node.router is not None:
            pL = torch.sigmoid(node.router(Xa).squeeze(-1))
        else:
            left_mask = Xa[:, node.split_feature] <= node.threshold
            pL = left_mask.to(Xa.dtype)

        left_out = self._forward_recursive(node.left, Xa, wa * pL, flow)
        right_out = self._forward_recursive(node.right, Xa, wa * (1.0 - pL), flow)
        out[mask] = left_out + right_out
        return out

    # ------------------------------------------------------------------ Phase A
    @torch.no_grad()
    def observe_and_grow(self, X: torch.Tensor, R: torch.Tensor) -> List[TSNode]:
        """Decay-accumulate stats over the batch, then grow impure leaves. No grads."""
        self.total_queries += X.size(0)
        self._accumulate_recursive(self.root, X, R, torch.ones(X.size(0), device=X.device), decay=self.decay)

        _, vX_root, _, vR_root = self.root.moments()
        trace_root = vX_root.sum().clamp_min(0.0) + 1e-8
        var_root = vR_root.clamp_min(0.0) + 1e-8

        grown: List[TSNode] = []
        if len(self.all_nodes) >= self.max_leaves:
            return grown

        # Split search uses the recent-member ring buffers, so streaming (batch=1)
        # traffic still accumulates enough evidence to grow the tree.
        buffers: List[Tuple[TSNode, torch.Tensor, torch.Tensor]] = []
        self._collect_leaf_buffers(self.root, buffers)

        for node, xm, rm, _wm in buffers:
            if not node.is_leaf():
                continue
            if float(node.N) < self.min_samples or xm.size(0) < 4:
                continue
            _, vX, _, vR = node.moments()
            G = self.alpha * (vX.sum() / trace_root).item() + self.beta * (vR / var_root).item()
            if G <= self.growth_threshold:
                continue
            split = self._best_split(xm, rm)
            if split is None:
                continue
            feat, thr, reduction = split
            if reduction <= self.min_reduction:
                continue
            self._bifurcate(node, feat, thr)
            grown.append(node)
        return grown

    @torch.no_grad()
    def _collect_leaf_buffers(
        self,
        node: TSNode,
        out: List[Tuple[TSNode, torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> None:
        if node.is_leaf():
            if node.bufX is not None and node.bufX.size(0) > 0:
                out.append((node, node.bufX, node.bufR, node.bufW))
            return
        self._collect_leaf_buffers(node.left, out)
        self._collect_leaf_buffers(node.right, out)

    @torch.no_grad()
    def _accumulate_recursive(
        self,
        node: TSNode,
        X: torch.Tensor,
        R: torch.Tensor,
        weights: torch.Tensor,
        decay: float,
    ) -> None:
        mask = weights > EPS
        if not mask.any():
            return
        Xa, Ra, wa = X[mask], R[mask], weights[mask]
        node.accumulate(Xa, Ra, wa, decay)
        node.update_drift(Ra.mean())

        if node.is_leaf():
            node.push(Xa, Ra, wa)
            return
        if node.is_soft and node.router is not None:
            pL = torch.sigmoid(node.router(Xa).squeeze(-1))
        else:
            pL = (Xa[:, node.split_feature] <= node.threshold).to(Xa.dtype)
        self._accumulate_recursive(node.left, Xa, Ra, wa * pL, decay)
        self._accumulate_recursive(node.right, Xa, Ra, wa * (1.0 - pL), decay)

    @torch.no_grad()
    def _collect_leaves(
        self,
        node: TSNode,
        X: torch.Tensor,
        R: torch.Tensor,
        weights: torch.Tensor,
        out: List[Tuple[TSNode, torch.Tensor, torch.Tensor]],
    ) -> None:
        mask = weights > EPS
        if not mask.any():
            return
        Xa, Ra, wa = X[mask], R[mask], weights[mask]
        if node.is_leaf():
            out.append((node, Xa, Ra))
            return
        if node.is_soft and node.router is not None:
            pL = torch.sigmoid(node.router(Xa).squeeze(-1))
        else:
            pL = (Xa[:, node.split_feature] <= node.threshold).to(Xa.dtype)
        self._collect_leaves(node.left, Xa, Ra, wa * pL, out)
        self._collect_leaves(node.right, Xa, Ra, wa * (1.0 - pL), out)

    @torch.no_grad()
    def _best_split(
        self,
        X: torch.Tensor,
        R: torch.Tensor,
    ) -> Optional[Tuple[int, float, float]]:
        """Greedy axis-aligned split: feature & mean-threshold maxing reward variance red'n."""
        if X.size(0) < 2:
            return None
        base = R.var(unbiased=False).item()
        best: Optional[Tuple[int, float, float]] = None
        for f in range(self.in_features):
            thr = X[:, f].mean().item()
            left = X[:, f] <= thr
            frac = left.float().mean().item()
            if not (0.05 < frac < 0.95):
                continue
            if left.all() or (~left).all() or left.sum() < 2 or (~left).sum() < 2:
                continue
            rl = R[left].var(unbiased=False).item()
            rr = R[~left].var(unbiased=False).item()
            red = base - (frac * rl + (1.0 - frac) * rr)
            if best is None or red > best[2]:
                best = (f, thr, red)
        return best

    @torch.no_grad()
    def _bifurcate(self, node: TSNode, feature: int, threshold: float) -> None:
        """Create children inheriting the parent readout; register in optimizer."""
        self.creation_counter += 1
        birth = self.creation_counter
        left_child = TSNode(self.in_features)
        right_child = TSNode(self.in_features)
        left_child.created_at = birth
        right_child.created_at = birth
        left_child.depth = node.depth + 1
        right_child.depth = node.depth + 1
        left_child.uid = f"n{len(self.all_nodes)}"
        right_child.uid = f"nn{len(self.all_nodes) + 1}"

        # Knowledge inheritance (the tree's "unified K/V" trick): children start with the
        # parent's readout so output does not jump and gradients stay smooth.
        left_child.readout.weight.copy_(node.readout.weight)
        left_child.readout.bias.copy_(node.readout.bias)
        right_child.readout.weight.copy_(node.readout.weight)
        right_child.readout.bias.copy_(node.readout.bias)
        if self.soft_routing:
            left_child.readout.requires_grad_(True)
            right_child.readout.requires_grad_(True)

        node.make_internal(feature, threshold, soft=self.soft_routing)
        node.left = left_child
        node.right = right_child
        self.all_nodes.append(left_child)
        self.all_nodes.append(right_child)
        self.leaves_created += 1
        self._register(left_child, right_child)   # born today -> learn from today

    # ------------------------------------------------------------------ Phase B
    def learn(self, X: torch.Tensor, R: torch.Tensor) -> float:
        """Update leaf readout weights toward R (autograd); base model never involved."""
        if not self._train_params:
            return 0.0
        opt = self._optimizer()
        opt.zero_grad()

        leaves: List[Tuple[TSNode, torch.Tensor, torch.Tensor]] = []
        with torch.no_grad():
            self._collect_leaves(self.root, X, R, torch.ones(X.size(0), device=X.device), leaves)

        total = 0.0
        alive = 0
        for node, xm, rm in leaves:
            if xm.numel() == 0:
                continue
            pred = node.readout(xm).squeeze(-1)
            total += F.mse_loss(pred, rm).sum().item()
            alive += 1
        if alive == 0:
            return 0.0
        loss = total / X.size(0)

        loss_b = torch.tensor(0.0)
        for node, xm, rm in leaves:
            if xm.numel() == 0:
                continue
            pred = node.readout(xm).squeeze(-1)
            loss_b = loss_b + F.mse_loss(pred, rm).sum()
        loss_b = loss_b / X.size(0)
        loss_b.backward()
        torch.nn.utils.clip_grad_norm_(self._train_params, 1.0)
        opt.step()
        return loss

    # ------------------------------------------------------------------ explain
    def explain(self, X: torch.Tensor) -> Dict:
        """Decision path + leaf stats for a single (row) feature vector."""
        x = X.unsqueeze(0)
        path: List[Dict] = []
        node = self.root
        vis_nodes: List[TSNode] = []

        while not node.is_leaf():
            ch = node.left
            if node.is_soft and node.router is not None:
                p = torch.sigmoid(node.router(x).squeeze(-1)).item()
                go_left = p >= 0.5
                path.append({"node": node.uid, "type": "soft", "p_left": round(p, 4)})
            else:
                fname = FeatureBridge.FEATURE_NAMES[node.split_feature]
                val = x[0, node.split_feature].item()
                path.append({"node": node.uid, "split": f"{fname} <={node.threshold:.3f}",
                             "value": round(val, 3), "threshold": round(node.threshold, 3)})
                go_left = val <= node.threshold
            node = ch if go_left else node.right
            vis_nodes.append(node)

        lo, hi = node.wilson()
        mX, _, mR, vR = node.moments()
        return {
            "path": path,
            "leaf": {
                "uid": node.uid,
                "depth": node.depth,
                "N_eff": round(float(node.N), 1),
                "mean_r": round(float(mR), 3),
                "var_r": round(float(vR), 3),
                "wilson": [round(lo, 3), round(hi, 3)],
                "centroid": node.centroid(),
                "drifted": bool(node.drifted),
                "hint": hint_for(node),
            },
        }

    # ------------------------------------------------------------------ knowledge map
    def knowledge_map(self) -> Dict:
        leaves: List[TSNode] = []
        self._collect_all_leaves(self.root, leaves)
        out_leaves = []
        for node in leaves:
            lo, hi = node.wilson()
            mX, _, mR, vR = node.moments()
            out_leaves.append({
                "uid": node.uid,
                "depth": node.depth,
                "path": self._path_string(node),
                "N_eff": round(float(node.N), 1),
                "mean_r": round(float(mR), 3),
                "var_r": round(float(vR), 3),
                "wilson": [round(lo, 3), round(hi, 3)],
                "centroid": node.centroid(),
                "drifted": bool(node.drifted),
                "hint": hint_for(node),
            })
        out_leaves.sort(key=lambda d: -d["N_eff"])

        mX, _, mR, vR = self.root.moments()
        drift = [
            {"uid": n.uid, "fast": round(float(n.r_fast), 3), "slow": round(float(n.r_slow), 3)}
            for n in self.all_nodes if n.drifted
        ]
        return {
            "total_queries": self.total_queries,
            "nodes": len(self.all_nodes),
            "leaves": len(leaves),
            "depth": self.depth(),
            "root_stats": {"N_eff": round(float(self.root.N), 1),
                           "mean_r": round(float(mR), 3), "var_r": round(float(vR), 3)},
            "coverage": round(len([l for l in leaves if float(l.N) > 1]) / max(len(leaves), 1), 3),
            "leaves_created": self.leaves_created,
            "drifted_nodes": drift,
            "leaves": out_leaves,
        }

    @torch.no_grad()
    def _collect_all_leaves(self, node: TSNode, out: List[TSNode]) -> None:
        if node.is_leaf():
            out.append(node)
            return
        self._collect_all_leaves(node.left, out)
        self._collect_all_leaves(node.right, out)

    def _path_string(self, target: TSNode) -> str:
        node = self.root
        parts: List[str] = []
        while node != target and not node.is_leaf():
            if node.is_soft and node.router is not None:
                parts.append(f"{node.uid}[soft]")
                node = node.left  # ambiguous; hard mode is the default explanation path
                continue
            fname = FeatureBridge.FEATURE_NAMES[node.split_feature]
            # determine side by centroid side
            mX, _, _, _ = target.moments()
            if mX[node.split_feature].item() <= node.threshold:
                parts.append(f"{fname}<={node.threshold:.2f}")
                node = node.left
            else:
                parts.append(f"{fname}>{node.threshold:.2f}")
                node = node.right
        return " -> ".join(parts)

    def leaves(self) -> List[TSNode]:
        out: List[TSNode] = []
        self._collect_all_leaves(self.root, out)
        return out

    def depth(self) -> int:
        def h(n: TSNode) -> int:
            return 1 if n.is_leaf() else 1 + max(h(n.left), h(n.right))
        return h(self.root)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# --------------------------------------------------------------------------------------
# Hint generation
# --------------------------------------------------------------------------------------

_HINT_TEMPLATES = {
    "reliable": "Region is historically reliable (n={n}, R={r}). Ask for the {thing} first.",
    "unreliable": "Region is unreliable (n={n}, R={r}). Suggest reformulating or requesting more detail.",
    "uncertain": "Region shows hedging/uncertainty (n={n}). Ask the user to clarify scope.",
}


def hint_for(node: TSNode) -> str:
    lo, hi = node.wilson()
    r = float(node.sumR / max(float(node.N), 1e-9))
    n = int(float(node.N))
    if lo > 0.2:
        return _HINT_TEMPLATES["reliable"].format(n=n, r=round(r, 2), thing="stack trace/detail")
    if hi < 0.0:
        return _HINT_TEMPLATES["unreliable"].format(n=n, r=round(r, 2))
    return _HINT_TEMPLATES["uncertain"].format(n=n)


# --------------------------------------------------------------------------------------
# The attachable sidecar
# --------------------------------------------------------------------------------------

class TreeSight:
    """Attach to any callable base model and explain/learn from its live inference.

    Parameters
    ----------
    base : callable
        ``(query: str, **kw) -> (response: str, meta: dict)``. ``meta`` may contain
        ``consistency`` and/or ``entropy`` used by the feedback abstraction.
    bridge, feedback : feature-extraction / reward layers.
    """

    def __init__(
        self,
        base: Callable[[str], Tuple[str, Dict]],
        bridge: Optional[FeatureBridge] = None,
        feedback: Optional[FeedbackAbstraction] = None,
        tree: Optional[TSTree] = None,
    ) -> None:
        self.base = base
        self.bridge = bridge or FeatureBridge()
        self.feedback = feedback or FeedbackAbstraction()
        self.tree = tree or TSTree(in_features=len(FeatureBridge.FEATURE_NAMES))
        self.attached = True
        self._trace: List[Dict] = []

    # ------------------------------------------------------------------ main API
    def run(
        self,
        query: str,
        user_rating: float = 0.0,
        steer: bool = True,
    ) -> Dict:
        """Full cycle: predict -> feature -> reward -> grow -> learn -> explain -> steer.

        Returns a dict with the response, the explanation, and (optionally) the steered
        context that was injected into the base model.
        """
        # Predict with the (unchanged) base model.
        response, meta = self.base(query)

        # Optional steer-back: retrieves the leaf hint and re-runs with added context.
        steered_response = response
        steered_meta = meta
        if steer and self.tree.total_queries > 0:
            feats = self.bridge.build(query, response, meta)
            x = feats.tensor()
            hint = self.tree.explain(x)["leaf"]["hint"]
            steered_response, steered_meta = self.base(query, hint=hint)

        feats_final = self.bridge.build(query, steered_response, steered_meta)
        x = feats_final.tensor()
        r = self.feedback.reward(steered_meta, user_rating)

        # Phase A + Phase B: the tree learns from this live inference.
        grown = self.tree.observe_and_grow(x.unsqueeze(0), torch.tensor([r]))
        loss = self.tree.learn(x.unsqueeze(0), torch.tensor([r]))

        explanation = self.tree.explain(x)
        record = {
            "query": query,
            "response": steered_response,
            "reward": round(r, 3),
            "meta": {k: v for k, v in steered_meta.items() if k != "logprobs"},
            "grown": [n.uid for n in grown] if isinstance(grown, list) else grown,
            "path": explanation["path"],
            "leaf": explanation["leaf"],
        }
        self._trace.append(record)
        return {
            "response": steered_response,
            "explanation": explanation,
            "grown_nodes": [n.uid for n in grown],
            "learn_loss": round(loss, 5),
            "trace": record,
        }

    # ------------------------------------------------------------------ exports
    def knowledge_map(self, pretty: bool = False) -> str:
        km = self.tree.knowledge_map()
        return json.dumps(km, indent=2) if pretty else json.dumps(km)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str) -> None:
        """Persist the whole learned tree (topology + stats + readout weights).

        The binary format is torch-native; loading restores the tree exactly so it keeps
        learning across restarts. The base model is never serialized.
        """
        self.tree._opt = None   # optimizer state is scratch, not knowledge
        torch.save(self.tree, path)

    @staticmethod
    def load(path: str, base: Callable[[str], Tuple[str, Dict]],
             bridge: Optional[FeatureBridge] = None,
             feedback: Optional[FeedbackAbstraction] = None) -> "TreeSight":
        """Rehydrate a TreeSight + saved tree, re-registering trainable params."""
        tree = torch.load(path, weights_only=False)
        TreeSight._reindex(tree)
        return TreeSight(base=base, bridge=bridge, feedback=feedback, tree=tree)

    @staticmethod
    def _reindex(tree: "TSTree") -> None:
        """Rebuild the incremental-param bookkeeping after deserialization."""
        tree._train_params = []
        tree._opt = None
        tree._pcount = 0
        for node in tree.all_nodes:
            tree._register(node)

    def trace(self, last: Optional[int] = None) -> Dict:
        return {"events": self._trace[-last:]} if last else {"events": self._trace}


# --------------------------------------------------------------------------------------
# Mock "existing model" for zero-dependency dry runs
# --------------------------------------------------------------------------------------

class MockBase:
    """A stand-in for ANY pretrained model: its 'uncertainty' is topic-dependent.

    This lets you watch the sidecar discover behavior regions even without a real model
    on disk. Meta carries ``consistency`` and ``entropy`` so FeedbackAbstraction works.
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = torch.Generator().manual_seed(seed)

    def _region(self, query: str) -> str:
        q = query.lower()
        if any(k in q for k in ("def ", "import ", "error:", "traceback", "->", "```", "foo(")):
            return "code"
        if q.endswith("?") or any(w in q for w in ("why", "what", "how")):
            return "ambiguous"
        return "factual"

    def __call__(self, query: str, hint: Optional[str] = None) -> Tuple[str, Dict]:
        region = self._region(query)
        if region == "code":
            consistency = 0.35 + 0.2 * torch.rand(1, generator=self.rng).item()
            entropy = 0.7
            body = ("I'm not entirely sure about the exact fix. It might be a type issue, "
                    "perhaps in the caller. Could you share the full traceback?")
        elif region == "ambiguous":
            consistency = 0.55 + 0.15 * torch.rand(1, generator=self.rng).item()
            entropy = 0.5
            body = ("It depends on what you mean. If you mean X, then yes; if you mean Y, "
                    "probably not. Could you clarify?")
        else:
            consistency = 0.85 + 0.1 * torch.rand(1, generator=self.rng).item()
            entropy = 0.15
            body = "The answer is the standard definition: it is a well-established fact in the literature."
        if hint:
            body = f"{body} [context: {hint}]"
        meta = {
            "consistency": float(consistency),
            "entropy": float(entropy),
            "region": region,
        }
        return body, meta


# --------------------------------------------------------------------------------------
# Demo / CLI
# --------------------------------------------------------------------------------------

import torch.nn.functional as F  # noqa: E402  (placed here to keep imports grouped)


def _show(events: Dict) -> None:
    e = events["trace"]
    print(f"\nQ: {e['query'][:52]}")
    print(f"   R={e['reward']:+.2f}  path={[s.get('split') or s.get('type') for s in e['path']] or []}  "
          f"leaf={e['leaf']['uid']} (n={e['leaf']['N_eff']}, r={e['leaf']['mean_r']:+.2f})")
    for uid in e["grown"]:
        print(f"   SPLIT -> new leaf {uid}")


def demo(
    queries: Sequence[str],
    real_feedback: bool,
    soft_routing: bool,
    seed: int = 0,
) -> None:
    torch.manual_seed(seed)
    base = MockBase(seed=seed)
    tree = TSTree(
        in_features=len(FeatureBridge.FEATURE_NAMES),
        min_samples=4.0,
        growth_threshold=0.25 if soft_routing else 0.35,
        soft_routing=soft_routing,
    )
    sidecar = TreeSight(base, tree=tree)

    print("TreeSight-Live | attach-on-inference continual interpretability")
    print(f"  routing={'soft' if soft_routing else 'hard-axis'}  real_feedback={real_feedback}\n")

    for i, q in enumerate(queries):
        rating = 0.0
        if real_feedback:
            rating = float(input(f"  rate [{q[:40]}] (-1/0/+1): ") or 0.0)
        out = sidecar.run(q, user_rating=rating)
        _show(out)

    print("\n" + "=" * 72)
    print("KNOWLEDGE MAP  (what the sidecar learned from live inference)")
    print("=" * 72)
    print(sidecar.knowledge_map(pretty=True))

    print("\nDrift report:", sidecar.tree.knowledge_map()["drifted_nodes"])

    # Demo of steer-back: re-ask a code query and show the retrieved hint.
    code_q = "fix this error: TypeError: 'NoneType' object is not subscriptable"
    out = sidecar.run(code_q, steer=True)
    print(f"\nSteer-back demo on: {code_q}")
    print(f"  hint injected -> leaf: {out['explanation']['leaf']['hint']}")
    print(f"  trace length : {len(sidecar._trace)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="TreeSight-Live demo (mock base, no deps).")
    ap.add_argument("--queries", type=int, default=24, help="number of mock queries")
    ap.add_argument("--real-feedback", action="store_true",
                    help="ask for -1/0/+1 per query instead of auto reward")
    ap.add_argument("--soft-routing", action="store_true",
                    help="use sigmoid routers that learn (adamw) instead of hard axis splits")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    queries: List[str] = (
        ["what is transformer?"] * 3
        + ["def f(): return x", "why did the test fail?", "how does attention work?"]
        + ["error: name 'y' is not defined", "what is the capital of France?"]
        + ["`a = b`", "why ?", "how to fix memory leak in torch?"]
    )
    while len(queries) < args.queries:
        queries = queries + queries[: args.queries - len(queries)]
    demo(queries[: args.queries], real_feedback=args.real_feedback,
         soft_routing=args.soft_routing, seed=args.seed)


if __name__ == "__main__":
    main()