"""Core dynamic soft decision tree implementation.

Implements the architecture formalized in ``.agents/docs/docs-0.md`` (first-principles
mathematics) and ``.agents/docs/docs-1.md`` (PyTorch two-phase paradigm):

* A ``DynamicNode`` is an independent ``nn.Module`` holding routing parameters, leaf
  parameters, topological pointers and running sufficient statistics.
* A ``DynamicSoftTree`` performs a *sparse, masked* forward pass in ``O(active_nodes)``
  and grows new nodes only when a measure-theoretic purity criterion is violated.

The two-phase paradigm (growth phase with ``no_grad`` + training phase with Autograd)
guarantees gradient flow through the newly created topology.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

# Probability mass below this threshold is pruned. This neutralizes the "product
# trap" (vanishing path probabilities) by truncating flow before machine epsilon.
EPS: float = 1e-5


class DynamicNode(nn.Module):
    """A single node in the dynamic tree.

    Every node is an independent computational entity maintaining its own local state
    (``theta_v`` from docs-0.md, section 1.1). Inheriting from ``nn.Module`` ensures the
    routing and leaf parameters are tracked by PyTorch's Autograd.
    """

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Routing parameters (used only when the node is internal).
        self.route_linear = nn.Linear(in_features, 1)

        # Leaf parameters (used only when the node is a leaf).
        self.leaf_linear = nn.Linear(in_features, out_features)

        # Topological pointers: empty children => this node is a leaf (v in L).
        self.left: Optional[DynamicNode] = None
        self.right: Optional[DynamicNode] = None

        # Creation bookkeeping for visualization: an opaque id, the birth counter
        # (from the owning tree), and a flag cleared once the node has been seen by a
        # visualizer after its creation.
        self.uid: str = f"n{id(self)}"
        self.created_at: int = 0
        self.is_new: bool = True

        # Sufficient statistics (empirical measure accumulators), not optimizer-updated.
        self.register_buffer("N", torch.tensor(0.0))
        self.register_buffer("sum_X", torch.zeros(in_features))
        self.register_buffer("sum_X2", torch.zeros(in_features))
        self.register_buffer("sum_Y", torch.zeros(out_features))
        self.register_buffer("sum_Y2", torch.zeros(out_features))

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def reset_stats(self) -> None:
        self.N.zero_()
        self.sum_X.zero_()
        self.sum_X2.zero_()
        self.sum_Y.zero_()
        self.sum_Y2.zero_()

    def accumulate_stats(self, X: torch.Tensor, Y: torch.Tensor, weights: torch.Tensor) -> None:
        """Accumulate weighted sufficient statistics for the growth criterion."""
        w = weights.unsqueeze(-1)
        self.N += weights.sum()
        self.sum_X += (X * w).sum(dim=0)
        self.sum_X2 += ((X**2) * w).sum(dim=0)
        self.sum_Y += (Y * w).sum(dim=0)
        self.sum_Y2 += ((Y**2) * w).sum(dim=0)

    def local_stats(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (mean_X, var_X, mean_Y, var_Y) from the accumulated statistics."""
        if self.N <= 0:
            mean_X = torch.zeros_like(self.sum_X)
            var_X = torch.zeros_like(self.sum_X2)
            mean_Y = torch.zeros_like(self.sum_Y)
            var_Y = torch.zeros_like(self.sum_Y2)
        else:
            mean_X = self.sum_X / self.N
            var_X = (self.sum_X2 / self.N) - (mean_X**2)
            mean_Y = self.sum_Y / self.N
            var_Y = (self.sum_Y2 / self.N) - (mean_Y**2)
        return mean_X, var_X, mean_Y, var_Y

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"DynamicNode(in={self.in_features}, out={self.out_features}, "
            f"leaf={self.is_leaf()})"
        )


class DynamicSoftTree(nn.Module):
    """The dynamic soft decision tree.

    Manages the dynamically-growing topology and orchestrates the sparse, masked forward
    pass. Instantiating the tree creates only the root; further nodes are registered in
    ``self.all_nodes`` (an ``nn.ModuleList`` so the optimizer discovers them) as the tree
    bifurcates.
    """

    def __init__(self, in_features: int, out_features: int = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.root = DynamicNode(in_features, out_features)
        self.root.is_new = False
        # ModuleList registration ensures dynamically-added nodes are tracked.
        self.all_nodes: nn.ModuleList = nn.ModuleList([self.root])
        # Monotonic counter recording the order/batch in which nodes are created; used
        # by the visualizer to highlight newly created nodes.
        self.creation_counter: int = 0

    def reset_creation_state(self) -> None:
        """Re-mark every node as newly created (useful to restart observations)."""
        for node in self.all_nodes:
            node.is_new = True

    # ------------------------------------------------------------------
    # Forward (with Autograd graph)
    # ------------------------------------------------------------------
    def forward(self, X: torch.Tensor, flow: Optional[dict] = None) -> torch.Tensor:
        """Sparse, masked forward pass returning predictions of shape (N, out_features).

        When ``flow`` is a mutable dict keyed by node ``uid``, the total probability mass
        entering each node (sum over the batch) is accumulated into it. This lets the
        visualizer heat-map "how much data flows through each node" without rebuilding a
        separate graph. Passing ``flow`` does not alter the autograd graph.
        """
        batch_size = X.size(0)
        weights = torch.ones(batch_size, device=X.device)
        return self._forward_recursive(self.root, X, weights, flow=flow)

    def _forward_recursive(
        self,
        node: DynamicNode,
        X: torch.Tensor,
        weights: torch.Tensor,
        flow: Optional[dict] = None,
    ) -> torch.Tensor:
        """Recursively route data, masking to keep O(active) compute and a clean graph."""
        if flow is not None:
            flow[node.uid] = flow.get(node.uid, 0.0) + float(weights.sum().detach())

        mask = weights > EPS
        if not mask.any():
            return torch.zeros(X.size(0), self.out_features, device=X.device)

        X_act = X[mask]
        w_act = weights[mask]

        if node.is_leaf():
            preds = node.leaf_linear(X_act) * w_act.unsqueeze(-1)
            out = torch.zeros(X.size(0), self.out_features, device=X.device)
            out[mask] = preds
            return out

        logits = node.route_linear(X_act).squeeze(-1)
        p_left = torch.sigmoid(logits)
        p_right = 1.0 - p_left

        left_out = self._forward_recursive(node.left, X_act, w_act * p_left, flow=flow)
        right_out = self._forward_recursive(node.right, X_act, w_act * p_right, flow=flow)
        combined = left_out + right_out

        out = torch.zeros(X.size(0), self.out_features, device=X.device)
        out[mask] = combined
        return out

    # ------------------------------------------------------------------
    # Phase 1: evaluate statistics and grow (no autograd)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate_and_grow(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        growth_threshold: float,
        min_samples: float = 10,
        alpha: float = 0.5,
        beta: float = 0.5,
        root_stats: Optional[dict] = None,
    ) -> List[DynamicNode]:
        """Reset statistics, accumulate flow over the current topology, then bifurcate
        any impure leaf.

        Parameters
        ----------
        X : (N, in_features) input tensor.
        Y : (N, out_features) target tensor.
        growth_threshold : strict purity threshold ``tau`` in (0, 1). A leaf splits when
            ``G(v) > tau`` (docs-0.md, section 3.2).
        min_samples : minimum effective sample count required before a split is allowed.
        alpha, beta : weighting hyperparameters (``G(v) = alpha*norm_X + beta*norm_Y``).
        root_stats : optional scaling stats for the scale-invariant criterion. When
            ``None``, the root's own accumulated statistics are used.

        Returns the list of nodes that were split during this pass.
        """
        for node in self.all_nodes:
            node.reset_stats()

        batch_size = X.size(0)
        weights = torch.ones(batch_size, device=X.device)
        self._accumulate_recursive(self.root, X, Y, weights)

        # Normalization statistics, scale-invariant against the root (docs-0, section 3.1).
        if root_stats is None:
            _, var_root_X, _, var_root_Y = self.root.local_stats()
            root_trace = var_root_X.sum().clamp_min(0.0)
            root_var_Y = var_root_Y.sum().clamp_min(0.0)
        else:
            root_trace = root_stats.get("trace_X", 1.0)
            root_var_Y = root_stats.get("var_Y", 1.0)

        nodes_to_split: List[DynamicNode] = []
        for node in self.all_nodes:
            if not node.is_leaf():
                continue
            if node.N < min_samples:
                continue

            _, var_X, _, var_Y = node.local_stats()
            trace_cov_X = var_X.sum().clamp_min(0.0)
            var_Y_sum = var_Y.sum().clamp_min(0.0)

            norm_X = trace_cov_X / (root_trace + 1e-8)
            norm_Y = var_Y_sum / (root_var_Y + 1e-8)
            growth_score = alpha * norm_X + beta * norm_Y

            if growth_score > growth_threshold:
                nodes_to_split.append(node)

        split_nodes: List[DynamicNode] = []
        for node in nodes_to_split:
            mean_X, var_X, _, _ = node.local_stats()
            self._bifurcate_node(node, mean_X, var_X)
            split_nodes.append(node)

        return split_nodes

    def _accumulate_recursive(
        self, node: DynamicNode, X: torch.Tensor, Y: torch.Tensor, weights: torch.Tensor
    ) -> None:
        """Route data and accumulate statistics without building an autograd graph."""
        mask = weights > EPS
        if not mask.any():
            return

        X_act = X[mask]
        Y_act = Y[mask]
        w_act = weights[mask]

        if node.is_leaf():
            node.accumulate_stats(X_act, Y_act, w_act)
            return

        logits = node.route_linear(X_act).squeeze(-1)
        p_left = torch.sigmoid(logits)
        self._accumulate_recursive(node.left, X_act, Y_act, w_act * p_left)
        self._accumulate_recursive(node.right, X_act, Y_act, w_act * (1.0 - p_left))

    def _bifurcate_node(
        self, node: DynamicNode, mean_X: torch.Tensor, var_X: torch.Tensor
    ) -> None:
        """Create children, initialize the routing hyperplane orthogonal to the direction
        of maximum local variance, and inherit leaf knowledge (docs-0, section 4)."""
        # 1. Data-driven fault line: align normal with max-variance direction.
        w_route = var_X / (torch.norm(var_X) + 1e-8)
        b_route = -torch.dot(w_route, mean_X)

        # 2. Create children.
        self.creation_counter += 1
        birth = self.creation_counter
        left_child = DynamicNode(self.in_features, self.out_features)
        right_child = DynamicNode(self.in_features, self.out_features)
        left_child.created_at = birth
        right_child.created_at = birth
        left_child.uid = f"n{len(self.all_nodes)}"
        right_child.uid = f"nn{len(self.all_nodes) + 1}"
        left_child.is_new = True
        right_child.is_new = True
        node.is_new = False

        # 3. Configure parent routing + inherit leaf knowledge (prevents catastrophic
        #    forgetfulness and keeps gradients smooth).
        with torch.no_grad():
            node.route_linear.weight.copy_(w_route.unsqueeze(0))
            node.route_linear.bias.copy_(b_route.unsqueeze(0))
            left_child.leaf_linear.weight.copy_(node.leaf_linear.weight)
            left_child.leaf_linear.bias.copy_(node.leaf_linear.bias)
            right_child.leaf_linear.weight.copy_(node.leaf_linear.weight)
            right_child.leaf_linear.bias.copy_(node.leaf_linear.bias)

        # 4. Update topology.
        node.left = left_child
        node.right = right_child

        # 5. Register new nodes so the optimizer can see their parameters.
        self.all_nodes.append(left_child)
        self.all_nodes.append(right_child)

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------
    def leaves(self) -> List[DynamicNode]:
        """Return the current leaves of the tree in traversal order."""
        result: List[DynamicNode] = []

        def visit(node: DynamicNode) -> None:
            if node.is_leaf():
                result.append(node)
            else:
                visit(node.left)
                visit(node.right)

        visit(self.root)
        return result

    def depth(self) -> int:
        """Return the current maximum depth of the tree (root = 1)."""

        def h(node: DynamicNode) -> int:
            if node.is_leaf():
                return 1
            return 1 + max(h(node.left), h(node.right))

        return h(self.root)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def traverse(self) -> Sequence[DynamicNode]:
        """Return all nodes (pre-order) including the root and every created child."""
        return list(self.all_nodes)




"""Real-time visualization of the dynamic soft decision tree.

Builds a Graphviz (``pygraphviz``) digraph reflecting the *current* topology of a
:class:`softtree.core.DynamicSoftTree` and renders it to PNG. It is designed to be driven
from the training loop so you can watch, in real time, how probability mass flows through
the tree and how new nodes are created when a leaf bifurcates:

* **Internal nodes** (which route data) are drawn as orange diamonds.
* **Leaves** (which predict) are drawn as blue ellipses.
* Node fill intensity ("heat") reflects the total probability mass flowing through the
  node for the most recent batch — hotter = more data.
* **Newly created nodes** are drawn with a thick red border, and the node whose leaf
  params were just inherited is annotated, so you can see exactly where growth happened.

Dependencies
------------
This module uses ``pygraphviz`` (bundled with a working ``dot``) and falls back to
documenting itself if it is unavailable. It is intentionally separate from the core so the
model itself never requires a graph library.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from softtree.core import DynamicNode, DynamicSoftTree

try:  # pragma: no cover - optional dependency
    import pygraphviz as pgv
    _HAS_GRAPHVIZ = True
except Exception:  # pragma: no cover - optional dependency
    pgv = None  # type: ignore[assignment]
    _HAS_GRAPHVIZ = False

# Node styling ---------------------------------------------------------------
_NODE_KW = {"fontname": "Helvetica", "fontsize": "10", "style": "filled"}
_INTERNAL = {"shape": "diamond", "fillcolor": "#ffd27f", "color": "#b57a00"}
_LEAF = {"shape": "ellipse", "fillcolor": "#9ecbff", "color": "#2058a8"}
_NEW_BORDER = {"color": "#ff1a1a", "penwidth": "4.0"}
_EDGE = {"color": "#555555", "fontsize": "8"}


def _hm9(value: float) -> str:
    """Map a value in [0, 1] to one of 9 sequential blue color stops."""
    stops = [
        "#e8f1fb", "#c9dff6", "#a9cdf1", "#89bbeb", "#68a8e6",
        "#4796e0", "#2a7fc7", "#1f66a3", "#1a4f80",
    ]
    v = min(max(value, 0.0), 0.999)
    return stops[int(v * len(stops))]


def _node_label(node: DynamicNode, optional: Optional[Dict[str, str]] = None) -> str:
    label = [
        node.uid,
        f"leaf={int(node.is_leaf())}",
        f"n={node.N:.0f}",
    ]
    if optional:
        for key, value in optional.items():
            label.append(f"{key}={value}")
    return "\\n".join(label)


def build_graph(
    tree: DynamicSoftTree,
    flow: Optional[Dict[str, float]] = None,
    drop_new_flags: bool = True,
    node_extra: Optional[Dict[str, Dict[str, str]]] = None,
) -> "pgv.AGraph":
    """Build a ``pygraphviz`` digraph from the tree's current topology.

    Parameters
    ----------
    tree : the tree to render.
    flow : optional dict mapping node ``uid`` -> total probability mass (as captured by
        ``model.forward(X, flow=...)``). Used to heat-map node fill colour.
    drop_new_flags : whether to clear each node's ``is_new`` flag after rendering so the
        next render only highlights nodes created in the meantime.
    node_extra : optional mapping of node ``uid`` -> extra ``key=value`` annotations to
        show under each node label.

    Returns the graph object (call ``.draw(path)`` or ``g.to_string()`` to use it).
    """
    if not _HAS_GRAPHVIZ:  # pragma: no cover - optional dependency
        raise ImportError(
            "pygraphviz is required for visualization. Install it (e.g. via the "
            "'aisystem' conda env) before using this module."
        )

    g = pgv.AGraph(directed=True, strict=True)
    g.graph_attr.update(rankdir="TB", nodesep="0.45", ranksep="0.55", splines="true")
    g.node_attr.update(_NODE_KW)
    g.edge_attr.update(_EDGE)

    node_extra = node_extra or {}
    max_flow = max(flow.values()) if flow else None

    def add_node(node: DynamicNode) -> None:
        style = _LEAF if node.is_leaf() else _INTERNAL
        attrs = dict(style)

        if flow is not None and max_flow:
            attrs["fillcolor"] = _hm9(flow.get(node.uid, 0.0) / max_flow)

        if node.is_new:
            attrs.update(_NEW_BORDER)

        extra = node_extra.get(node.uid)
        g.add_node(
            node.uid,
            label=_node_label(node, optional=extra),
            **attrs,
        )

    def visit(node: DynamicNode) -> None:
        add_node(node)
        if node.is_leaf():
            return
        g.add_edge(node.uid, node.left.uid, label="L")
        g.add_edge(node.uid, node.right.uid, label="R")
        visit(node.left)
        visit(node.right)

    visit(tree.root)

    if drop_new_flags:
        for node in tree.all_nodes:
            node.is_new = False

    return g


def render_tree(
    tree: DynamicSoftTree,
    out_path: str,
    flow: Optional[Dict[str, float]] = None,
    node_extra: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Build the graph for ``tree`` and render it to ``out_path`` (e.g. PNG).

    Returns the absolute output path. Creates parent directories as needed.
    """
    g = build_graph(tree, flow=flow, drop_new_flags=True, node_extra=node_extra)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    g.draw(out_path, prog="dot", format=out_path.rsplit(".", 1)[-1])
    g.close()
    return os.path.abspath(out_path)


class TreeVisualizer:
    """Stateful helper that renders the tree after each interceptable event.

    Use it either directly (``viz = TreeVisualizer(dir); viz(tree)``) or pass its
    ``__call__`` as the ``on_batch`` hook of :func:`softtree.optimizers.train.train_dynamic_tree`.
    """

    def __init__(
        self,
        out_dir: str = "visualizations",
        batch_X: Optional[callable] = None,
        format: str = "png",  # noqa: A002 - name kept for clarity
    ):
        self.out_dir = out_dir
        self.format = format.lstrip(".")
        self.batch_X = batch_X
        self.shot = 0
        os.makedirs(out_dir, exist_ok=True)

    def __call__(self, tree: DynamicSoftTree, batch_X=None, **kwargs) -> str:
        """Render and save a frame; returns the path written."""
        flow: Dict[str, float] = {}
        X = batch_X if batch_X is not None else (self.batch_X() if self.batch_X else None)
        if X is not None:
            tree(X, flow=flow)
        self.shot += 1
        path = os.path.join(self.out_dir, f"tree_{self.shot:04d}.{self.format}")
        return render_tree(tree, path, flow=flow)

    def snapshot(self, tree: DynamicSoftTree) -> str:
        """Render a frame without any batch data (topology + stats only)."""
        self.shot += 1
        path = os.path.join(self.out_dir, f"tree_{self.shot:04d}.{self.format}")
        return render_tree(tree, path)


__all__ = ["build_graph", "render_tree", "TreeVisualizer", "DynamicSoftTree", "DynamicNode"]


"""Train the Dynamic Soft Decision Tree on a synthetic piecewise dataset.

This example demonstrates the two-phase paradigm:
  PHASE 1 (growth):  ``evaluate_and_grow`` bifurcates impure leaves based on the
                     measure-theoretic purity criterion.
  PHASE 2 (training): forward + loss + backward + optimizer step on the updated topology.

The target is non-linear (a clamped piecewise function of X), so the tree must split
multiple times to fit it, exercising the dynamic-growth mechanism.
"""

import torch
import torch.nn as nn

 
torch.manual_seed(0)
import os, sys

def make_data(n: int = 4000, d: int = 2, noise: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    X = torch.rand(n, d) * 4.0 - 2.0
    # Non-linear, multi-modal target requiring multiple splits.
    Y = torch.sin(3.0 * X[:, 0]) + 0.5 * X[:, 1] ** 2 + noise * torch.randn(n)
    return X, Y


def main() -> None:
    X, Y = make_data()

    model = DynamicSoftTree(in_features=X.size(1), out_features=1)
    print(f"Initial nodes: {len(model.all_nodes)} | leaves: {len(model.leaves())}")
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "visualizations/soft")
    os.makedirs(out_dir, exist_ok=True)
    viz = TreeVisualizer(out_dir=out_dir)

    batch_size = 256
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )
    def on_batch(m, batch_X, batch_Y, epoch, batch_index):
        # Render one frame per batch so you can watch topology evolve live.
        viz(m, batch_X=batch_X)

    losses = train_dynamic_tree(
        model,
        dataloader,
        epochs=30,
        growth_threshold=0.35,
        min_samples=50,
        lr=0.01,
        grow_every=1,
        grow_epochs=6,
        on_batch=on_batch,
    )
    print(f"\nFrames written to: {os.path.abspath(out_dir)}  (shot count: {viz.shot})")
    print(f"Final nodes: {len(model.all_nodes)} | leaves: {len(model.leaves())} | "
          f"depth: {model.depth()}")
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("\nTip: build an animation with:")
    print(f"  ffmpeg -framerate 4 -i {os.path.join(out_dir, 'tree_%04d.png')} out.gif")

    # Final predictions with the fully-grown topology.
    model.eval()
    with torch.no_grad():
        preds = model(X)

    mse = nn.MSELoss()(preds, Y.unsqueeze(-1)).item()

    print(f"\nFinal nodes        : {len(model.all_nodes)}")
    print(f"Final leaves       : {len(model.leaves())}")
    print(f"Final depth        : {model.depth()}")
    print(f"Train MSE          : {mse:.4f}")
    print(f"Last 5 epoch losses: {[round(v, 4) for v in losses[-5:]]}")

    assert len(model.all_nodes) > 1, "Expected the tree to have grown during training."


if __name__ == "__main__":
    main()
