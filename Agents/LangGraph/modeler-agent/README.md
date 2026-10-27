# Modeler Agent — NL → Math Program → Solver

**Env:** `conda activate aisystem` — uses `langgraph==1.1.9`, `pulp`, `cvxpy` (clarabel/scs), `scipy`, `torch` (MPS), `duckdb`, `faiss-cpu`, `chromadb` already installed.

**Run:**
```bash
conda activate aisystem
# ensure Ollama
ollama serve &
ollama list  # gemma4:12b
python main.py
```

**Graph:** `formulate → validate → solver (PuLP/CBC) → simulator (Monte Carlo 10k, MPS if available) → sensitivity → challenger → {reformulate loop ≤3 | human_gate interrupt} → report`

**MPS:** `run_monte_carlo_torch` auto-uses `torch.backends.mps` when `MpsAvailable`, else numpy fallback. Check `DEVICE` in `src/tools.py`.

**Swap persistence:** In `src/graph.py` replace `MemorySaver()` with `SqliteSaver(sqlite3.connect("checkpoints.db"))` — graph unchanged.

**Tutorials mapped:** T01 ReAct (formulate→validate), T02 State+Memory (ModelerState + thread_id), T03 RAG not needed, T04 HITL/interrupt + eval.
