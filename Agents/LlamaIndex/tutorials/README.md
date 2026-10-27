# LlamaIndex Tutorials — Learn by Building (Local-first)

Hands-on tutorial series written to expand the mental model in `../LlamaIndex.md`.
Everything runs **100% locally** on Ollama — no API keys, no cloud providers.

> **Version note (2026):** written against `llama-index-core 0.14.x`.
> The current API is workflow-first: `FunctionAgent` / `AgentWorkflow`, `Settings` singleton
> (no more `ServiceContext`), and `ctx.store`-based agent state.

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) running on `http://localhost:11434`
- Models: `llama3.1` (or `qwen3`), `nomic-embed-text`
- Small router model for Tutorial 07: `qwen3:4b` (or `llama3.2:3b`)

```bash
ollama pull llama3.1 && ollama pull nomic-embed-text && ollama pull qwen3:4b
```

## How to Use This Series

**Do them in order.** Each tutorial depends on the previous one and ends with exercises.

| # | Tutorial | What you'll build/learn |
| - | -------- | ----------------------- |
| 01 | [Fundamentals & Local Setup](01-fundamentals-and-setup.md) | Install, `Settings`, wiring Ollama LLM + embeddings, `Document` |
| 02 | [The RAG Pipeline](02-rag-pipeline.md) | Documents → nodes → embeddings → Chroma → index → retriever → query engine, persistence, evaluation basics |
| 03 | [Tools](03-tools.md) | `FunctionTool`, `QueryEngineTool`, `ToolSpec`, utility tools, MCP; why tool descriptions are agent design |
| 04 | [Agents](04-agents.md) | `FunctionAgent`/`ReActAgent`/`AgentWorkflow`, `Context`, streaming events, agentic RAG, handoffs |
| 05 | [Workflows](05-workflows.md) | `Workflow`, events, branches, loops, joins, shared state, human-in-the-loop |
| 06 | [End-to-End Agentic RAG](06-agentic-rag-end-to-end.md) | A full local project: RAG + agent + workflow + evaluation + observability |
| 07 | [Adaptive Routing](07-adaptive-routing.md) | Make the system query- and data-adaptive: Route → Retrieve → Rerank → Verify → Synthesize, query cache, faithfulness gate, audit trail |

## Golden Mental Model (from `LlamaIndex.md` §84)

```text
User → Agent → LLM → Tools / RAG / Memory / Workflow → Observation → State → Agent …
```

This series shows exactly how LlamaIndex packages that loop.

## Adaptive Design Principle (Tutorial 07)

```text
Spend compute only where it changes the answer.
query → route → (skip retrieval? how much?) → synthesize × route → verify (grounded?)
```

## Project Skeleton Used in Tutorial 06

```text
llamaindex-lab/
├── data/            # your knowledge base
├── src/
│   ├── settings.py  # Settings wired to Ollama
│   ├── index.py     # build/load Chroma-backed index
│   ├── tools.py     # FunctionTools + QueryEngineTool
│   ├── agent.py     # AgentWorkflow
│   ├── workflow.py  # orchestrator workflow w/ human checkpoint
│   └── eval.py      # faithfulness / relevancy / correctness
├── chroma_db/       # persisted vector index
└── main.py          # CLI
```

## Key Facts Cheat Sheet

- Core imports: `from llama_index.core import ...`
- Integration imports mirror the pip package: `pip install llama-index-llms-ollama` → `from llama_index.llms.ollama import Ollama`
- Configure once through `Settings` (llm, embed_model, node_parser); override per-call with explicit kwargs.
- QueryEngine → `QueryEngineTool` → agent capability = **agentic RAG**.
- Agents are async: `await agent.run(...)`; reuse a `Context` for memory.
- Agents are workflows; workflows are explicit control flow; combine them for real systems.
- Prefer `aevaluate`/`Runner` (not `nest_asyncio`) on modern Python.
- Never change embedding models after an index exists (dimensions must match).

## References

- Official docs: <https://developers.llamaindex.ai>
- LlamaHub registry: <https://llamahub.ai>
- Source: <https://github.com/run-llama/llama_index>