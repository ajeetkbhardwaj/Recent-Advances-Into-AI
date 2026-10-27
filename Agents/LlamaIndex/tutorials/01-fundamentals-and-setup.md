# Tutorial 01 — LlamaIndex Fundamentals & Local Ollama Setup

> **Companion to:** `../LlamaIndex.md` sections 0–10, 34 (mental model).
> **Goal:** Understand what LlamaIndex is, install it correctly, wire it to a local Ollama model, and learn the `Settings` object — the single most important configuration pattern in the framework.

---

## 1.1 What We Are Actually Installing

LlamaIndex is **not one library**. Since v0.10 the project is split into:

```text
llama-index-core          ← the framework (abstractions, data structures, workflows)
llama-index-llms-ollama   ← one LLM integration
llama-index-embeddings-*  ← one or more embedding integrations
llama-index-readers-*     ← data loaders
llama-index-vector-stores-* ← vector database backends
...hundreds more (LlamaHub)
```

The umbrella `pip install llama-index` is just a **starter bundle** that pulls core + OpenAI integrations. Because we are **not** using OpenAI, we install the individual pieces we need.

> **Mental model:** LlamaHub (llamahub.ai) is a *registry*. You never "install LlamaHub"; you browse it to discover the exact `llama-index-*` package, then `pip install` it. The import path mirrors the package name:
>
> ```text
> pip install llama-index-llms-ollama
> → from llama_index.llms.ollama import Ollama
> ```

---

## 1.2 Environment Setup (Python 3.11+)

`llama-index-core` requires Python `>=3.10`. Use 3.11 or 3.12.

```bash
# Create the project directory
mkdir -p ~/llamaindex-lab && cd ~/llamaindex-lab

# Virtual environment (uv is the fast default in 2026)
uv venv --python 3.12 .venv
source .venv/bin/activate

# Or classic:
# python3.12 -m venv .venv && source .venv/bin/activate
```

Install the exact set we need for a fully local, Ollama-backed stack:

```bash
pip install \
  llama-index-core \
  llama-index-llms-ollama \
  llama-index-embeddings-ollama \
  llama-index-embeddings-huggingface \
  llama-index-vector-stores-chroma \
  llama-index-readers-file
```

Optional extras used later in this series:

```bash
pip install llama-index-retrievers-bm25   # hybrid retrieval
pip install llama-index-tools-mcp          # MCP tools
```

### Pinning for reproducibility

For a learning project, pin exact versions:

```
llama-index-core==0.14.24
llama-index-llms-ollama==...
llama-index-embeddings-ollama==...
chromadb>=0.5.0,<1.0
```

> **Rule:** Never change your embedding model after an index is built — vector dimensions must match (nomic = 768 dim, bge-small = 384 dim).

Verify the install:

```python
import llama_index.core as core
print(core.__version__)
```

---

## 1.3 Verify Ollama Is Running

Ollama serves at `http://localhost:11434`. Pull the models you need:

```bash
ollama pull llama3.1      # function-calling LLM (good default)
ollama pull qwen3         # most reliable local function calling
ollama pull nomic-embed-text   # 768-dim embedding model
```

Check the server:

```bash
curl http://localhost:11434/api/tags
```

You should see your pulled models.

---

## 1.4 The `Settings` Singleton — The Critical Idea

Everything in LlamaIndex defaults to `Settings`:

```python
from llama_index.core import Settings
```

`Settings` is a **global configuration object** that every component consults when you don't pass an explicit value. The attributes that matter most:

| Attribute                     | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| `Settings.llm`              | The chat/text model                      |
| `Settings.embed_model`      | The embedding model                      |
| `Settings.node_parser`      | How documents are split into nodes       |
| `Settings.text_splitter`    | Alias for`node_parser`                 |
| `Settings.context_window`   | Max input tokens for the LLM             |
| `Settings.num_output`       | Tokens reserved for generation           |
| `Settings.chunk_size`       | Shorthand to set the splitter chunk size |
| `Settings.chunk_overlap`    | Shorthand to set splitter overlap        |
| `Settings.callback_manager` | Observability handlers                   |

> **History:** The old `ServiceContext` is deprecated (gone since v0.10). There is no `ServiceContext.from_defaults(...)` anymore. You configure once through `Settings`, and can still override per-call (e.g. `index.as_query_engine(llm=my_llm)`).

---

## 1.5 Wiring LlamaIndex to Ollama

### LLM

```python
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama

Settings.llm = Ollama(
    model="llama3.1",          # "model:tag" format also works
    request_timeout=360.0,     # default 30s is too short for long generations
    context_window=8000,       # explicit cap; local models truncate silently otherwise
    temperature=0.1,           # low temperature = more reliable tool calling
    is_function_calling_model=True,  # enable the tools API for compatible models
)
```

> `is_function_calling_model=True` tells LlamaIndex the model exposes the Ollama tools API. If you use it on a base non-supporting model you get broken tool calls, so keep it True **only** for models that support it (`llama3.x`, `qwen2/3`, `command-r`, `mistral`, `gemma3`).

### Embeddings — option A (Ollama)

```python
from llama_index.embeddings.ollama import OllamaEmbedding

Settings.embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
    embed_batch_size=10,
)
```

### Embeddings — option B (Hugging Face, tiny CPU footprint)

```python
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",  # 384 dim, ~67MB, runs on CPU
    device="cpu",
    embed_batch_size=8,
)
```

Put it all in one `settings.py` so every tutorial/script in the series uses the same models:

```python
# settings.py
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.node_parser import SentenceSplitter

Settings.llm = Ollama(
    model="llama3.1",
    request_timeout=360.0,
    context_window=8000,
    temperature=0.1,
    is_function_calling_model=True,
)

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=50)
```

---

## 1.6 Smoke Test — Is the Wiring Working?

```python
from llama_index.core import Settings
from settings import Settings  # the file above

# Direct LLM call
response = Settings.llm.complete("Say hello in one sentence.")
print(str(response))
```

```python
# Embedding check
vec = Settings.embed_model.get_text_embedding("Hello LlamaIndex!")
print(len(vec))   # 768 for nomic-embed-text, 384 for bge-small
```

If both produce output, the model layer is ready and Tutorial 02 (RAG) will work without touching anything else.

---

## 1.7 The Document Abstraction

The atomic input unit in LlamaIndex:

```python
from llama_index.core import Document

doc = Document(text="An AI agent is a system that uses a model to act in an environment.")
print(doc.text)
print(doc.id_)          # auto-generated UUID
print(doc.metadata)     # {} by default; pass metadata={"source": "..."} to add
```

Notes for current versions:

- `Document(text=...)` is fully supported (internally stored as a `text_resource` — multimodal resources are the newer backing model).
- Prefer `metadata=` over the old `extra_info=` alias, and `id_=` over the old `doc_id=` (both still accepted as deprecated shims).
- A `Document` **is** a `Node` in the type hierarchy; a `TextNode` with plain `.text` is still the most common chunk type.

```python
from llama_index.core import Document

docs = [
    Document(text="Agents can use tools.", metadata={"source": "intro.md"}),
    Document(text="Tools provide external capabilities.", metadata={"source": "intro.md"}),
    Document(text="RAG lets an LLM retrieve external information.", metadata={"source": "intro.md"}),
]
```

---

## 1.8 Loading Real Files

The built-in `SimpleDirectoryReader` reads a folder of files (and can recurse):

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader(
    input_dir="./data",
    recursive=True,
    filename_as_id=True,
).load_data()

print(f"Loaded {len(documents)} documents")
```

To load one specific file:

```python
documents = SimpleDirectoryReader(input_files=["./data/agents.md"]).load_data()
```

> LlamaHub provides hundreds of specialized readers (`llama-index-readers-google`, `llama-index-readers-notion`, PDFs via `llama-index-readers-file`, …). The pattern is always the same: `pip install llama-index-readers-<x>` → `from llama_index.readers.<x> import ...`.

---

## 1.9 Verify Full Pipeline End-to-End (Minimal)

A minimal complete run — index + query — verifying Settings propagates:

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings

# Settings.llm and Settings.embed_model are already Ollama-backed

documents = SimpleDirectoryReader(input_dir="./data").load_data()
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_query_engine()
response = query_engine.query("What is an AI agent?")
print(response)
```

If this answers from your documents, your local LlamaIndex stack is fully functional.

---

## 1.10 Exercises

1. **Package map:** For each abstraction below, write the exact `pip install` package and the import statement: `Ollama`, `OllamaEmbedding`, `ChromaVectorStore`, `SimpleDirectoryReader`.
2. **Settings probe:** Print `Settings.llm.model`, `Settings.embed_model.model_name`, and the effective chunk size before and after you set `Settings.chunk_size = 1024`. Observe how the global allows local overrides.
3. **Classify without RAG:** Use `Settings.llm.complete(...)` to classify a sentence as positive/negative/neutral — no index, no documents. This is the "LLM layer only" baseline that RAG later improves.
4. **Tracing on:** Set `Settings.callback_manager` to a `TokenCountingHandler` and call `complete()` twice; print the token usage before/after (`print(token_counter.total_llm_token_count)`).

---

## 1.11 Wrapping Up

You now have:

```text
Local machine
 ├── Ollama server (LLM + embeddings, no API keys)
 └── LlamaIndex core
      └── Settings (llm, embed_model, node_parser)
           └── Document / SimpleDirectoryReader  ← the data entrance
```

Next tutorial builds the rest of the RAG stack: **Nodes → splitting → vector stores → index → retriever → query engine**.

---

## References

- Installation: [https://developers.llamaindex.ai/python/framework/getting_started/installation](https://developers.llamaindex.ai/python/framework/getting_started/installation)
- Settings guide: [https://developers.llamaindex.ai/python/framework/module_guides/supporting_modules/settings](https://developers.llamaindex.ai/python/framework/module_guides/supporting_modules/settings)
- Ollama LLM integration: [https://developers.llamaindex.ai/python/framework/integrations/llm/ollama](https://developers.llamaindex.ai/python/framework/integrations/llm/ollama)
- Ollama embeddings: [https://developers.llamaindex.ai/python/framework/integrations/embeddings/ollama_embedding](https://developers.llamaindex.ai/python/framework/integrations/embeddings/ollama_embedding)
- Starter local tutorial: [https://developers.llamaindex.ai/python/framework/getting_started/starter_example_local](https://developers.llamaindex.ai/python/framework/getting_started/starter_example_local)
- LlamaHub: [https://llamahub.ai](https://llamahub.ai)
