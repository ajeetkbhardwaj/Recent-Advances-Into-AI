Yes. The slides are a **compressed summary of Gemma 4**, but they leave out quite a lot of the architecture that is useful for understanding _why_ the family behaves the way it does.

 I’ll turn the slides into a more complete lecture-style set of notes, separating **what Google officially specifies** from lower-level implementation details that can be inferred from released configurations.

 # Gemma 4 — Detailed Architecture and Design

 Official Gemma 4 Model Card \
  Gemma 4 Technical Report

 ## 1\. The big picture

 Gemma 4 is not one model. It is a **family of five models** designed for different compute regimes:

 | Model | Architecture | Total params | Active params | Layers | Context | Modalities |
| --- | --- | --- | --- | --- | --- | --- |
| **E2B** | Dense + PLE | 5.1B\* | 2.3B effective | 35 | 128K | Text/Image/Audio |
| **E4B** | Dense + PLE | 8B\* | 4.5B effective | 42 | 128K | Text/Image/Audio |
| **12B Unified** | Dense | 11.95B | 11.95B | 48 | 256K | Text/Image/Audio |
| **26B A4B** | MoE | 25.2B | 3.8B | 30 | 256K | Text/Image |
| **31B** | Dense | 30.7B | 30.7B | 60 | 256K | Text/Image |

 \*The E2B/E4B numbers are unusual: Google reports an **effective parameter count** because of their Per-Layer Embeddings. The physical parameter count including embeddings is larger.  Google AI for Developers

 So the family has three major architectural ideas:

 > **Efficient small dense models + encoder-free multimodal 12B + sparse MoE 26B + large dense 31B**

 All five share an important **hybrid attention philosophy**.

---

 # 2\. The common backbone: decoder-only Transformer

 At the highest level, Gemma 4 remains a **causal decoder-only Transformer**.

 Conceptually:

```
                    ┌───────────────────┐
Text tokens ───────►│                   │
Image tokens ──────►│  Input embedding  │
Audio tokens ──────►│ / modality input  │
                    └─────────┬─────────┘
                              │
                              ▼
                 ┌─────────────────────────┐
                 │ Transformer Layer 1     │
                 │ Local / Sliding Attn    │
                 │ FFN / MoE               │
                 └───────────┬─────────────┘
                             │
                 ┌─────────────────────────┐
                 │ Transformer Layer 2     │
                 │ Local / Sliding Attn    │
                 │ FFN / MoE               │
                 └───────────┬─────────────┘
                             │
                            ...
                             │
                 ┌─────────────────────────┐
                 │ Global Attention Layer  │
                 └───────────┬─────────────┘
                             │
                             ▼
                       LM Head
                             │
                             ▼
                       Next token
```

 The key difference from a conventional Transformer is **how attention is distributed across layers**.

---

 # 3\. Gemma 4's most important architectural innovation: Hybrid Attention

 A normal Transformer might use:

```
Layer 1   Full attention
Layer 2   Full attention
Layer 3   Full attention
...
Layer N   Full attention
```

 This is expensive for a 256K-token context because full attention has quadratic interaction during processing and produces a large KV cache.

 Gemma 4 instead alternates:

```
Local → Local → Local → Local → Local → Global
Local → Local → Local → Local → Local → Global
...
```

 The **local layers** use a sliding window.

 The **global layers** can attend to the entire context.

 Google explicitly states that the final layer is always global and that global layers use **unified K/V** and **Proportional RoPE (p-RoPE)** to reduce long-context memory requirements.  Google AI for Developers

 ### Why?

 Because most token interactions don't require the entire 256K context.

 For example:

```
"The cat sat on the mat because ______"
```

 Usually the model needs nearby linguistic information.

 It doesn't need to compare the current token against every token 200,000 positions earlier.

 So local attention handles:

 > **short-range reasoning cheaply**

 while global attention periodically performs:

 > **long-range information mixing**

 This is a very important design principle.

---

 # 4\. Sliding-window attention

 Suppose the window is:

```
1024 tokens
```

 For token 50,000, a local layer might only see approximately:

```
49,489 ... 50,512
          ↑
       current token
```

 rather than:

```
1 ... 50,000
```

 This drastically reduces the KV-cache burden for those layers.

 Gemma 4 uses:

 - **512-token windows** for E2B/E4B
- **1024-token windows** for 12B, 26B A4B and 31B.  Google AI for Developers

---

 # 5\. Global attention layers

 Every so often, Gemma 4 inserts a full-attention layer.

 Conceptually:

```
Local
Local
Local
Local
Local
Global
Local
Local
Local
Local
Local
Global
...
Global ← final layer
```

 For the 26B A4B model, released configurations indicate a particularly clear pattern:

```
30 total layers

25 local/sliding
5 global
```

 with the global layers appearing approximately every sixth layer, including the final layer.  Sebastian Raschka, PhD+1

 Thus:

```
25 × cheap local attention
+
5 × expensive but memory-efficient global attention
```

 This is one reason Gemma 4 can support a **256K context** without behaving like a conventional 256K full-attention Transformer.

---

 # 6\. Why global attention doesn't destroy the memory advantage

 This is where another Gemma 4 idea becomes important.

 ## Unified K/V

 In conventional multi-head attention:

```
Q → many query heads
K → many key heads
V → many value heads
```

 Gemma 4 reduces the number of K/V representations in the global layers.

 So you can think of it as:

```
Many Q heads
      │
      ├──► shared K
      │
      └──► shared V
```

 This is related to **grouped-query attention / shared K-V representations**.

 The benefit:

 > The model retains many query heads for expressive attention while storing fewer K/V vectors.

 That matters enormously for a 256K context because the KV cache is one of the largest inference-memory consumers.

 Google specifically calls out unified Keys and Values in global layers.  Google AI for Developers

---

 # 7\. Proportional RoPE — p-RoPE

 Gemma 4 also changes positional encoding behavior between local and global attention.

 The basic idea of RoPE is:

```
Q → rotate according to position
K → rotate according to position
```

 so attention can understand relative position.

 For extremely long contexts, however, simply extending ordinary RoPE can become problematic.

 Gemma 4 therefore uses **Proportional RoPE (p-RoPE)** in global layers.

 So conceptually:

```
Local attention
    ↓
ordinary/local positional encoding

Global attention
    ↓
p-RoPE
```

 This is another component of the long-context strategy.

---

 # 8\. The small models: E2B and E4B

 The "E" means **effective**.

 They are specifically designed for:

 - phones
- laptops
- edge devices
- local/private assistants
- low-memory inference

 Their parameter counts are:

```
E2B:
2.3B effective
5.1B including embeddings

E4B:
4.5B effective
8B including embeddings
```

 Google uses **Per-Layer Embeddings (PLE)** in these models.  Google AI for Developers

---

 # 9\. Per-Layer Embeddings — PLE

 This is an unusual idea.

 A conventional Transformer might have:

```
Token
  ↓
Embedding
  ↓
Layer 1
  ↓
Layer 2
  ↓
Layer 3
```

 Gemma E2B/E4B give layers their own embedding tables.

 Conceptually:

```
Token ID
   │
   ├──► Layer 1 embedding
   ├──► Layer 2 embedding
   ├──► Layer 3 embedding
   ├──► ...
   └──► Layer N embedding
```

 These tables contain many parameters, but they're primarily used as **lookup tables**, so Google reports a much smaller _effective_ parameter count for the model's computational footprint.

 This is particularly attractive for:

 > **on-device models where memory and compute are constrained.**

  Google AI for Developers
---

 # 10\. Gemma 4 12B Unified — the unusual one

 The 12B model is arguably the most architecturally interesting model in the family.

 It is called:

 > **Gemma 4 12B Unified**

 because it removes conventional multimodal encoders.

 Most multimodal LLMs look roughly like:

```
Image
  ↓
Vision Encoder
  ↓
Visual embeddings
  ↓
       ┌─────────────┐
Text → │ LLM         │
       └─────────────┘
```

 Audio similarly has:

```
Audio
 ↓
Audio Encoder
 ↓
Audio embeddings
 ↓
LLM
```

 Gemma 4 12B instead attempts:

```
Image patches ──────┐
                    │
Audio waveform ─────┼──► projection ─► shared Transformer
                    │
Text tokens ────────┘
```

 Google describes this as an **encoder-free architecture** in which raw image patches and audio waveforms are projected directly into the LLM's embedding space.  Google AI for Developers

 This is a major conceptual difference.

---

 # 11\. Why the 12B Unified architecture matters

 Traditional multimodal systems are often:

```
Vision model
      +
Audio model
      +
Language model
```

 The 12B Unified approach is closer to:

```
              ┌──────────────┐
Image ───────►│              │
              │              │
Audio ───────►│   ONE        │
              │ Transformer  │
Text ────────►│              │
              │              │
              └──────────────┘
```

 Therefore the model can learn relationships directly between modalities.

 For example:

```
Image
  ↓
Text
  ↓
Audio
  ↓
Text
```

 can exist in one sequence.

 This makes **interleaved multimodal reasoning** much more natural.

---

 # 12\. The 26B A4B model — Gemma 4's MoE architecture

 This is the sparse model in the family.

 Its architecture is:

```
25.2B total parameters
        │
        ▼
     MoE layer
        │
        ├── Expert 1
        ├── Expert 2
        ├── Expert 3
        ├── ...
        ├── Expert 128
        │
        └── Shared expert
```

 But a token doesn't execute all 128 experts.

 Instead:

```
128 routed experts
       │
       ▼
router
       │
       ├── Expert 17
       ├── Expert 42
       ├── Expert 63
       ├── ...
       └── Expert 119

+ shared expert
```

 Only **8 routed experts** are activated per token, plus the shared expert.

 Google reports:

```
Total: 25.2B
Active: 3.8B
Experts: 128
Active routed experts: 8
Shared expert: 1
Layers: 30
```

  Google AI for Developers
---

 # 13\. Why 26B A4B is called "A4B"

 The naming is:

```
26B A4B
│    │
│    └── ~4B active
│
└──── 26B total capacity
```

 More precisely, Google reports approximately **3.8B active parameters**.

 So:

 > **26B describes capacity; A4B describes computation per token.**

 This is the fundamental MoE tradeoff.

 ### Dense 26B

 Every token:

```
26B parameters participate
```

 ### MoE 26B A4B

 Every token approximately:

```
3.8B parameters participate
```

 while the model retains access to:

```
25.2B total parameters
```

 That is why the model can approach the quality of a much larger dense model while requiring substantially less computation per token.

---

 # 14\. What does the router actually do?

 For each token:

```
hidden state
     │
     ▼
  Router
     │
     ▼
scores for 128 experts
     │
     ▼
top-8 selection
     │
     ├──► Expert A
     ├──► Expert B
     ├──► Expert C
     ├──► ...
     └──► Expert H
             +
        shared expert
```

 The outputs are then combined and passed to the next Transformer stage.

 This allows experts to specialize statistically.

 For example, one expert may become particularly useful for:

```
mathematics
```

 another:

```
code
```

 another:

```
multilingual text
```

 although such specialization should not be interpreted as a strict human-readable assignment—the experts are learned components, not manually labeled modules.

---

 # 15\. MoE does NOT mean 4B memory

 This distinction is very important.

 People sometimes say:

 > "26B A4B is basically a 4B model."

 That's not quite correct.

 ### Compute

 Approximately:

```
~3.8B active
```

 ### Weight memory

 You still need access to the full:

```
~25.2B parameters
```

 unless you use techniques such as quantization/offloading.

 Therefore:

```
MoE advantage
     ↓
lower compute per token

NOT necessarily
     ↓
lower model storage
```

 This is one of the most important concepts to remember when comparing MoE and dense models.

---

 # 16\. The 31B model — the dense flagship

 The 31B is the largest dense Gemma 4.

```
30.7B parameters
60 layers
1024-token local window
256K context
Text + Image
~550M vision encoder
```

  Google AI for Developers  Unlike 26B A4B:

```
31B → dense
26B A4B → sparse MoE
```

 So the 31B spends substantially more computation per token, but has a simpler dense execution pattern.

 This is why the 31B generally sits at the top of the Gemma 4 benchmark results shown in your slides.

---

 # 17\. Multimodal architecture

 Gemma 4 isn't merely a text LLM with an image API bolted onto it.

 It is designed for:

```
Text
Image
Video
Audio*
```

 where audio is natively supported by:

```
E2B
E4B
12B Unified
```

 and the larger 26B/31B models support image/video processing but not native audio input.  Google AI for Developers

---

 # 18\. Vision processing

 The image path can process:

 - variable aspect ratios
- variable resolutions
- documents
- screenshots
- charts
- handwriting
- OCR
- UI elements
- photographs

 The visual representation isn't fixed to one image size.

 Gemma 4 supports visual token budgets:

```
70
140
280
560
1120
```

 Higher token budgets mean:

```
more visual detail
       ↓
higher compute
```

 Lower budgets mean:

```
less detail
       ↓
faster inference
```

 Google recommends higher budgets for things such as small-text OCR and document analysis, while lower budgets can be useful for captioning or processing many video frames.  Google AI for Developers

---

 # 19\. Video understanding

 Gemma 4 doesn't have to be interpreted as a conventional dedicated video-language architecture.

 A useful conceptual model is:

```
Video
 │
 ├── Frame 1 → image representation
 ├── Frame 2 → image representation
 ├── Frame 3 → image representation
 ├── ...
 └── Frame N → image representation
                 │
                 ▼
             Transformer
```

 The official model card specifies video processing as sequences of frames, with up to **60 seconds at 1 frame/second** under the documented configuration.  Google AI for Developers

---

 # 20\. Audio architecture

 For E2B, E4B and 12B Unified:

```
Audio waveform
      ↓
audio representation / projection
      ↓
LLM token/embedding space
      ↓
Transformer
      ↓
text
```

 Capabilities include:

 - automatic speech recognition
- speech-to-text
- speech translation

 with a documented maximum audio duration of **30 seconds**.  Google AI for Developers

 The 12B model is particularly interesting because it does this without a separate conventional audio encoder.

---

 # 21\. Interleaved multimodal input

 One of the capabilities emphasized in your second slide is:

 > **You can mix modalities within a prompt.**

 For example:

```
[IMAGE]
"Describe this diagram."

[AUDIO]
"Now explain what the speaker says."

[IMAGE]
"Compare it with this second diagram."
```

 Conceptually, the model sees a single sequence:

```
Text
↓
Image representation
↓
Text
↓
Audio representation
↓
Text
↓
Image representation
↓
Text
```

 This is much more flexible than:

```
image → separate vision model
audio → separate audio model
text → LLM
```

---

 # 22\. Thinking mode

 Gemma 4 introduces configurable reasoning/thinking.

 Conceptually:

```
User prompt
     ↓
Thinking enabled?
     │
     ├── No ──► Answer
     │
     └── Yes
           ↓
       reasoning
           ↓
       final answer
```

 The model card documents a special `<|think|>` control mechanism and a separate thought channel.  Google AI for Developers

 This is important because **thinking is not necessarily a separate model**.

 It is primarily a generation mode in which the model is allowed to produce an intermediate reasoning trajectory before the final response.

---

 # 23\. System / assistant / user roles

 Gemma 4 also changes the prompting interface compared with older Gemma releases.

 It supports standard:

```
system
user
assistant
```

 roles.

 That makes it much easier to build:

```
Chatbots
Agents
Tool-using systems
Coding assistants
```

 The system prompt can also control thinking behavior.  Google AI for Developers

---

 # 24\. Function calling and agents

 Gemma 4 isn't intended only as:

```
prompt → answer
```

 It supports structured tool use.

 The architecture therefore fits an agent loop:

```
User
 ↓
Gemma
 ↓
decides tool required
 ↓
Function call
 ↓
External tool
 ↓
Tool result
 ↓
Gemma
 ↓
reason
 ↓
next tool / final answer
```

 This is why the benchmark suite includes agentic/function-use evaluations such as Tau2.

---

 # 25\. Coding

 Gemma 4 has substantial coding capability.

 The model card specifically highlights:

 - code generation
- code completion
- code correction
- agentic workflows

 The benchmark in your slide shows a particularly large difference between model sizes.

 For example, the displayed **Codeforces ELO** scores are approximately:

```
31B       2150
26B A4B   1718
12B       1659
E4B        940
E2B        633
```

 So the architecture scales meaningfully with model capacity.

---

 # 26\. Long-context architecture

 This is one of the strongest design themes of Gemma 4.

 The context sizes are:

```
E2B/E4B       128K
12B           256K
26B A4B       256K
31B           256K
```

  Google AI for Developers  The trick isn't simply:

 > "Make the Transformer capable of 256K."

 Instead, several architectural techniques work together:

```
             Gemma 4 Long Context
                    │
        ┌───────────┼────────────┐
        ↓           ↓            ↓
 Sliding-window   Global       Unified
 attention        layers       K/V
        │           │            │
        └───────────┼────────────┘
                    ↓
                  p-RoPE
                    ↓
             lower KV memory
```

 That combination is more important than any individual feature.

---

 # 27\. Why hybrid attention matters for KV cache

 Imagine 256K tokens.

 With ordinary full attention:

```
Every layer
     ↓
KV for 256K tokens
```

 With hybrid attention:

```
25 local layers
     ↓
KV limited to ~1024-token window

5 global layers
     ↓
KV grows with full context
```

 So the long-context memory growth is concentrated in relatively few layers.

 That is an elegant architecture-level solution.

---

 # 28\. The Transformer block

 At a conceptual level, each Gemma 4 layer can be understood as:

```
                x
                │
                ▼
          RMS normalization
                │
                ▼
             Attention
                │
                ▼
          residual connection
                │
                ▼
          RMS normalization
                │
                ▼
           Feed Forward
            /        \
         Dense       MoE*
            \        /
             combine
                │
                ▼
          residual connection
```

 `*` The 26B A4B model replaces the ordinary dense feed-forward capacity with the sparse expert mechanism.

 The exact implementation contains additional normalization and attention details; the important lecture-level idea is:

 > **Gemma 4 is still fundamentally a Transformer decoder, but its attention and feed-forward computation have been redesigned for efficiency.**

---

 # 29\. Dense vs MoE architecture

 This is worth putting directly into your notes.

 ### Dense Gemma

```
Token
  ↓
Attention
  ↓
FFN
  ↓
Next layer
```

 All model parameters participate.

 ### MoE Gemma

```
Token
  ↓
Attention
  ↓
Router
  ↓
Choose experts
  │
  ├── Expert 1
  ├── Expert 7
  ├── Expert 18
  ├── ...
  └── Expert 128
       ↓
Selected experts
       ↓
Combine
       ↓
Next layer
```

 Only a small subset participates for each token.

---

 # 30\. Model-family philosophy

 The five models form an interesting deployment ladder:

```
                    Gemma 4
                       │
       ┌───────────────┼────────────────┐
       │               │                │
     Edge           Workstation       Server
       │               │                │
   E2B / E4B       12B / 26B A4B       31B
       │               │                │
   efficiency       balanced           quality
```

 But 26B A4B is a special point:

```
Large capacity
      +
Sparse computation
      =
high intelligence / compute efficiency
```

---

 # 31\. Performance — what your slides are showing

 Your benchmark slide is making a broader point than "31B wins."

 Look at the pattern:

 ### General reasoning

```
31B       85.2 MMLU Pro
26B A4B   82.6
12B       77.2
E4B       69.4
E2B       60.0
```

 ### AIME

```
31B       89.2
26B A4B   88.3
12B       77.5
E4B       42.5
E2B       37.5
```

 ### Vision — MMMU Pro

```
31B       76.9
26B A4B   73.8
12B       69.1
E4B       52.6
E2B       44.2
```

 ### Long context

```
31B       66.4
26B A4B   44.1
12B       43.4
E4B       25.4
E2B       19.1
```

 These are the official model-card results for the instruction-tuned models.  Google AI for Developers

 The important lesson is:

 > **Model scale still matters, but architectural efficiency allows the smaller/sparse models to approach much larger models in selected tasks.**

---

 # 32\. Your final graph — why it matters

 The last slide compares **intelligence against total model size**.

 The interesting points are:

```
Gemma 4 31B Thinking
        ↑
        │
Gemma 4 26B A4B Thinking
        ↑
        │
       other models
```

 The key observation is that Gemma 4 occupies a strong part of the:

 > **quality vs parameter-count trade-off**

 curve.

 Especially interesting is:

```
26B total
~3.8B active
```

 yet it performs much closer to the 31B dense model than its active-parameter count would suggest.

 That is exactly what an effective MoE architecture is supposed to accomplish.

---

 # 33\. Training

 The model card says the pre-training data contains:

```
Web documents
Code
Mathematics
Images
Audio
```

 and covers **140+ languages**, with a training-data cutoff of **January 2025**.  Google AI for Developers

 So the model isn't merely trained on text and then retrofitted with vision.

 It is trained as a **multimodal model**.

 The training pipeline also includes filtering for:

 - sensitive information
- harmful content
- CSAM
- quality/safety criteria

 according to Google's model documentation.  Google AI for Developers

---

 # 34\. Multilingual capability

 Gemma 4 distinguishes between:

```
35+ languages
```

 with out-of-the-box support and

```
140+ languages
```

 represented in pre-training.

 That distinction is important.

 It doesn't necessarily mean every one of the 140+ languages has equal instruction-following quality.

---

 # 35\. A useful mental model for the whole family

 If I had to compress Gemma 4 into one architecture diagram:

```
                         GEMMA 4
                            │
            ┌───────────────┴────────────────┐
            │                                │
       INPUT MODALITIES                 TEXT DECODER
            │                                │
    ┌───────┼────────┐                       │
    │       │        │                       │
   Text   Image    Audio                     │
            │       │                        │
            │       └── E2B/E4B/12B          │
            │                                │
            └──────────────┐                 │
                           ▼                 │
                   modality embedding        │
                           │                 │
                           └────────┬────────┘
                                    ▼
                         ┌──────────────────┐
                         │ Transformer      │
                         │                  │
                         │ Local attention  │
                         │       ↓          │
                         │ FFN / MoE        │
                         │       ↓          │
                         │ Local attention  │
                         │       ↓          │
                         │ ...              │
                         │       ↓          │
                         │ Global attention │
                         │       ↓          │
                         │ Final global     │
                         └────────┬─────────┘
                                  │
                                  ▼
                              LM Head
                                  │
                                  ▼
                              Text output
```

 And the major variants change the middle:

```
E2B/E4B
   ↓
Dense + PLE

12B
   ↓
Dense + encoder-free multimodal input

26B A4B
   ↓
MoE + 128 experts + top-8 routing

31B
   ↓
Large dense Transformer
```

---

 # 36\. What is actually novel about Gemma 4?

 I would summarize the architectural lessons as **six ideas**:

 ### ① Hybrid attention

```
local attention + occasional global attention
```

 instead of full attention everywhere.

 ### ② Memory-efficient global attention

```
unified K/V + p-RoPE
```

 to make long contexts more practical.

 ### ③ Sparse MoE

```
25B capacity
≈3.8B active/token
```

 in 26B A4B.

 ### ④ Per-Layer Embeddings

 Used in E2B/E4B to make small models more parameter-efficient.

 ### ⑤ Encoder-free multimodality

 The 12B model directly incorporates image/audio representations into the language-model backbone instead of depending on separate conventional modality encoders.

 ### ⑥ Reasoning as a controllable generation mode

 Thinking is integrated into the model's generation interface rather than requiring a completely separate "reasoning model."

 These are the architectural ideas that matter much more than simply memorizing:

 > "E2B = 2.3B, E4B = 4.5B, 31B = 30.7B."

---

 # 37\. One important correction to the slide

 Your first slide says:

 > **"4 dense models + 1 sparse model"**

 That's correct if it means the four dense variants are:

```
E2B
E4B
12B Unified
31B
```

 and the one sparse model is:

```
26B A4B
```

 So the family has **five models total**, not four.

 The official model card confirms exactly these five variants.  Google AI for Developers

---

 # 38\. Final "exam answer"

 If you were asked:

 > **"Explain the architecture of Gemma 4."**

 A strong concise answer would be:

 > **Gemma 4 is a family of open-weight, decoder-only Transformer models ranging from effective 2.3B to 30.7B parameters. The family combines dense and Mixture-of-Experts architectures and uses hybrid attention, interleaving local sliding-window attention with periodic global attention. Global layers use unified K/V representations and proportional RoPE to reduce long-context memory costs. E2B and E4B use Per-Layer Embeddings for parameter-efficient edge deployment. The 12B Unified model is encoder-free and directly projects image patches and audio into the language-model embedding space. The 26B A4B model uses 128 routed experts, selecting 8 experts plus a shared expert per token, giving roughly 3.8B active parameters despite 25.2B total parameters. The 31B model is the largest dense variant. Across the family, Gemma 4 supports reasoning/thinking, long-context processing up to 256K tokens, multimodal image/video understanding, audio on smaller models, coding, multilingual tasks, and structured function calling.**  Google AI for Developers+1

 ### The deepest takeaway

 The central engineering philosophy of Gemma 4 is:

 **Don't make every token pay the full computational and memory cost of a huge Transformer.**

 Instead:

```
                 Gemma 4 efficiency strategy
                           │
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   Local attention       MoE routing       PLE
        │                  │                  │
   cheap context       sparse compute     edge efficiency
        │                  │                  │
        └──────────────────┼──────────────────┘
                           ↓
                 Global attention
                           ↓
                 long-range reasoning
                           ↓
                    strong model
```

 That is the architectural story behind the benchmark results in your slides: **Gemma 4 is less about one revolutionary Transformer block and more about combining several efficiency mechanisms—hybrid attention, sparse experts, parameter-efficient embeddings, and unified multimodal processing—into different model sizes optimized for different hardware.**  Google AI for Developers

Worked for 10s

Absolutely. If your goal is **architecture reproduction**, not merely “understanding the model,” then we should make this much more precise.

 One important correction to my previous answer: the **Gemma 4 12B Unified configuration is now publicly available**, and it gives us substantially more implementation detail than was visible in your slides. The official/Hugging Face configuration specifies **48 layers, 3,840 hidden size, 15,360 FFN size, 16 attention heads, 8 KV heads for sliding layers, 1 global KV head, 1,024 sliding window, 256K context, and 5:1 sliding/global layer pattern**.  Hugging Face+1

 Below is the architecture I would use as the **master implementation diagram**.

---

 # Gemma 4 12B Unified — Implementation-Level Architecture

 ## 1\. Complete system-level Mermaid

 Mermaid flowchart: User / Training Sequence, Text Tokens, Image, Raw Audio 16 kHz, Video, Tokenizer Vocabulary = 262,144, Token IDs B × T, Scaled Token Embedding 262144 × 3840 Embedding × sqrt(3840), Image preprocessing, 16 × 16 pixel patches, 3 × 3 patch merge, 48 × 48 model patches 6912 raw channels, LayerNorm 6912, Dense Projection 6912 → 3840, LayerNorm 3840, Add factorized 2D positional embedding X embedding + Y embedding, Final LayerNorm 3840, RMSNorm 3840, Linear 3840 → 3840, Raw waveform 16 kHz, Chunk into 640 samples 40 ms/frame, RMSNorm 640, Linear 640 → 3840, Video frames, Each frame processed through image pipeline, Visual soft-token sequence, Multimodal sequence assembly Text + Image Soft Tokens + Audio Soft Tokens, Initial hidden states B × S × 3840, Layer 1 Sliding Attention, Layer 2 Sliding Attention, Layer 3 Sliding Attention, Layer 4 Sliding Attention, Layer 5 Sliding Attention, Layer 6 GLOBAL Attention, Layers 7-11 Sliding Attention ×5, Layer 12 GLOBAL Attention, Repeat 6-layer pattern 8 times total, Layer 48 GLOBAL Attention, Final RMSNorm 3840, LM Head 3840 → 262144, Next-token logits B × S × 262144, Training: shifted cross entropy, Inference: sampling / decoding

 The important point is that **12B Unified has no conventional vision tower and no conventional audio encoder**. Image patches and raw audio are converted directly into the same 3,840-dimensional space consumed by the decoder.  GitHub+1

---

 # 2\. The exact 48-layer Transformer layout

 This is important enough to show separately.

 The configuration is:

```
S S S S S G
S S S S S G
S S S S S G
S S S S S G
S S S S S G
S S S S S G
S S S S S G
S S S S S G
```

 where:

 - `S` = sliding-window attention
- `G` = full/global attention

 Therefore:

 **40 sliding layers + 8 global layers.**

 The global layers are:

```
6
12
18
24
30
36
42
48
```

 This exact 5:1 pattern is present in the released configuration/conversion code.  Hugging Face+1

 ### Mermaid

```
flowchart LR

    subgraph B1["Block 1"]
        L1["01 S"] --> L2["02 S"] --> L3["03 S"] --> L4["04 S"] --> L5["05 S"] --> L6["06 GLOBAL"]
    end

    subgraph B2["Block 2"]
        L7["07 S"] --> L8["08 S"] --> L9["09 S"] --> L10["10 S"] --> L11["11 S"] --> L12["12 GLOBAL"]
    end

    subgraph B3["Block 3"]
        L13["13 S"] --> L14["14 S"] --> L15["15 S"] --> L16["16 S"] --> L17["17 S"] --> L18["18 GLOBAL"]
    end

    subgraph B4["Block 4"]
        L19["19 S"] --> L20["20 S"] --> L21["21 S"] --> L22["22 S"] --> L23["23 S"] --> L24["24 GLOBAL"]
    end

    subgraph B5["Block 5"]
        L25["25 S"] --> L26["26 S"] --> L27["27 S"] --> L28["28 S"] --> L29["29 S"] --> L30["30 GLOBAL"]
    end

    subgraph B6["Block 6"]
        L31["31 S"] --> L32["32 S"] --> L33["33 S"] --> L34["34 S"] --> L35["35 S"] --> L36["36 GLOBAL"]
    end

    subgraph B7["Block 7"]
        L37["37 S"] --> L38["38 S"] --> L39["39 S"] --> L40["40 S"] --> L41["41 S"] --> L42["42 GLOBAL"]
    end

    subgraph B8["Block 8"]
        L43["43 S"] --> L44["44 S"] --> L45["45 S"] --> L46["46 S"] --> L47["47 S"] --> L48["48 GLOBAL"]
    end

    B1 --> B2 --> B3 --> B4 --> B5 --> B6 --> B7 --> B8
```

---

 # 3\. Exact model configuration

 This is the table I would put directly into an implementation specification.

 | Parameter | Gemma 4 12B Unified |
| --- | --- |
| Total parameters | \~11.96B |
| Architecture | Decoder-only Transformer |
| Layers | **48** |
| Hidden size | **3840** |
| Vocabulary | **262,144** |
| Context | **262,144** |
| FFN intermediate | **15,360** |
| Attention heads | **16** |
| Sliding KV heads | **8** |
| Global KV heads | **1** |
| Sliding head dimension | **256** |
| Global head dimension | **512** |
| Sliding window | **1024** |
| Sliding/global ratio | **5:1** |
| Attention dropout | **0** |
| Attention bias | **False** |
| Activation | **GELU (PyTorch tanh approximation)** |
| RMSNorm epsilon | **1e-6** |
| Final logit softcap | **30** |
| MoE | **No** |
| PLE | **No** |
| KV sharing across layers | **No** |
| Word embeddings tied to LM head | **Yes** |
| Default dtype in checkpoint | **bfloat16** |

 These values are directly reflected in the released configuration.  Hugging Face

---

 # 4\. One Transformer layer in exact detail

 This is where implementation becomes interesting.

 A Gemma 4 Unified decoder layer is approximately:

```
flowchart TD

    X[Input x]

    X --> N1[RMSNorm]
    N1 --> ATT[Self Attention]

    ATT --> PAN[Post-Attention RMSNorm]

    X --> R1[Residual]
    PAN --> ADD1((+))
    R1 --> ADD1

    ADD1 --> N2[Pre-FFN RMSNorm]
    N2 --> MLP[GEGLU-style MLP]

    MLP --> N3[Post-FFN RMSNorm]

    ADD1 --> R2[Residual]
    N3 --> ADD2((+))
    R2 --> ADD2

    ADD2 --> LS[Layer Scalar]
    LS --> OUT[Layer Output]
```

 The released implementation actually contains **four RMSNorm operations per decoder layer**:

```
input_layernorm
post_attention_layernorm
pre_feedforward_layernorm
post_feedforward_layernorm
```

 followed by residual additions and a per-layer scalar initialized to 1.  GitHub

---

 # 5\. Exact layer equations

 Let:

 $$
x_l \in \mathbb{R}^{S\times3840}
$$

 be the input to layer $l$.

 ### Attention branch

 $$
u = RMSNorm(x_l)
$$

 then:

 $$
a = Attention(u)
$$

 then:

 $$
a' = RMSNorm(a)
$$

 and:

 $$
h = x_l + a'
$$

 Then the FFN:

 $$
z = RMSNorm(h)
$$

 $$
m = MLP(z)
$$

 $$
m' = RMSNorm(m)
$$

 $$
x_{l+1} = (h+m') \cdot s_l
$$

 where:

 $$
s_l = 1
$$

 at initialization.

 That last layer scalar is easy to miss if you're implementing from a high-level Transformer library. The released implementation explicitly has it.  GitHub

---

 # 6\. The MLP

 The FFN is **not** a simple:

```
Linear → GELU → Linear
```

 It is gated.

 The released implementation is:

 $$
MLP(x)=W_{down}
\left[
GELU(W_{gate}x)
\odot
(W_{up}x)
\right]
$$

 with:

```
hidden = 3840
intermediate = 15360
```

 So:

```
flowchart LR

    X["x<br/>3840"]

    X --> G["Gate projection<br/>3840 → 15360"]
    G --> GELU["GELU"]

    X --> U["Up projection<br/>3840 → 15360"]

    GELU --> MUL((⊙))
    U --> MUL

    MUL --> D["Down projection<br/>15360 → 3840"]

    D --> Y["MLP output<br/>3840"]
```

 The exact implementation uses:

```
down(
    gelu(gate(x)) * up(x)
)
```

 with no biases.  GitHub

---

 # 7\. Sliding attention

 This is **GQA**.

 The configuration says:

```
Q heads = 16
KV heads = 8
head dimension = 256
```

 Therefore:

```
Q dimension = 16 × 256
            = 4096

K dimension = 8 × 256
            = 2048

V dimension = 8 × 256
            = 2048
```

 Mermaid:

 Mermaid flowchart: Hidden state 3840, Q projection 3840 → 4096, K projection 3840 → 2048, V projection 3840 → 2048, 16 Query heads 16 × 256, 8 Key heads 8 × 256, 8 Value heads 8 × 256, Repeat K heads GQA ×2, Repeat V heads GQA ×2, Scaled dot-product attention, Output projection 4096 → 3840

 Because there are 16 query heads and 8 KV heads:

 $$
\text{GQA group size}=16/8=2
$$

 The released implementation explicitly repeats K/V across query groups.  GitHub

---

 # 8\. Sliding attention window

 For sliding layers:

 $$
W=1024
$$

 So each token can attend only to the appropriate causal local region.

 Conceptually:

```
Tokens:

t-1023 ... t-2 t-1 t
  │              │
  └──────────────┘
      1024 window
```

 The implementation creates a dedicated sliding-window causal mask.  GitHub

---

 # 9\. Global attention is substantially different

 This is one of the most important details.

 Global layers use:

```
16 query heads
1 KV head
head dimension = 512
```

 Therefore:

```
Q = 16 × 512 = 8192
K = 1 × 512  = 512
V = 1 × 512  = 512
```

 And the configuration sets:

```
attention_k_eq_v = true
```

 meaning **there isn't a separate V projection** in global attention; the K projection output is reused as V. The implementation explicitly sets `v_proj=None` in this mode and assigns `value_states = key_states`.  GitHub

 ### Global attention

 Mermaid flowchart: Hidden state 3840, Q projection 3840 → 8192, K projection 3840 → 512, 16 Q heads 16 × 512, 1 K head 512, V = K 512, Broadcast K to 16 Q heads, Broadcast V to 16 Q heads, FULL GLOBAL ATTENTION, Output projection 8192 → 3840

 This is effectively an **MQA-like global attention mechanism**, with the additional K=V projection sharing.

---

 # 10\. Why global head dimension is 512

 Notice:

```
Sliding:
head_dim = 256

Global:
head_dim = 512
```

 That's deliberate.

 So:

```
local:
16 × 256 = 4096 attention width

global:
16 × 512 = 8192 attention width
```

 The expensive global operation gets a larger per-head representation while using only **one KV head**.

 This is one of the more unusual aspects of the architecture. The configuration explicitly exposes `global_head_dim=512`.  Hugging Face

---

 # 11\. RoPE is different for local and global layers

 There are **two positional-encoding regimes**.

 ### Sliding layers

```
RoPE type = default
theta = 10,000
head dimension = 256
```

 ### Global layers

```
RoPE type = proportional
theta = 1,000,000
partial_rotary_factor = 0.25
head dimension = 512
```

 This is explicitly present in the configuration.  Hugging Face

 Mermaid:

 Mermaid flowchart: Position IDs, Sliding layer, Global layer, Standard RoPE theta = 10,000 head\_dim = 256, Proportional RoPE theta = 1,000,000 partial rotary = 0.25 head\_dim = 512

---

 # 12\. Important: 12B does NOT use MoE

 This model is:

```
Dense
```

 not:

```
MoE
```

 The configuration has:

```
enable_moe_block = false
num_experts = null
```

 So every token goes through the same 48 Transformer layers and every layer's FFN.  Hugging Face

 This is important because the 26B A4B architecture is a completely different story.

---

 # 13\. The encoder-free vision architecture

 Now let's go all the way down to pixels.

 The pipeline is:

 Mermaid flowchart: RGB Image, Divide into 16×16 patches, 3×3 patch merging, 48×48 merged patch, Flatten, 6912 values 48×48×3, LayerNorm 6912, Linear 6912 → 3840, LayerNorm 3840, Add factorized 2D position X embedding + Y embedding, LayerNorm 3840, RMSNorm, Linear 3840 → 3840, Image soft tokens 3840-dimensional

 This is **not** a ViT.

 There is:

```
NO:
- Vision Transformer
- self-attention vision encoder
- SigLIP tower
- CLIP tower
```

 Instead:

 > **raw pixels → patch projection → positional embeddings → LM space**

 The Hugging Face implementation explicitly describes this as the replacement for a vision tower.  GitHub+1

---

 # 14\. Exact image dimensions

 The image path starts with:

```
patch_size = 16
pooling_kernel = 3
```

 therefore:

 $$
16\times3=48
$$

 The resulting model patch is:

```
48 × 48 × 3
```

 which gives:

 $$
48\times48\times3=6912
$$

 input features.

 Then:

 $$
6912\rightarrow3840
$$

 The configuration specifies a default of **280 visual soft tokens**, with supported budgets:

```
70
140
280
560
1120
```

 The processor requires image dimensions compatible with the 48-pixel model patch geometry.  GitHub

---

 # 15\. Factorized 2D positional embedding

 This isn't simply:

```
Embedding[x,y]
```

 Instead, conceptually:

```
Position(i,j)

     │
 ┌───┴────┐
 ↓        ↓
X-table  Y-table
 ↓        ↓
Ex       Ey
 └───┬────┘
     ↓
Ex + Ey
```

 The configuration has:

```
mm_posemb_size = 1120
```

 and the implementation describes the positional table as factorized by the two spatial axes.  GitHub+1

---

 # 16\. Audio architecture

 This is even simpler.

 There is **no Conformer** and no mel-spectrogram encoder in Unified 12B.

 Raw audio:

```
16,000 samples/sec
```

 is divided into:

```
640 samples/frame
```

 which corresponds to:

 $$
640/16000=0.04s
$$

 so:

 > **40 ms per audio soft token**

 Then:

 Mermaid flowchart: Raw waveform 16 kHz, Chunk 640 samples, RMSNorm 640, Linear 640 → 3840, Audio soft token 3840

 There is no temporal downsampling in this unified audio path, so:

 $$
N_{audio}=\lceil N_{samples}/640\rceil
$$

 The official implementation documentation confirms this exact design.  GitHub

---

 # 17\. The multimodal fusion point

 This is perhaps the most conceptually important feature.

 There is **one Transformer**.

 Not:

```
Vision Transformer
      ↓
LLM

Audio Encoder
      ↓
LLM
```

 Instead:

 Mermaid flowchart: Text token embeddings 3840, Image soft tokens 3840, Audio soft tokens 3840, Sequence Fusion, ONE sequence B × S × 3840, 48-layer Transformer

 For example:

```
Text
Image
Text
Audio
Text
Image
Text
```

 can become:

```
[T1 T2 T3]
[IMG1 IMG2 ... IMG280]
[T4 T5]
[AUD1 AUD2 ...]
[T6]
[IMG...]
[T7 T8]
```

 and the same Transformer processes the resulting sequence.

 That is what **Unified** really means.

---

 # 18\. But there is an important attention-mask detail

 The configuration says:

```
use_bidirectional_attention = "vision"
```

 This means the vision modality gets special bidirectional attention behavior, while normal text remains causal.  Hugging Face+1

 So don't implement everything as one ordinary lower-triangular causal mask.

 Conceptually:

 Mermaid flowchart: Multimodal sequence, Token type, Text, Vision, Causal attention token cannot see future text, Bidirectional vision behavior, Attention mask, Transformer

 This is a particularly important implementation detail that a high-level description of Gemma 4 can easily miss.

---

 # 19\. Text embedding

 The text vocabulary is:

 $$
V=262144
$$

 and:

 $$
d=3840
$$

 Therefore:

```
Embedding matrix:

262144 × 3840
```

 The implementation scales the embedding output by:

 $$
\sqrt{3840}
$$

 before sending it into the Transformer.  GitHub

 And:

```
lm_head.weight
       =
embed_tokens.weight
```

 because the model uses tied embeddings.  Hugging Face+1

---

 # 20\. Output head

 At the end:

```
48 Transformer layers
       ↓
Final RMSNorm
       ↓
3840
       ↓
Linear / tied embedding matrix
       ↓
262144 logits
```

 So:

 $$
H\in\mathbb{R}^{B\times S\times3840}
$$

 becomes:

 $$
logits\in\mathbb{R}^{B\times S\times262144}
$$

 The configuration also specifies:

```
final_logit_softcapping = 30
```

 The implementation applies:

 $$
z' = 30\tanh(z/30)
$$

 before the probability distribution.  GitHub+1

---

 # 21\. Full architecture in one Mermaid diagram

 Here is the one I would actually save as the **Gemma-4-12B architecture master diagram**:

```
flowchart TB

    %% ============================================================
    %% INPUT
    %% ============================================================

    INPUT["Multimodal Input Sequence"]

    %% ============================================================
    %% TEXT
    %% ============================================================

    INPUT --> TEXT["TEXT"]

    TEXT --> TOK["Tokenizer<br/>Vocab = 262,144"]
    TOK --> E["Embedding<br/>262144 × 3840"]
    E --> ES["× sqrt(3840)"]

    %% ============================================================
    %% IMAGE
    %% ============================================================

    INPUT --> IMAGE["IMAGE"]

    IMAGE --> I1["16×16 pixel patches"]
    I1 --> I2["3×3 patch merge"]
    I2 --> I3["48×48×3 = 6912"]
    I3 --> I4["LayerNorm"]
    I4 --> I5["Linear<br/>6912 → 3840"]
    I5 --> I6["LayerNorm"]
    I6 --> I7["Factorized 2D Position<br/>X + Y"]
    I7 --> I8["LayerNorm"]
    I8 --> I9["RMSNorm"]
    I9 --> I10["Linear<br/>3840 → 3840"]

    %% ============================================================
    %% AUDIO
    %% ============================================================

    INPUT --> AUDIO["AUDIO"]

    AUDIO --> A1["16 kHz waveform"]
    A1 --> A2["640-sample frames<br/>40 ms"]
    A2 --> A3["RMSNorm"]
    A3 --> A4["Linear<br/>640 → 3840"]

    %% ============================================================
    %% VIDEO
    %% ============================================================

    INPUT --> VIDEO["VIDEO"]

    VIDEO --> V1["Extract frames"]
    V1 --> V2["Image pipeline"]
    V2 --> V3["Visual soft tokens"]

    %% ============================================================
    %% FUSION
    %% ============================================================

    ES --> F["MULTIMODAL SEQUENCE FUSION"]
    I10 --> F
    A4 --> F
    V3 --> F

    F --> X0["B × S × 3840"]

    %% ============================================================
    %% TRANSFORMER
    %% ============================================================

    X0 --> L1["Layer 1<br/>SLIDING"]
    L1 --> L2["Layer 2<br/>SLIDING"]
    L2 --> L3["Layer 3<br/>SLIDING"]
    L3 --> L4["Layer 4<br/>SLIDING"]
    L4 --> L5["Layer 5<br/>SLIDING"]
    L5 --> L6["Layer 6<br/>GLOBAL"]

    L6 --> L7["Layers 7–11<br/>SLIDING ×5"]
    L7 --> L12["Layer 12<br/>GLOBAL"]

    L12 --> P1["Repeat pattern<br/>Layers 13–18"]
    P1 --> P2["Repeat pattern<br/>Layers 19–24"]
    P2 --> P3["Repeat pattern<br/>Layers 25–30"]
    P3 --> P4["Repeat pattern<br/>Layers 31–36"]
    P4 --> P5["Repeat pattern<br/>Layers 37–42"]
    P5 --> P6["Repeat pattern<br/>Layers 43–48"]

    P6 --> N["Final RMSNorm<br/>3840"]

    %% ============================================================
    %% LM HEAD
    %% ============================================================

    N --> HEAD["Tied LM Head<br/>3840 → 262144"]

    HEAD --> CAP["Logit Softcap<br/>30 × tanh(logit/30)"]

    CAP --> OUT["Next Token Distribution"]

    %% ============================================================
    %% TRAINING
    %% ============================================================

    OUT --> CE["Shifted Cross Entropy"]
    CE --> TRAIN["Backpropagation"]

    %% ============================================================
    %% INFERENCE
    %% ============================================================

    OUT --> SAMPLE["Sampling / Decoding"]

    %% ============================================================
    %% LAYER DETAIL
    %% ============================================================

    L5 -.-> SD["SLIDING ATTENTION<br/>16 Q heads × 256<br/>8 KV heads × 256<br/>1024 window<br/>GQA<br/>Standard RoPE"]

    L6 -.-> GD["GLOBAL ATTENTION<br/>16 Q heads × 512<br/>1 KV head × 512<br/>K = V projection<br/>Full context<br/>Proportional RoPE"]

    %% ============================================================
    %% MLP
    %% ============================================================

    L1 -.-> MLP["Every Transformer Layer<br/>Gated MLP<br/>3840 → 15360 → 3840<br/>GELU(tanh approximation)"]

    %% ============================================================
    %% NORMALIZATION
    %% ============================================================

    L1 -.-> NORM["4 RMSNorms/layer<br/>Pre-Attention<br/>Post-Attention<br/>Pre-FFN<br/>Post-FFN<br/>ε = 1e-6"]
```

---

 # 22\. The critical implementation constants

 If you were writing `Gemma4_12B_UnifiedConfig`, these are the numbers I would regard as **non-negotiable** for architectural fidelity:

```
vocab_size = 262_144

hidden_size = 3_840
intermediate_size = 15_360

num_hidden_layers = 48

num_attention_heads = 16
num_key_value_heads = 8

global_head_dim = 512
head_dim = 256
num_global_key_value_heads = 1

sliding_window = 1_024

max_position_embeddings = 262_144

attention_dropout = 0.0
attention_bias = False

hidden_activation = "gelu_pytorch_tanh"

rms_norm_eps = 1e-6

final_logit_softcapping = 30.0

tie_word_embeddings = True

attention_k_eq_v = True

num_kv_shared_layers = 0

enable_moe_block = False

use_double_wide_mlp = False

use_bidirectional_attention = "vision"
```

 And:

```
layer_types = (
    ["sliding_attention"] * 5 +
    ["full_attention"]
) * 8
```

 That is the backbone configuration exposed by the released implementation.  Hugging Face+1

---

 # 23\. The vision constants

```
patch_size = 16
pooling_kernel_size = 3

model_patch_size = 48

patch_dim = 48 * 48 * 3
# = 6912

mm_embed_dim = 3840
output_proj_dims = 3840

mm_posemb_size = 1120

default_num_soft_tokens = 280

supported_soft_tokens = [
    70,
    140,
    280,
    560,
    1120
]
```

 These values are documented in the released configuration and model documentation.  Hugging Face+1

---

 # 24\. Audio constants

```
sample_rate = 16_000

audio_samples_per_token = 640

audio_embed_dim = 640

audio_output_dim = 640

# shared multimodal projection:
640 → 3840
```

 Thus:

```
1 audio token = 40 ms
```

 and:

```
30 sec audio
≈ 30 × 16000 / 640
≈ 750 audio tokens
```

 The model documentation confirms the 640-sample framing and lack of temporal downsampling.  GitHub

---

 # 25\. One subtle point: don't confuse "Unified" with "everything is identical"

 There are actually **three different dimensional regimes**:

```
                 Gemma 4 12B Unified

Text hidden
   3840
     │
     ├───────────────┐
     │               │
Sliding Attention  Global Attention
     │               │
     ↓               ↓
 head = 256       head = 512
 KV = 8           KV = 1
     │               │
     └───────┬───────┘
             │
           3840
```

 and before that:

```
Image:
6912 → 3840

Audio:
640 → 3840
```

 So the **fusion space** is 3840, but the internal attention geometry is not uniform across the 48 layers.

 That is a very important implementation detail.

---

 # 26\. What is still _not_ publicly specified enough to claim exact reproduction?

 This is where I would draw a line between:

 ### Publicly specified

 We can reproduce with high confidence:

 - number of layers
- hidden dimension
- FFN dimension
- attention dimensions
- layer pattern
- sliding window
- vocabulary
- RoPE configurations
- RMSNorm
- MLP structure
- vision patch geometry
- audio framing
- multimodal projection
- embedding tying
- logit softcap
- attention mask architecture

 The released configuration and implementation provide these details.  Hugging Face+2

 ### Not equivalent to knowing the original training recipe

 Architecture alone does **not** tell us exactly:

 - the complete proprietary/pretraining mixture and sampling weights
- every data filtering rule
- exact curriculum
- exact optimizer schedule
- every augmentation
- every training-stage transition
- exact distributed-training topology
- exact initialization implementation beyond documented configuration
- every post-training data mixture

 So you can build an **architecturally faithful Gemma 4 12B implementation**, but that does not mean an independently trained model will reproduce Google's checkpoint.

 The official technical report describes Gemma 4 as open-weight and gives the high-level architecture/training story, while the released implementation exposes the actual model configuration and forward-pass structure.  arXiv+1

---

 # 27\. The implementation hierarchy I'd recommend

 If you're actually going to implement it, don't start with a giant 12B model.

 Build these components in this exact order:

```
Phase 1
│
├── RMSNorm
├── GELU-tanh
├── Gated MLP
└── Token embedding
        ↓
Phase 2
│
├── Sliding GQA
├── 1024 causal window
└── Standard RoPE
        ↓
Phase 3
│
├── Global MQA
├── 512 head dimension
├── 1 KV head
├── K=V projection
└── Proportional RoPE
        ↓
Phase 4
│
└── 48-layer hybrid stack
        ↓
Phase 5
│
├── Raw-image patch embedder
└── Factorized 2D position
        ↓
Phase 6
│
└── Raw-audio 640-sample embedder
        ↓
Phase 7
│
└── Multimodal attention masks
        ↓
Phase 8
│
└── Tied LM head + softcap
```

 That gives you a **true architecture implementation**, rather than just wrapping an existing model.

 For reference, the closest implementation-level sources are the released `config.json` and Hugging Face's `modeling_gemma4_unified.py`; the latter is over 1,000 lines and exposes the actual attention, normalization, MLP, vision, audio and multimodal-forward implementations.  Hugging Face+1

 Gemma 4 12B configuration  ·  Gemma 4 Unified implementation  ·  Gemma 4 Unified documentation

 **The next useful step would be to turn this architecture into a second, even more detailed Mermaid diagram at the tensor level — showing every tensor shape (`B×S×3840`, `B×16×S×256`, `B×8×S×256`, masks, RoPE dimensions, residual paths, image-token insertion and audio-token insertion) for both training and autoregressive inference.** That would be close to a blueprint from which you could implement the model in PyTorch/JAX without having to infer the architecture yourself.
