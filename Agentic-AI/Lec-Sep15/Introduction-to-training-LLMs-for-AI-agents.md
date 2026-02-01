# Title : [Introduction to training LLMs for AI agents](https://www.perplexity.ai/search/agentic-ai-course-by-rdi-berke-YxFeCqUPQrGxIfXt8Z3oUg#5)

We will focus onto the pretraining, reasoning RL, and post‑training LLMs — to move from prediction to reliable task execution.[3]

**LLMs & chatbots took over the world like ChatGPT ?**
LLMs and chatbots achieved unprecedented consumer scale, with ChatGPT reaching an estimated 100 million monthly active users just two months after launch and climbing to hundreds of millions weekly by 2025, which set expectations for models that can reason, align to preferences, and act as agentic systems.[1][2]

Training those systems follows a simple core: massive next‑token pretraining for capability, plus post‑training (instruction tuning and RLHF) for usefulness and safety, with optional reasoning RL and tool-use training to turn assistants into agents.[3][4][5]

**Why they exploded as usefull AI tool for human ?**
ChatGPT set the record for the fastest-growing consumer app by hitting ~100 million monthly users two months post‑launch, illustrating a step‑change in accessibility and utility.[1]
Industry trackers report hundreds of millions of weekly users in 2025, signaling sustained mainstream adoption that drives investment into reliability, safety, and task completion.[2]

**What could be possible training pipeline for building them ?**
It begins with large‑scale next‑token pretraining on diverse, filtered text using a decoder‑only Transformer, which builds broad knowledge and latent skills.[5]
Post‑training then adds instruction supervision and preference optimization (e.g., RLHF or related methods) so models follow instructions, align with user preferences, and avoid unsafe behavio+r.[4]
For agents, a reasoning RL stage can be inserted to improve step‑by‑step problem solving, followed by evaluation loops that measure task success rather than only static benchmarks.[3]

Thus modern pipeline would be pretraining -> reasoning RL -> post-training.
**What agents add to the llms capabilities ?**
Agentic systems extend chat behavior with tool use, browsing, and multi‑turn planning; they are trained and evaluated on trajectories and outcomes rather than single responses.[6]

Ex-1 : Meta's Llama family : They pretrain a strong foundation, then apply instruction tuning and alignment to produce helpful assistants.[5]

Ex-2 : DeepSeek‑R1 explicitly uses reinforcement learning to incentivize multi‑step reasoning, then distills those behaviors into smaller models initialized from bases like Llama/Qwen.[7][8]

Ex-3 : Moonshot’s Kimi‑Researcher trains end‑to‑end with RL over full trajectories to learn planning, search, and tool use for complex research tasks.[6]

A practical guide to build agentic systems

- Start with a solid pretrained base (e.g., Llama‑x foundation) and apply supervised instruction tuning on high‑quality assistant data.[5]
- Add preference optimization: RLHF if a reward model and RL stack are available, or direct preference optimizers as infrastructure‑lighter alternatives.[4]
- If deep reasoning or long‑horizon tasks are goals, include a reasoning RL stage and/or distillation from a stronger reasoning teacher.[8][7]
- For agent behavior, design end‑to‑end tasks that reward successful tool use and multi‑turn completion, and evaluate on outcome‑based metrics.[6]

## General LLM modern training pipeline

LLM training pipeline is first pretraining for general capability, reasoning RL for step‑by‑step problem solving, and classic post‑training/RLHF to maximize user utility and safety. For scale training pipeline, Llama 3 was pretrained on over 15T tokens which aligns with the >10T tokens, months, multi-million-dollar compute.

1. **Pretraining**
   Pretraining teaches broad world knowledge by predicting the next token over a massive filtered corpus, typically running for months and dominated by data and compute bottlenecks. Open source models like Llama 3 explicitly cite “over 15T tokens,” illustrating the modern scale underpinning today’s assistants and agents.

- Data: Internet‑scale, filtered text, e.g., Llama 3 “over 15T tokens.”
- Time: Months for large runs due to data/compute throughput limits
- Compute cost: > $10M for frontier‑scale regimes, driven by GPU time and scale.
- Bottleneck: Data quality/quantity and compute throughput.
- Example: Llama 3 pretraining stage.

2. **Reasoning RL**
   Reasoning RL optimizes models on tasks with objective answers (math, coding, science) to elicit reliable multi‑step solutions beyond imitation, as exemplified by DeepSeek‑R1’s RL‑driven reasoning stages.

The Practical scale heuristics for it that highlights the engineering overhead of RL environments and stabilizing “hacks.”

- Data: ~1M problems for objective, auto‑verifiable tasks.
- Time: Weeks for iterative RL loops and evaluations.
- Compute cost: > $1M depending on model size and rollouts.
- Bottleneck: RL environments, reward design, and stabilization tricks.
- Example: DeepSeek‑R1’s multi‑stage RL for reasoning.

3. **Classical Post-training/RLHF**
   Classic post‑training steers a capable base into a helpful assistant via supervised instruction tuning and preference optimization (e.g., RLHF, DPO), the recipe popularized post‑GPT‑3 and formalized in many current stacks. It is faster and cheaper than pretraining but highly sensitive to data quality and evaluation design for utility and safety.

- Data: ~100k problems/interactions for SFT and preference optimization.
- Time: Days for well‑engineered pipelines on established bases.
- Compute cost: > $100K depending on model size and method (SFT, RLHF, DPO).
- Bottleneck: High‑quality instruction/preference data and robust evals.
- Example: Llama “chat/instruct” variants derived from pretrained bases.

Post‑training is everything after base pretraining, encompassing supervised instruction fine‑tuning and preference optimization as well as online RL variants that optimize behavior and utility.

## Pre-training

The pretraining stage for LLMs, aligned with the way modern systems are built.

### Method

- Pretraining goal and task: teach broad world knowledge via autoregressive language modeling where the objective is to predict the next token given context using cross‑entropy loss.
- AR language models: inference runs as tokenize → forward pass → predict next‑token probabilities (softmax) → sample/decide → detokenize; training repeats this over sequences with teacher forcing.
- From n‑grams to neural LMs: estimate 
  $$
  P(X \mid \text{“the grass is”}) = \frac{\text{Count}(X, \text{“the grass is”})}{\text{Count}(\text{“the grass is”})}
  $$

  ; this fails because counts explode and strings are mostly unique, so generalization requires distributed representations via neural networks.

### Data

- Pretraining data: start from broad, “reasonable” internet text; quality matters more than any one source, so heavy filtering is critical (language ID, boilerplate/JS removal, near‑dup/dup removal, toxicity/NSFW blocklists, repetition and length filters).
- Example “FineWeb‑style” curation trajectory: crawl‑scale corpus reduces in stages, e.g., hundreds of billions of raw documents → tens of billions after base filters → ~20B after robust deduplication → final cleaned set supporting 10T–20T+ tokens for training.
- Midtraining: continued pretraining (< 1T tokens) to shift desired properties: more scientific/coding/multilingual data, document formats that favor instruction‑following, longer contexts (e.g., moving from 4k to 128k via long‑sequence curricula), and reasoning‑heavy sources.

### Compute

- Dominant driver: for fixed architecture/optimizer families, performance is largely governed by total training compute allocated across parameters and tokens; this enables extrapolation from small‑scale experiments via scaling laws.
- Scaling‑driven development: instead of hyperparameter sweeps on giant models, fit “scaling recipes” on small/medium models for a few days, then extrapolate to large runs and train the final model once for weeks.
- Architectural choice via scaling: transformers empirically show better constants and scaling slopes than legacy RNN/LSTM families for language modeling at scale.
- Chinchilla guidance: compute‑optimal allocation balances parameters and tokens; a common rule‑of‑thumb is around 20 tokens per parameter for training, while production often pushes higher tokens‑per‑parameter ratios (for lower inference cost) when budgets allow.
- Bitter lesson: favor simple, scalable methods that best leverage computation and data rather than intricate algorithmic tricks; the biggest gains tend to come from scale.

### Worked example (back‑of‑envelope)

- Target: a frontier‑scale model (e.g., 400B parameters) trained on 

  $$
  15.6
  $$

  T tokens.
- Training FLOPs approximation: 

  $$
  \text{FLOPs} \approx 6 N P = 6 \times 15.6 \times 10^{12} \times 405 \times 10^{9} \approx 3.8 \times 10^{25}
  $$

  .
- Throughput/cost: with 16k H100s at an effective 400 TFLOPS/GPU, total time 

  $$
  \approx \frac{3.8 \times 10^{25}}{400 \times 10^{12} \times 3600} \approx 26 \text{M GPU‑hours}
  $$

  ; calendar time 
  $$
  \approx \frac{26 \text{M}}{16{,}000 \times 24} \approx 70 \text{ days}
  $$

  ; at \$2/GPU‑hour, direct compute 
  $$
  \approx \$52
  $$

  M (excl. storage, networking, engineering).
- Carbon estimate: 

  $$
  26\text{M GPU‑h} \times 0.7 \text{ kW} \times 0.24 \text{ kg CO}_2\text{/kWh} \approx 4{,}400 \text{ tCO}_2\text{e}
  $$

   (very configuration‑dependent).

### One‑line summary

Pretraining: predict the next word on large‑scale internet text with > 10T tokens, running for months at > \$10M compute, bottlenecked by data and compute; examples include DeepSeek v3 and the next Llama generation.

## Post-training

Post‑training transforms a pretrained language model into a helpful assistant or capable reasoner by adding supervised instruction tuning and preference/RL optimization, because pure language modeling does not directly optimize for user utility or correctness.[1][2]
Since 2022, “classic” alignment (SFT → RLHF/DPO) became the norm for assistants, and since 2024, reasoning‑focused RL (e.g., DeepSeek‑R1) and test‑time scaling rose to prominence for complex tasks.[3][4][1]

### Method

- Supervised finetuning (SFT): finetune the pretrained model on input→desired output pairs using next‑token prediction to teach instruction following, formatting, early reasoning, tool‑use scaffolds, and other target behaviors.[2][5]
- Scalable SFT data via synthetic generation: seed with human exemplars, then expand with synthetic instruction–response pairs (e.g., Stanford Alpaca’s 52K self‑instruct set), often filtered or ranked before training.[6][7]
- Rejection sampling with verifiers: generate multiple candidates with a temporary model and keep only those that pass tests or rubric‑based checks (used in reasoning datasets and agent/tool‑use data creation).[8][3]
- Rich SFT pipelines for agents: Kimi K2 synthesizes large tool‑use corpora with simulated users, real and synthetic tools, and rubric‑based rejection to collect multi‑turn trajectories before RL.[9][8]
- Reinforcement learning for reasoning/agents: optimize behavior directly against rewards rather than cloning humans; reward sources include verifiable outcomes (unit tests, exact answers), reward models (RLHF), and LLM‑as‑a‑judge.[10][3]
- GRPO for reasoning RL: DeepSeek‑R1 uses Group Relative Policy Optimization to compute advantages over groups of sampled solutions without a value network, improving reasoning with scalable rollouts.[11][3]
- RLHF pipeline: sample pairs from an SFT policy, collect human or AI preference labels, and optimize with PPO or a direct method like DPO that avoids explicit reward modeling.[2][10]
- Test‑time scaling: allocate more inference compute (longer chains, multiple samples, self‑consistency) to boost correctness on hard problems, complementing training‑time RL.[4]

### Data

- Classic alignment data: 5k–500k instruction interactions capturing desired style, safety, and utility; small, high‑quality sets can move behavior substantially (e.g., LIMA shows strong alignment from ~1k curated pairs).[12][1]
- Reasoning data: hard, auto‑verifiable tasks (math/coding/science) to enable objective rewards and scalable filtering or rejection sampling for correctness.[3][11]
- Synthetic instruction data: Alpaca‑style distillation from stronger models, often followed by filtering/ranking before SFT to reduce noise and mode collapse.[7][13]
- Agent/tool‑use data: Kimi K2 builds large synthetic corpora by simulating tools, tasks, users, and environments with rubric‑based acceptance; multi‑turn trajectories are retained only when successful.[8][9]
- Preference labels and judges: human preference data remains gold standard but expensive; AI‑judging scales cheaply yet introduces bias risks like preference leakage that must be mitigated.[14][10]

### Compute

- SFT compute: comparatively modest versus pretraining; cost and time scale with dataset size and model size, with throughput dominated by efficient data pipelines and mixed‑precision training.[1][2]
- RL compute: sampling is the bottleneck because multiple rollouts per prompt are required; infra must support concurrent rollouts, verifiers, and long trajectories, as highlighted in Kimi K2’s agent RL setup.[3][8]
- Infrastructure tips: collocate engines and environments to minimize latency, parallelize sampling, and pause/trim long‑tail rollouts to keep utilization high during agent RL.[9][8]
- Preference optimization at scale: PPO requires stable reward modeling and KL control; DPO simplifies infra by optimizing directly from pairwise preferences when reward modeling or on‑policy RL is impractical.[10][2]

## Evaluation

Evaluation for LLMs splits into close‑ended tests with automatic correctness checks and open‑ended judgments based on human or AI preferences, and both are crucial for model selection and production readiness.[1][2]
Quantifying progress is sensitive to prompting and dataset effects, so rigorous, repeatable setups are needed to identify improvements, pick winners, and decide launch readiness.[3]

### Close‑ended

Close‑ended evaluation uses tasks with few possible answers so verification is automated, such as multiple‑choice accuracy on MMLU’s 57 subjects.[4][1]
Benchmarks like MMLU offer breadth and easy scoring, but results can vary with prompting and run settings, which motivates careful protocol control.[5][3]

- Example: MMLU measures percentage correct over 15,908 multiple‑choice questions spanning diverse disciplines and is widely used for model comparison.[1][4]
- Challenge: sensitivity to prompt formats, decoding settings, and inconsistencies across runs requires standardized harnesses for fair comparisons.[3][5]
- Contamination: train–test overlap can inflate leaderboards, leading to “contamination‑free” variants like MMLU‑CF; this is most critical for public reporting, though iterative internal development may weight it less.[6][7]

### Open‑ended

Open‑ended assistant behavior is hard to auto‑grade, so evaluations rely on human preference or AI‑judge comparisons between model outputs for the same prompt.[2][8]
Chatbot Arena implements blinded, pairwise human battles aggregated into Elo/BT‑style ratings to rank models “in the wild.”[8][9]

- Human preference evals: blinded pairwise comparisons (e.g., Chatbot Arena) provide high‑fidelity signals but are costly and slower to iterate.[9][2]
- LLM‑as‑a‑judge: AlpacaEval automates pairwise comparisons by asking a strong model to choose the better answer, achieving ~0.98 Spearman correlation with Arena while running in under 3 minutes and costing under $10.[10][11][12]
- Caveats: AI‑judge setups can introduce spurious correlations or bias, so many teams calibrate against Arena‑style human scores and monitor drift over time.[13][14]

## Systems

Modern LLM systems are bottlenecked by compute, memory, and interconnect, so practical performance comes from maximizing hardware utilization with precision, kernels, and parallelism rather than only buying more GPUs.[1][2]
The key techniques: mixed precision, operator fusion and tiling (e.g., FlashAttention), memory sharding (ZeRO), and data/pipeline/tensor parallelism, with MFU as the North Star metric.[3][4][5][6]

### Core problems

- GPUs are scarce and expensive, with multi‑GPU training limited by interconnect bandwidth and synchronization overheads, making resource allocation and optimized pipelines critical.[1]
- Progress is measured by model FLOPs utilization (MFU), the ratio of achieved throughput to hardware peak; PaLM formalized MFU and reported high utilization via systems co‑design.[2]

### GPU basics

- GPUs execute massively parallel kernels (SIMT), excelling at throughput on uniform operations across many threads.[1]
- Specialized tensor cores accelerate matrix multiplications significantly beyond general ALUs, making GEMM‑heavy kernels the sweet spot for performance.[6]
- Training is often memory‑ or IO‑bound rather than compute‑bound, so the challenge is keeping cores fed through better memory access and scheduling.[7]
- Memory hierarchy matters: on‑chip storage (registers/L1/SMEM) is fast but small, and HBM is large but slow, so algorithms that minimize HBM traffic win.[3]
- MFU: systems aim to raise the ratio of observed FLOPs to peak; MFU is widely used in LLM training efficiency reports and helps compare engineering choices.[2]

### Low precision

- Mixed precision reduces memory traffic and boosts tensor‑core throughput by using bf16/fp16 where safe, with accumulations/updates in higher precision to preserve stability.[6]
- Automatic Mixed Precision (AMP) orchestrates casting of weights/activations/gradients to low precision with safe higher‑precision updates, balancing speed and accuracy.[8]
- PyTorch and CUDA stacks recommend bf16 on modern hardware for its wider dynamic range and simpler scaling versus fp16, improving training stability.[9]

### Operator fusion and tiling

- Operator fusion reduces kernel launches and avoids writing intermediate tensors to HBM, letting compiled fused kernels keep data on‑chip longer.[10]
- torch.compile performs graph capture and kernel fusion to mitigate memory‑bound kernels by combining them with compute‑bound ones.[7]
- FlashAttention uses IO‑aware tiling and kernel fusion to compute exact attention with far fewer HBM reads/writes, yielding large end‑to‑end speedups on long sequences.[3]
- Follow‑ons like FlashAttention‑2/3 further improve parallelism and utilization on newer GPUs, sustaining attention efficiency at scale.[11][12]

### Parallelization

- Naive data parallelism replicates the full model on each GPU, summing gradients each step, increasing throughput but not reducing per‑GPU memory.[5]
- ZeRO sharding partitions optimizer states, gradients, and then parameters across data‑parallel ranks to remove DP redundancy, enabling much larger models.[4]
- With Adam‑style optimizers, full‑precision training typically needs about 16 bytes per parameter for weights, gradients, optimizer states, and master copies, motivating sharding and precision tricks.[13]
- Pipeline parallelism partitions layers across devices and streams micro‑batches to reduce idle “bubbles,” enabling near‑linear scale when well balanced.[14]
- Tensor (intra‑layer) parallelism splits large matrices within layers across devices, synchronizing partial results to train models with multi‑GPU layers.[5]
- State‑of‑the‑art systems combine data, pipeline, and tensor parallelism (“3D parallelism”) with careful schedules to scale to thousands of GPUs.[1]

### Memory and bandwidth optimizations

- Tiling orders computation to reuse tiles in fast memory before reloading from HBM, reducing global memory transactions in GEMM and attention kernels.[3]
- Kernel fusion avoids round‑trips to HBM between simple ops (e.g., linear + activation), cutting bandwidth pressure and launch overhead.[10]
- Compilers and vendor libraries increasingly generate fused kernels and Hopper‑specific instructions to approach GEMM‑like utilization for attention.[12]

### Architecture sparsity

- Mixture of Experts (MoE) activates only a subset of expert FFNs per token via gating, increasing parameter count without proportional FLOPs per token.[15]
- Switch Transformer simplifies routing to a single expert per token, reducing communication and improving training stability and speed at trillion‑parameter scales.[15]

### Practical rules of thumb

- Use bf16/AMP by default to save memory and boost throughput, falling back to fp16 with loss scaling or fp32 only when needed.[9][6]
- Track MFU and tokens/sec as primary KPIs; optimize fusion, tiling, and batch/micro‑batch sizes to raise utilization before adding more GPUs.[2]
- For memory: apply ZeRO Stage‑3 (and offload as needed), gradient checkpointing, and activation recompute to fit larger models or batches per GPU.[4]
- Scale with 3D parallelism: start with DP + ZeRO, then add pipeline and tensor parallelism via Megatron‑LM‑style partitioning.[5][1]
- Prefer IO‑aware kernels like FlashAttention for long contexts and enable torch.compile to harvest affordable fusion gains across the graph.[10][3]

## References

1. https://pytorch.org/blog/a-primer-on-llm-post-training/
2.
