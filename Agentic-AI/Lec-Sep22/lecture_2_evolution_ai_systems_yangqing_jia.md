
# Lecture 2: Evolution of System Designs from an AI Engineer Perspective

**Speaker:** Yangqing Jia, VP at NVIDIA

## Background

Berkeley PhD (2009–2013), creator of Caffe, and contributor to TensorFlow and PyTorch. Founded Lepton AI, which was acquired by NVIDIA. The lecture focuses on the evolution of AI systems across algorithms, applications, and infrastructure, drawing on experience moving from research to engineering to entrepreneurship.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## Core Themes Overview

The lecture covers four main topics: LLM algorithms, the thriving application space, the emergence of AI Infrastructure as the third pillar of enterprise IT, and how hardware and software designs are seeing historical concepts reappear. The fundamental design consideration across all areas is the unique nature of AI workloads, which are distinct from conventional computer science paradigms.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## 0. Demystifying LLMs and AGI: The Chinese Typewriter Problem

Yangqing Jia suggests that while LLMs are undeniably getting smarter, their foundation rests on simple principles learned in basic computer science.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## The Chinese Typewriter Analogy

Historically, Chinese typing required a massive keyboard (3,000–5,000 characters) and a mechanical handle that operators had to move to select metal keys. The core problem was efficiency: minimizing the distance the handle had to move between characters.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

* **Early Solution:** Libraries started counting character pairings (what we now call bigrams and n-grams) to group characters that frequently occurred together, significantly improving typing speed compared to a random keyboard layout[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)
* **Modern Parallel (Next Token Prediction):** This process of counting and predicting what you would type next is analogous to the concept of next token prediction used in modern Large Language Models (LLMs)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)
* **Scale and Context:** Conventional Natural Language Processing (NLP) was constrained to looking at about four or five bigrams. Today, models like GPT use history, and large models have context lengths that can span millions of tokens, allowing them to compress massive amounts of knowledge into the capability to predict the next token[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## 1. Algorithmic Evolution Driving LLM Capabilities

The field has not yet seen a plateauing effect in model quality, with continuous innovation occurring across both closed-source and open-source models.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## Competitive Landscape

Closed-source models (e.g., GPT-5, Grok, Gemini, Kimi) continue to lead in absolute quality, particularly in reasoning. However, the quality gap is shrinking, moving from over a year in 2023 to an estimated six months today, as open-source models (DeepSeek, Qwen) rapidly catch up.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## Key Algorithmic Waves

| Algorithm Wave (Date)                              | Description / Impact                                                                                                                                                                                                                | Analogies                                                                                                              |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| GPT (3.5) (Nov 2022)                               | Structural innovation fundamentally improving capabilities, such as text understanding[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                                                                                           | AlexNet[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                                                             |
| Mixture-of-Experts (MoE) (Dec 2023)                | Uses massively sparse and sparsely activated models to improve efficiency and decrease size while maintaining quality[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                                                            | Ensemble Learning, Inception/ResNet (using sparse parameterization)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo) |
| Test-Time Scaling (Sep 2024)                       | Models "mumble longer" during testing time, reflecting on intermediate results to achieve better outcomes[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                                                                        | Fully Convolutional Networks, Multi-instance learning[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)               |
| Reinforcement Learning (RL) (Jan 2025 and earlier) | Allows for the definition of a more principled and sophisticated loss function tied directly to the final end result, rather than just using the next token prediction objective[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo) | General RL, GANs[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                                                    |

## 2. Application Space Thrives

Model consumption is seeing substantial growth, reinforcing the need for scalable systems. The total token consumption has surged about 10x in the last year, reaching around 4.9 trillion tokens per day on platforms like OpenRouter.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## Consumer Applications (ToC)

The consumer app landscape is highly fluid and competitive. Top application areas include Chat (ChatGPT, Grok, Gemini), Coding (Cursor, Replit), and Search (Perplexity).[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**Willingness to Pay:** Revenue is largely driven by prosumers—small consumers who embed AI tools into their daily work and productivity tasks :[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

* Examples include Cursor (embedded coding practices), Runway (multimedia creativity), and Cloud (a recording device that uses LLMs to output meeting summaries, action items, and debating points for legal or consulting industries)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)
* People are often willing to spend a couple of hundred dollars per month on SaaS tools that boost productivity[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## Enterprise Applications (ToB)

The enterprise market is growing faster than conventional ones, yet slower than the consumer side. It is considered hopeful and nascent.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**High Barrier to Entry:** Enterprise applications have a unique barrier due to the required "dirty work," demanding accuracy, robust integration of authentication/access control (ACL), and platform unification.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**Example - Enterprise Search:** Startups like Glean solve pain points by connecting to multiple internal data sources (Google Drive, GitHub, internal wiki) and indexing them based on the user's access rights.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**The Role of RAG:** Retrieval Augmented Generation (RAG) remains highly important, especially for enterprise and vertical applications, where accuracy over indexed facts is critical and general search (common knowledge) is insufficient. An effective RAG pipeline often uses multiple stages: a weaker model for coarse ranking (optimizing cost) followed by an LLM for fine-grained ranking (optimizing accuracy).[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**Synergy:** Model builders and application companies (like Duolingo) are forming complementary relationships, using AI models to efficiently create interactive content and leveraging deep domain expertise for a 1+1 > 2 effect.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## 3. AI Infrastructure as the Third IT Pillar

AI infrastructure has established itself as a distinct third pillar in IT strategy, alongside the earlier Web Service Cloud and Data Cloud. This trend is driven by "The Bitter Lesson": the most effective way to solve AI problems is through general methods applied with a large amount of computation and data.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## Historical Evolution of IT Pillars

| Era                          | Focus                                                                                                                | Key Characteristic                                                                                             |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Scientific Computing (1970s) | Large-scale physics/weather simulations[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                           | Matrix multiplication on CPU clusters[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                       |
| Web Service Cloud (2000s)    | Web serving, microservices (AWS)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                                  | IO >> Compute, Embarrassingly Parallel[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                      |
| Data Cloud (2010s)           | E-commerce, processing exabyte-scale data (Snowflake, Databricks)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo) | IO >> Compute, Distributed systems (MapReduce/Spark)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)        |
| AI Cloud (2020s)             | Modern AI applications, high performance, heterogeneous infra[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)     | Compute >> IO, Exaflops computation power, Very Distributed[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo) |

## Differences Between Conventional and AI Cloud

AI workloads present unique challenges because they are compute-dominant and require large amounts of computation and communication, similar to scientific computation.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

| Feature                  | Conventional Cloud                                                                              | AI Cloud                                                                                                                                               |
| ------------------------ | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Workload                 | Varied (compute, storage, network, database)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo) | Unified (Numerical computation, matrix multiplication)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                                              |
| Supply Chain Flexibility | High (CPU virtualization, easy migration)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)    | Low (Large training jobs are hard to live migrate; highly specialized GPUs are not interchangeable)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo) |
| Distributed Training     | Highly available/fault-tolerant[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)              | Fragile (If one GPU machine fails, the whole training job often needs restarting)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                   |

## System Design Challenges and Best Practices

The emergence of Neocloud companies (e.g., CoreWeave, Lambda) focuses on aggregating GPUs and specialized AI workloads.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**Developer Efficiency:** Researchers prefer a simple mindset, requesting full machines and running jobs using MPI (like the older Slurm systems). They generally dislike the Kubernetes abstraction often favoured by Site Reliability Engineers (SREs), which complicates concepts like parallelism and completion.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**Infrastructure Efficiency:** GPUs fail more frequently than expected due to software bugs, faulty hardware, and network issues. Thus, abstractions are needed to hide operational burdens from developers.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**Best Practices for AI Platforms:**

1. **Multi-cloud Supply Chain Management:** Necessary due to the chronic shortage and expense of GPUs[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)
2. **Elasticity and Utilization Management:** Essential for maximizing the use of expensive GPU hours[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)
3. **AI-Native Platform:** Building a control plane to unify development, training, and inference, abstracting away Kubernetes jargon and handling orchestration faults (e.g., using frameworks like Ray/Anyscale or SkyPilot)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## 4. Hardware and Software Design: Back to the Future

Hardware and software design are seeing a return to concepts previously used in large-scale computation.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## Integrated Systems (DGX and NVL72)

Traditional cloud (Open Compute Project/OCP) emphasized modular, small machines that operate independently (good for microservices). Modern AI systems, however, are moving toward tightly integrated, mainframe-like architectures.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**DGX Boxes:** Beefy servers (4U rack units) hosting eight GPUs and two CPUs.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**NVL72 System:** A rack-level system integrating servers with high-bandwidth NV switches. Critically, this design allows machines to directly access other machine's memory within the rack without requiring permission (similar to UPC in HPC), behaving like one large, single computer.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

**Benefits:** This integration simplifies tasks like prompt caching and allows massive models to be deployed without needing to worry about chopping them into pieces to fit individual machines.[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## The Scale of Growth

The increased scale of models drives these hardware innovations :[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

* The state-of-the-art computer vision model, GoogLeNet, had about 6 million parameters[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)
* Today, an 8 billion parameter model (LLaMa 8B) is considered very small[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)
* This massive increase in parameters necessitates rack-level scaling for training and inference and an order of magnitude increase in computation and memory[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## Practical Takeaways and Role Perspectives

## Practical Takeaways for System Design

For companies building AI applications, operational wisdom from conventional cloud services remains helpful, but specific AI-native discipline is needed :[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

* **Design for Failure:** Assume accelerator faults and architect job orchestration for preemptions and restarts[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)
* **Unify the Platform:** Streamline development, training, and inference (dev→train→infer) in one control plane, including first-class primitives for RAG, data indexing, evaluation, and security[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)
* **Plan for Inference:** Model choice, context length, and test-time scaling directly influence cost and latency; budget and Service Level Objectives (SLOs) must shape system defaults[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## Startup Challenges

While the market shows solid traction, one of the biggest challenges encountered during the entrepreneurial phase (Lepton AI) was not purely technical, but GPU supply chain management. Due to chronic shortages and high arbitration, founders spent significant time procuring GPUs and negotiating flexible contracts (e.g., six-month terms, committed spending with flexible scale up/down).[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)

## Role Perspective Table

| Role         | Primary focus                                                                                                                                                       | System implications                                                                                                                                                                      |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Researcher   | Explore architectures (MoE, RL) and training signals to unlock new capabilities and constraints[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                  | Must anticipate shifting bottlenecks and evaluation methods that alter deployment patterns[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                                            |
| Engineer     | Translate research into reliable frameworks (Caffe, PyTorch)[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                                                     | Must operationalize distributed training/inference and harden pipelines against hardware and orchestration faults[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo)                     |
| Entrepreneur | Build AI-native cloud that unifies dev, train, and infer, aligned to enterprise needs and supply-chain realities[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo) | Must prioritize multi-cloud flexibility, cost-aware inference scaling, and platform UX that balances rapid iteration with reliability[youtube](https://www.youtube.com/watch?v=xqRAS6rAouo) |

1. [https://www.youtube.com/watch?v=xqRAS6rAouo](https://www.youtube.com/watch?v=xqRAS6rAouo)
