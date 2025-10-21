**Reinforcement Learning for LLMs.** Reinforcement learning (RL) is increasingly adopted to enhance LLM reasoning (Wu, 2025); Luo et al. (2025b), as demonstrated by OpenAI's o1/o3/o4 (OpenAI et al., 2024b). DeepSeek-R1 (DeepSeek-Al et al., 2025) achieves comparable capabilities and further introduces the Group Relative Policy Optimization (GRPO) (Shao et al., 2024) for scalable end-to-end training. GRPO-based reasoning has been extended to tasks such as visual understanding (Shen et al., 2025), logical reasoning (Xie et al., 2025), and program synthesis (Ma et al., 2025). RL-enhanced agents have also shown strong performance in multi-turn interaction (Lu et al., 2025; Feng et al., 2025a) and open-domain retrieval (Jin et al., 2025a; Song et al., 2025; Zheng et al., 2025; Sun et al., 2025), highlighting RL's potential in agentic GraphRAG frameworks (Gao et al., 2025).

# 3 PRELIMINARIES

We formalize the GraphRAG pipeline into three stages as detailed below:

(a) **Knowledge Graph Construction.** This stage extracts structured relational facts from raw text. Given a knowledge collection $K = \{d_1, d_2, \dots, d_N\}$, the goal is to extract facts $f_d$ from each semantic unit $d \in K$ and aggregate them into a unified graph $\mathcal{G}_K$:

$$
\mathcal{G}_K \sim \sum_{d \in K} \pi_{\text{ext}}(f_d | d), \quad (1)
$$

where $\pi_{\text{ext}}$ denotes an LLM-based extractor that parses each $d$ into a set of relation-entity pairs $f_d = \{(r_i, v_r, r_i)\}$, with $r_i$ as the relation and $v_r = \{v_1, \dots, v_n\}$ the participating entities.

(b) **Graph Retrieval.** Graph retrieval is formulated as a two-step process over $\mathcal{G}_K$: (1) retrieving candidate reasoning paths and (2) pruning irrelevant ones. Conditioned on a query $q$, the model first retrieves a candidate set $\mathcal{X}_q = \{x_1, \dots, x_m\}$ and then selects a relevant subset $\mathcal{Z}_q \subseteq \mathcal{X}_q$. The overall objective is to maximize the expected joint likelihood of the two steps:

$$
\max_q \mathbb{E}_{\mathcal{Z}_q \sim P(\mathcal{Z}_q | \mathcal{G}_K)} \left[ \prod_{t=1}^{T_x} P_0(x_t | x_{<t}, q, \mathcal{G}_K) \cdot \prod_{t=1}^{T_z} P_0(z_t | z_{<t}, \mathcal{X}_q, q) \right], \quad (2)
$$

where $T_x$ and $T_z$ denote the number of retrieved and selected paths, respectively.

(c) **Answer Generation.** Given a query $q$ and selected paths $\mathcal{Z}_q$, answer generation produces a natural language answer $y$ grounded in graph-based evidence, formulated as:

$$
P(y \mid q, \mathcal{G}_K) = \sum_{\mathcal{Z}_q \subseteq \mathcal{X}_q} P(y \mid q, \mathcal{Z}_q) \cdot P(\mathcal{Z}_q \mid q, \mathcal{G}_K), \quad (3)
$$

where $P(y \mid q, \mathcal{Z}_q)$ is generation likelihood and $P(\mathcal{Z}_q \mid q, \mathcal{G}_K)$ is retrieval-pruning distribution.

# 4 METHODOLOGY: GRAPH-R1

In this section, as illustrated in Figure 3, we introduce Graph-R1, including agent initialization, multi-turn graph interaction, and outcome-directed end-to-end reinforcement learning.

## 4.1 KNOWLEDGE CONSTRUCTION AND AGENT INITIALIZATION

Graph-R1 adopts an LLM-driven agent, initialized with a knowledge hypergraph environment $\mathcal{G}_H$, the action space $A$, the state space $S$, and the answer target $y_q$ for the given query $q$.

**Graph Environment $\mathcal{G}_H$.** To support agentic reasoning, we propose a lightweight method for constructing a knowledge hypergraph $\mathcal{G}_H$ from given domain knowledge $K = \{d_1, d_2, \dots, d_N\}$. For each chunk unit $d \in K$, an LLM-based extractor $\pi_{\text{ext}}$ identifies $m$ n-ary relational facts, where each comprises a semantic segment $h_i$ and a set of participating entities $V_h = \{v_1, \dots, v_n\}$. A shared encoder $\phi(\cdot)$ is then used to generate semantic embeddings for both entities and relations:

$$
\mathcal{G}_H = (V, \mathcal{E}_H, \phi), \quad \text{where } \pi_{\text{ext}}(d) \to \{(h_i, V_h)\}_{i=1}^m, \phi(v) = \text{Enc}(v), \phi(h_i) = \text{Enc}(h_i), \quad (4)
$$

where each $h_i$ defines a hyperedge $h_i \in \mathcal{E}_H$ connecting its associated entities $V_h$, as $v \in V$. The resulting hypergraph $\mathcal{G}_H$ encodes high-order relational structures with rich semantic grounding.