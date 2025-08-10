# 1) Project Overview

* **Problem:** Multi-hop supporting-paragraph selection (≤3 hops).
* **Input:** Question $Q$, candidate paragraphs $\{P_i\}_{i=1}^n$.
* **Output:** Set/chain of supporting paragraphs.
* **Approach:** Graph-aware encoder (GNN) + prefix-conditioned scorer + frontier-chain search.

# 2) Data Specification

* **Raw fields (per example):**

  * `id: <string>`
  * `question: <text>`
  * `paragraphs: List[str]` (length = n)
  * `labels: Dict`

    * `support_indices: List[int]` (size ≤ 3)
    * `ordered_chain: Optional[List[int]]` (if known)
* **Embeddings (precompute or on-the-fly):**

  * `emb_q: R[d_txt]`
  * `emb_p: R[n, d_txt]`
* **Auxiliary features (per paragraph):**

  * `sim_qp: R[n]` (cosine)
  * `overlap_qp: R[n]` (entity/token overlap)

# 3) Graph Construction Template

* **Nodes:** paragraphs (0..n-1); optional question node `q_idx = n`.
* **Node feature $x_i$:** `[emb_p[i] || sim_qp[i] || overlap_qp[i]]`
  **Question feature $x_q$:** `[emb_q || 1 || 1]`
* **Edges:**

  * Compute `sim_pp: R[n,n]` (cosine).
  * **Prune strategy:**

    * `top_k_neighbors = <8|10|12>` (symmetrize), **no self-loops** for max diversity
    * **OR** threshold: `tau_sim = <0.5–0.7>` with min-degree=3+ fallback
  * Add `q_idx -> top_kq` paragraph edges by `sim_qp`.
* **Edge features $e_{ij}$:** `[sim_pp[i,j], overlap_pp[i,j], sim_qp[i], sim_qp[j]]`
* **Masks/labels:**

  * `is_para_mask: Bool[n(+1)]`
  * `y: Int[n]` (0/1 per paragraph)

# 4) Model Specification

## 4.1) Encoder (Graph-aware)

* **Layer:** GATv2 / Graph Transformer (edge-aware).
* **Depth:** `L = <2|3>`, hidden `d = <256>`, heads `<4>`, dropout `<0.1–0.2>`.
* **Block:** `Norm(ReLU(GNN(h, edge_index, edge_attr))) + residual`.

## 4.2) Prefix-conditioned Scorer

* **Context $c_t$:** mean/attn pool over `{question} ∪ prefix_nodes`.
* **Score:** `s_i^(t) = MLP([H_i || c_t])` for candidates not yet used.
* *(Optional)* 1-layer **conditioning GNN** per hop with a "path token" seeded by $c_t$.

# 5) Training Plan

* **Teacher forcing over hops** $t=1..K$ with $K\le3$.
* **Loss per hop:**

  * *Ordered:* BCE/CE where positive is the gold $p_t$.
  * *Unordered:* BCE with positives = remaining golds not yet chosen.
* **Auxiliary node loss:** BCE over $H$ to mark supports (weight `lambda_aux = <0.2>`).
* **Optimizer:** AdamW(lr=`2e-4`, wd=`1e-4`), epochs `<20–30>`, early stop on dev F1.
* **Imbalance:** `pos_weight` or focal loss (`gamma=2`).

# 6) Inference (Frontier-Chain Search)

* **Parameters:** `chain_width B=<4>`, `max_hops K=<3>`, `top_k_expand=<5>`, `stop_threshold τ=<0.25 prob or logit cutoff>`.
* **Procedure:**

  1. Encode once → $H$.
  2. Start search with empty prefix.
  3. For each hop: for each chain item, build $c_t$, score frontier, extend with top-k; keep global top-B.
  4. Stop at K hops or when best next score < τ.
  5. Return best chain; support set = chain nodes.

# 7) Evaluation Metrics

* **Node metrics:** F1@k (k=#gold), Precision@k, MAP — macro over questions.
* **Chain metric:** exact match (if order available).
* **Calibration:** tune τ, B, top_k on dev.

# 8) Configuration Template (YAML)

```yaml
model:
  encoder: gatv2
  hidden_dim: 256
  layers: 2
  heads: 4
  dropout: 0.15
  use_question_node: true
scorer:
  type: mlp
  context_pool: mean
  conditioning_layer: false
graph:
  top_k_neighbors: 10
  top_k_question: 6
  keep_self_loops: false  # Better multi-hop diversity
  pruning_strategy: top_k  # or 'threshold'
  similarity_threshold: 0.6  # for threshold strategy
  min_degree: 3  # fallback connectivity
  symmetrize: true
  edge_features: [sim_pp, overlap_pp, sim_qp_src, sim_qp_dst]
training:
  max_hops: 3
  loss: bce
  aux_node_loss_weight: 0.2
  optimizer: adamw
  lr: 2.0e-4
  weight_decay: 1.0e-4
  epochs: 25
  batch_size: 16
  pos_weight: 4.0
inference:
  chain_width: 4
  top_k_expand: 5
  stop_threshold: 0.25
evaluation:
  metrics: [f1_at_k, precision_at_k, map, chain_em]
```

# 9) Module Layout

```
project/
  config.yaml
  data/
    make_embeddings.py
    build_graph.py
  models/
    encoder.py          # GraphEncoder (GATv2/Transformer)
    scorer.py           # PrefixScorer (+ optional NodeHead)
  train.py              # teacher-forcing, hop losses, aux loss
  infer.py              # frontier-chains
  metrics.py            # F1@k, MAP, chain EM
  utils/
    entities.py         # NER, overlap
    prune.py            # top-k / threshold pruning
    batching.py         # PyG DataLoader helpers
    seeds.py            # reproducibility
  README.md
```

# 10) Python Interface Stubs

```python
# data/build_graph.py
def build_graph(example, cfg):
    """
    Inputs: example = {id, question, paragraphs, support_indices, ordered_chain?}
    Returns: torch_geometric.data.Data with fields:
      x, edge_index, edge_attr, y, is_para_mask, q_idx
    """
    ...

# models/encoder.py
class GraphEncoder(nn.Module):
    def __init__(self, d_node:int, d_edge:int, hidden:int, layers:int, heads:int, dropout:float):
        super().__init__()
        ...
    def forward(self, x, edge_index, edge_attr):
        """Return H: (N(+1), hidden)"""
        ...

# models/scorer.py
class PrefixScorer(nn.Module):
    def __init__(self, hidden:int):
        super().__init__()
        ...
    def context(self, H, q_idx:int|None, prefix_ids:list[int]) -> torch.Tensor:
        ...
    def forward(self, H, q_idx:int|None, prefix_ids:list[int], candidate_ids:torch.Tensor) -> torch.Tensor:
        """Return logits: (M,) for candidates"""
        ...

# train.py
def train_epoch(loader, encoder, scorer, optim, cfg):
    """
    Teacher forcing over hops:
      for t in 1..K:
        prefix = gold_prefix(t)
        candidates = frontier(prefix)
        logits = scorer(H, q_idx, prefix, candidates)
        loss += hop_loss(logits, labels_t)
      + aux node loss
    """
    ...

# infer.py
def infer_frontier_chain(graph_data, encoder, scorer, cfg) -> list[int]:
    """
    Encode once -> H; frontier-chain search over hops to return best chain indices.
    """
    ...
```

# 11) Implementation Checklists

**MVP (Minimum Viable Product)**

* [ ] Embeddings + entity overlap
* [ ] Graph build + pruning + edge features
* [ ] Encoder (2-layer GATv2) + PrefixScorer (MLP)
* [ ] Training loop (teacher forcing, hop BCE, aux node BCE)
* [ ] Inference (frontier-chain)
* [ ] F1@k evaluation

**Phase 2 (Nice-to-have)**

* [ ] Conditioning GNN per hop
* [ ] Mutual-kNN pruning ablation
* [ ] Focal loss + calibration curve
* [ ] Attention visualization (edges on selected chain)

# 12) Paper-friendly Naming Conventions

* **Reasoning graph**, **reasoning chain**, **frontier set**, **frontier-chain search**.
* In text: "We perform **frontier-chain search** over candidate **reasoning chains**, scoring each next node with a **prefix-conditioned compatibility function** computed from **graph-aware embeddings**."
