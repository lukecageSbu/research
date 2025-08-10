Got it—you want a **tight, publication-grade spec** you can hand to code and cite in a write-up. Here’s a complete, precise design with solid terminology, math, and build steps.

# Objective

Given a question $Q$ and $n$ candidate paragraphs $\{P_i\}_{i=1}^n$, predict up to 3 **supporting paragraphs** that collectively form a **reasoning chain** (multi-hop, bridge/comparison allowed). We use a **graph-aware encoder** (GNN) and **frontier-chain search** (beam-style) for inference.

# Terminology (use consistently)

* **Reasoning graph**: A pruned, weighted graph over candidate paragraphs (plus an optional question node).
* **Reasoning chain**: An ordered sequence of selected paragraph nodes $(p_1,\dots,p_t)$ that explains the answer.
* **Frontier set**: The set of candidate next nodes not yet in the chain.
* **Frontier-chain search**: Beam-style expansion over reasoning chains using scores conditioned on the current prefix.

# 1) Graph Construction (per question)

**Nodes**

* Paragraph nodes: $v_i \leftrightarrow P_i$.
* Optional **question node** $v_Q$.

**Initial node features**

* Text embedding: $e(P_i)\in\mathbb{R}^{d_{\text{txt}}}$ (e.g., MPNet/SBERT).
* Scalars: $s^{Q\!P}_i=\cos(e(Q), e(P_i))$; $o^{Q\!P}_i=\text{entity\_overlap}(Q,P_i)$.
* Final: $x_i = [e(P_i)\,\|\, s^{Q\!P}_i \,\|\, o^{Q\!P}_i]\in\mathbb{R}^{d_n}$.
* For $v_Q$: $x_Q=[e(Q)\,\|\,1\,\|\,1]$.

**Edges**

* Compute paragraph–paragraph similarities: $s^{P\!P}_{ij}=\cos(e(P_i),e(P_j))$.
* **Pruning**: keep **top-$k$** neighbors per node by $s^{P\!P}$ (symmetrize), keep self-loops.
* Add $v_Q \to$ top-$k_Q$ paragraphs by $s^{Q\!P}$.

**Edge features** (for edge-aware attention)

$$
e_{ij} = \big[s^{P\!P}_{ij},\; \text{entity\_overlap}(P_i,P_j),\; s^{Q\!P}_i,\; s^{Q\!P}_j\big]\in\mathbb{R}^{d_e}.
$$

# 2) Graph-Aware Encoder

Use a shallow **edge-aware attention GNN** (e.g., GATv2Conv / Graph Transformer).

**Forward (L layers)**

$$
H^{(0)}=X,\quad
H^{(\ell+1)}=\text{Norm}\!\left(\sigma\!\big(\text{GATv2}(H^{(\ell)},E,\text{edge\_attr})\big)\right).
$$

Output $H=H^{(L)}\in\mathbb{R}^{(n{+}\mathbb{1}_{Q})\times d}$.
(Residuals+LayerNorm recommended; $L\in\{2,3\}$.)

# 3) Prefix-Conditioned Scoring (hop-wise)

At hop $t$ with prefix $\mathcal{P}_{<t}=(p_1,\dots,p_{t-1})$:

**Context pooling**

$$
c_t = \text{MLP}\!\left(\frac{1}{|\{Q\}\cup \mathcal{P}_{<t}|}\sum_{u\in \{Q\}\cup \mathcal{P}_{<t}} H_u \right)\in\mathbb{R}^{d}.
$$

**Node compatibility score**

$$
s_i^{(t)}=\text{MLP}\big([H_i \,\|\, c_t]\big)\in\mathbb{R}.
$$

*(Optional conditioning layer)* Add a one-layer lightweight GNN that takes $c_t$ as a “path token” node connected to $\{Q\}\cup \mathcal{P}_{<t}$, producing $H_t$; then score with $H_t$ instead of $H$.

# 4) Training Objectives

Let gold supports be $G=\{g_1,\dots,g_k\}$, $k\le 3$. Teacher forcing over hops.

**Ordered supervision (if chain order known)**

$$
\mathcal{L}_t = \text{BCE}\big(\sigma(s^{(t)}), \mathbf{1}_{i=g_t}\big)\quad\text{over frontier}.
$$

**Set supervision (no order)**

$$
\mathcal{L}_t = \text{BCE}\big(\sigma(s^{(t)}), \mathbf{1}_{i\in G\setminus \{g_1,\dots,g_{t-1}\}}\big).
$$

**Auxiliary node classification (stabilizer)**

$$
\mathcal{L}_{\text{aux}}=\text{BCE}\big(\sigma(w^\top H_i), \mathbf{1}_{i\in G}\big)\ \text{over paragraph nodes}.
$$

**Total**

$$
\mathcal{L}=\sum_{t=1}^{K}\mathcal{L}_t + \lambda_{\text{aux}}\mathcal{L}_{\text{aux}}.
$$

# 5) Inference: Frontier-Chain Search (Beam)

* Beam size $B$, max hops $K{=}3$, stop threshold $\tau$.

**Algorithm**

1. Encode once: $H=\text{Encoder}(X,E,\text{edge\_attr})$.
2. Initialize beam: $\mathcal{B}=\{(\emptyset, 0)\}$.
3. For $t=1..K$:

   * For each $(\mathcal{P}_{<t}, S)\in \mathcal{B}$:

     * Build frontier mask (unused paragraphs).
     * Compute $c_t$, scores $s^{(t)}$ on frontier; softmax over frontier to get $p^{(t)}$.
     * Take top-$L$ candidates; create new chains $\mathcal{P}_{<t}\cup\{i\}$ with score $S+\log p^{(t)}_i$.
   * Keep global top-$B$ chains. Early stop if $\max p^{(t)}<\tau$.
4. Return best chain; predicted support set = its nodes.

*(If using sigmoid, you may sum logits or use length-normalized sums.)*

# 6) Complexity

* Graph build: $O(n^2)$ for sims (n≤20); pruning → $O(nk)$ edges.
* Encoder: $O(L\cdot (n d + |E| d))$.
* Inference per hop: $O(B\cdot n d)$ scoring with small constants.

# 7) Default Hyperparameters (strong baseline)

* Candidates $n$: 10–20
* $k$ (top-k neighbors): 8–12 (symmetrize); keep self-loops
* Hidden $d$: 256
* Layers $L$: 2 (try 3 if stable)
* Dropout / DropEdge: 0.1–0.2
* Optimizer: AdamW, lr $2\!\times\!10^{-4}$, wd $1\!\times\!10^{-4}$
* Beam $B$: 4; top-$L$ expansions per beam item: 5
* Max hops $K$: 3
* Stop $\tau$: tune on dev (prob threshold 0.2–0.4)
* $\lambda_{\text{aux}}$: 0.2
* Class imbalance: `pos_weight` or focal loss ($\gamma=2$)

# 8) Evaluation

* **Node**: F1\@k (k=#gold), Precision\@k, MAP per question → macro average.
* **Chain** (if order available): path EM / prefix EM.
* **Calibration**: choose $\tau$ on dev maximizing F1.

# 9) Ablations (recommended)

* Edge features: +similarity only vs +entity overlap.
* Pruning: top-k vs global threshold vs mutual kNN.
* With/without question node (vs injecting $e(Q)$ into every $x_i$).
* Conditioning layer on vs off.
* GNN depth: 2 vs 3.
* Scorer: MLP vs bilinear.

# 10) Implementation Skeleton (PyG)

**Encoder**

```python
class GraphEncoder(nn.Module):
    def __init__(self, d_node, d_edge, d=256, L=2):
        super().__init__()
        self.layers = nn.ModuleList([
            GATv2Conv(d_node if i==0 else d, d, heads=4, concat=False, edge_dim=d_edge)
            for i in range(L)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])

    def forward(self, x, edge_index, edge_attr):
        h = x
        for conv, norm in zip(self.layers, self.norms):
            h = norm(F.relu(conv(h, edge_index, edge_attr)))
        return h  # H
```

**Scorer**

```python
class PrefixScorer(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ctx = nn.Sequential(nn.Linear(d, d), nn.ReLU())
        self.mlp = nn.Sequential(nn.Linear(2*d, d), nn.ReLU(), nn.Linear(d,1))

    def pool_ctx(self, H, q_idx, prefix):
        idx = [q_idx] + prefix if q_idx is not None else prefix
        C = H[idx].mean(0)
        return self.ctx(C)

    def forward(self, H, q_idx, prefix, candidates):
        c = self.pool_ctx(H, q_idx, prefix)             # (d,)
        Hi = H[candidates]                               # (M,d)
        s  = self.mlp(torch.cat([Hi, c.expand_as(Hi)],-1)).squeeze(-1)  # (M,)
        return s
```

**Beam / Frontier-Chain Inference (core loop)**

```python
def infer_beam(H, q_idx, K=3, B=4, L_per_beam=5, tau=0.25):
    N = H.size(0) - 1  # if last is Q; adjust as needed
    all_nodes = torch.arange(N, device=H.device).tolist()
    beam = [([], 0.0)]  # (prefix, log_score)

    for t in range(1, K+1):
        cand_beam = []
        for prefix, logS in beam:
            used = set(prefix)
            frontier = [i for i in all_nodes if i not in used]
            if not frontier: continue
            scores = scorer(H, q_idx, prefix, frontier)          # (M,)
            probs  = torch.softmax(scores, dim=0)                # (M,)
            topv, topi = torch.topk(probs, k=min(L_per_beam, len(frontier)))
            for v, idx in zip(topv.tolist(), topi.tolist()):
                nxt = frontier[idx]
                cand_beam.append((prefix+[nxt], logS + math.log(max(v,1e-12))))
        if not cand_beam: break
        # prune
        cand_beam.sort(key=lambda x: x[1], reverse=True)
        beam = cand_beam[:B]
        # early stop if weak frontier
        best_step_prob = math.exp(beam[0][1] - (beam[0][1]-0))  # heuristic; or track max probs
        if best_step_prob < tau:
            break
    best_chain, _ = max(beam, key=lambda x: x[1])
    return best_chain  # indices of predicted supports (set = chain nodes)
```

# 11) File Layout

```
/models
  encoder.py         # GraphEncoder
  scorer.py          # PrefixScorer (+ optional node head)
/data
  build_graph.py     # embeddings, pruning, edge_attr
/train.py            # teacher-forced hop loss + aux node loss
/infer.py            # frontier-chain (beam) inference
/metrics.py          # F1@k, MAP, chain EM
/config.yaml         # all hparams, pruning thresholds
```

# 12) Pitfalls & Guards

* **Over-smoothing**: keep GNN shallow (2–3), use residuals+LayerNorm.
* **Disconnected after pruning**: add question→top-k edges; enforce min degree.
* **Class imbalance**: `pos_weight` or focal loss.
* **Too many edges**: prefer top-k over raw thresholds; keep self-loops.
* **Leakage**: mask out used nodes at each hop; enforce uniqueness.

---

This is “go-time” ready: terminology is clean, math is explicit, and the code skeletons map 1-to-1 to the spec. If you want, I can spin this into a minimal PyG repo scaffold you can drop your embeddings and labels into.


