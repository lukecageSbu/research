Alright — let’s lay out a **detailed, end-to-end design plan** for your setup so we can handle:

* **Multiple graphs** (one per question)
* **Fully connected initially → pruned by similarity + reasoning cues**
* **Bridge/comparison reasoning** via edge-aware GNN
* **Targets** = supporting fact paragraphs (nodes)
* **Optional** = label bridge/comparison edges

---

## **1. Problem Restatement**

Given:

* A question $Q$
* A set of paragraphs $P_1, P_2, \dots, P_n$ for that question
* Labels: **supporting facts** (subset of paragraphs)
  Goal: **Train one GNN** to predict supporting facts, learning both paragraph relevance and the relationships (bridges/comparisons) between them.

---

## **2. Graph Construction (Per Question)**

### **Nodes**

* $N$ paragraph nodes: each node feature = \[CLS] embedding from a language model (e.g., Sentence-BERT, MPNet).
* 1 **question node**: embedding from the question text.
* Total nodes per graph = $N + 1$.

### **Node Feature Vector**

* Concatenate multiple signals:

  ```
  [Paragraph Embedding; Paragraph–Question Similarity Score; Entity Overlap with Question]
  ```

  (normalize all numeric features before concat)

---

### **Edges**

Initially fully connected, then prune:

#### **Step 1: Compute similarities**

* Cosine similarity between paragraph embeddings → sim(Pi, Pj)
* Entity overlap score = # of named entities shared between Pi and Pj

#### **Step 2: Keep informative edges**

* Keep edges where:

  ```
  sim ≥ τ_sim OR entity_overlap ≥ τ_ent
  ```
* Always connect **question node → top-k paragraphs** by question–paragraph similarity.

#### **Step 3: Edge features**

Each edge gets:

```
[cosine similarity, entity overlap, question relevance indicator]
```

Shape: (E, d\_edge) — e.g., 3 values per edge.

---

## **3. Dataset Format**

Using **PyTorch Geometric** `Data` object:

```
Data(
  x = node_features,           # [num_nodes, d_node]
  edge_index = edge_pairs,     # [2, num_edges]
  edge_attr = edge_features,   # [num_edges, d_edge]
  y = node_labels,             # [num_nodes] (0/1, question node ignored)
  is_para_mask = mask          # [num_nodes] (True if paragraph node)
)
```

Multiple questions → list of `Data` objects → batch with `DataLoader`.

---

## **4. Model Architecture**

### **Layer type**

We need **edge-aware message passing** for bridge/comparison reasoning:

* **GATv2Conv** (PyG) with `edge_dim` for edge features
* Or **Graph Transformer** layer (attention over edges)

### **Core idea**

* Message passing lets paragraphs exchange info based on edges; multi-hop layers allow bridging.
* Edge features bias the attention so that edges with high entity overlap or similarity get more weight.

---

### **Model Plan**

```python
class ReasoningGNN(nn.Module):
    def __init__(self, d_node, d_edge, d_hidden):
        super().__init__()
        self.g1 = GATv2Conv(d_node, d_hidden, heads=4, concat=False, edge_dim=d_edge)
        self.g2 = GATv2Conv(d_hidden, d_hidden, heads=4, concat=False, edge_dim=d_edge)
        self.norm1 = nn.LayerNorm(d_hidden)
        self.norm2 = nn.LayerNorm(d_hidden)
        self.node_head = nn.Linear(d_hidden, 1)   # Node classification head
        self.edge_head = nn.Linear(d_hidden*2, 1) # Optional edge classification

    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.g1(x, edge_index, edge_attr))
        h = self.norm1(h)
        h = F.relu(self.g2(h, edge_index, edge_attr))
        h = self.norm2(h)

        node_logits = self.node_head(h).squeeze(-1)

        # Edge logits: concat embeddings of both ends
        src, dst = edge_index
        edge_h = torch.cat([h[src], h[dst]], dim=-1)
        edge_logits = self.edge_head(edge_h).squeeze(-1)

        return node_logits, edge_logits
```

---

## **5. Loss Functions**

We can do **multi-task learning**:

1. **Node loss** (main task — supporting facts):

```python
loss_nodes = BCEWithLogitsLoss(pos_weight=pos_wt)(
    node_logits[is_para_mask],
    y[is_para_mask].float()
)
```

2. **Edge loss** (optional — bridge/comparison edges if labeled):

```python
loss_edges = BCEWithLogitsLoss()(
    edge_logits, y_edges.float()
)
```

3. **Total loss**:

```python
loss = loss_nodes + λ * loss_edges
```

λ = weight for edge supervision (start with 0.5–1.0).

---

## **6. Training Setup**

* **Optimizer**: AdamW(lr=2e-4, weight\_decay=1e-4)
* **Batching**: PyG DataLoader with `batch_size=16` graphs
* **Epochs**: 20–40 (early stopping on dev F1)
* **Eval metric**: F1\@k (multi-label) on paragraph selection

---

## **7. Inference**

Given a new question:

1. Build its graph (same preprocessing).
2. Run through GNN → get per-node sigmoid scores.
3. Select top-k paragraphs or score ≥ τ.
4. (Optional) Use predicted high-weight edges to explain *why* the paragraphs are linked.

---

## **8. Extensions for Better Reasoning**

* **Question as global context**: Instead of a single question node, inject question embedding into every node’s initial feature vector.
* **Edge dropout (DropEdge)** during training → improve generalization.
* **Top-k attention edges at inference** → interpret reasoning chains.
* **Contrastive loss**: encourage supporting fact nodes to be closer in embedding space.

---

✅ **Outcome:**
A single GNN model that:

* Takes one question-graph at a time.
* Uses **node features** (paragraph embeddings + question relevance) and **edge features** (similarity + entity overlap).
* Learns to identify supporting fact nodes.
* Optionally learns to classify edges as reasoning links (bridge/comparison).

---

If you want, I can **diagram this architecture** so it visually maps the question node, paragraph nodes, and bridge edges in your Hotpot-like reasoning case. That would make it easier to explain to others on your team.


Alright — here’s an **updated and consolidated design plan** for your case, merging the **Beam Retrieval multi-hop idea** with a **graph neural network encoder** that can handle up to **3 supporting paragraphs** (multi-hop reasoning, bridge/comparison).

---

## **1. Problem Restatement**

* Input: A question $Q$ + $n$ candidate paragraphs $\{P_1, P_2, \dots, P_n\}$.
* Output: Predict the set of **supporting paragraphs** (up to 3 per question) that collectively enable answering $Q$.
* Some require **1-hop**, others **2-hop**, some **3-hop** reasoning (bridge/comparison).
* Need to handle **order-aware** or **order-unaware** supervision.
* Process: Use a **single GNN model** to score supporting facts step-by-step (beam search at inference), **like Beam Retrieval**, but with graph-aware context.

---

## **2. Graph Construction (per question)**

### **Nodes**

* **Paragraph nodes**: LM embeddings (e.g., Sentence-BERT, MPNet) for each paragraph.
* **Question node**: LM embedding for $Q$.

  * Ensures conditioning on $Q$ during message passing.

### **Node features**

For paragraph nodes:

$$
x_i = [\text{LM}(P_i) \;\Vert\; \cos(Q, P_i) \;\Vert\; \text{entity\_overlap}(Q, P_i)]
$$

For the question node:

$$
x_Q = [\text{LM}(Q) \;\Vert\; 1 \;\Vert\; 1]
$$

---

### **Edges**

1. Start **fully connected** between paragraph nodes.
2. **Prune**:

   * Keep top-$k$ neighbors per node by cosine similarity.
   * OR keep edges where sim ≥ $\tau_{sim}$ or entity\_overlap ≥ $\tau_{ent}$.
3. Always connect **question node → top-k relevant paragraphs** by Q–P similarity.
4. Keep **self-loops**.

---

### **Edge features**

$$
e_{ij} = [\cos(P_i, P_j),\; \text{entity\_overlap}(P_i, P_j),\; \cos(Q, P_i),\; \cos(Q, P_j)]
$$

---

## **3. Model Architecture**

### **Base encoder**

* **Edge-aware GNN** (Graph Transformer or GATv2Conv with `edge_dim`).
* 2–3 layers with residuals + LayerNorm to avoid over-smoothing.
* Shared across hops.

### **Prefix conditioning (multi-hop)**

At hop $t$, we know the **partial path** $\hat{P}_{<t}$:

* Pool embeddings of $Q$ and selected paragraphs:
  $c_t = \text{Pool}(H_Q, H_{p_1}, ..., H_{p_{t-1}})$.
* Either:

  * **Light**: Pass $c_t$ only to the scoring head.
  * **Heavy**: Add a **path token node** with feature $c_t$, connect to selected nodes, run one extra GNN layer to propagate context.

---

### **Scoring head**

For each candidate paragraph node $i$ not in the prefix:

* $s_i^{(t)} = \text{MLP}([H_i \,\Vert\, c_t])$
* Produces logits for “next relevant passage” at hop $t$.

---

## **4. Training Objective**

### **Autoregressive hop-wise loss**

If ordered supervision (gold chain $p_1, p_2, p_3$ known):

$$
\mathcal{L}_t = -\sum_{i} \big[ \mathbf{1}_{i=p_t} \log\sigma(s_i^{(t)}) + (1-\mathbf{1}_{i=p_t}) \log (1 - \sigma(s_i^{(t)})) \big]
$$

If only set supervision:

* Positives = remaining golds not yet selected → **multi-label BCE** per hop.

---

### **Auxiliary node classification**

Extra BCE head on $H$ to predict support facts ignoring hops. Weight small (e.g., 0.2).
Helps stabilize early layers.

---

**Total loss**:

$$
\mathcal{L} = \sum_{t=1}^K \mathcal{L}_t + \lambda_{\text{aux}} \cdot \mathcal{L}_{\text{aux}}
$$

---

## **5. Inference (Beam Search)**

1. **Hop 1**: score all paragraphs given $Q$, keep top $B$ singletons.
2. **Hop t > 1**:

   * For each partial path in beam:

     * Score all unused paragraphs conditioned on that prefix.
   * Keep top $B$ paths.
3. **Stopping**:

   * Stop if best score < $\tau$ or $t = 3$.
4. Output: union of nodes in best path.

This mirrors Beam Retrieval’s decoding, but with **graph-aware encoding** at each step.

---

## **6. Implementation Notes**

* **Efficiency**:

  * Cache base GNN encoding $H$ once per question; only recompute conditioning for each hop.
  * Use top-k adjacency to keep edge count small ($O(nk)$).
* **Prevent duplicates**:

  * Mask out already selected nodes before scoring.
* **Batching**:

  * PyG `DataLoader` can batch multiple question-graphs (disconnected).

---

## **7. Hyperparameters to start**

* **n** = 10–20 candidate paragraphs
* **k\_neighbors** = 8 (graph pruning)
* **GNN hidden size** = 256
* **Layers** = 2 base + 1 optional conditioning
* **Beam size** $B$ = 4
* **Max hops** $K$ = 3
* **LR** = 2e-4, AdamW, wd=1e-4
* **Dropout** = 0.2

---

## **8. Advantages of this design**

* Keeps **Beam Retrieval’s multi-hop scoring** but in **graph context** → captures bridge/comparison reasoning.
* Edge features allow explicit modeling of relationships.
* Prefix conditioning makes scoring step-aware without rebuilding the whole graph each time.
* Beam search reduces early-hop errors.

---

If you want, I can now **draw this architecture** as a diagram — showing the question node, paragraph nodes, edge pruning, GNN encoding, prefix conditioning, and beam search loop — so you have both the written plan and a visual for your paper or team. Would you like me to do that?
