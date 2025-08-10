# Cell 2: imports + retriever
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import List, Dict, Tuple

import faiss
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

class QuantumWalkRetriever(nn.Module):
    def __init__(self, embed_model_name='all-MiniLM-L6-v2', k=5, hidden_dim=128, walk_steps=3):
        super().__init__()
        self.embedder = SentenceTransformer(embed_model_name)
        for p in self.embedder.parameters(): p.requires_grad = False
        self.k, self.walk_steps = k, walk_steps
        d = self.embedder.get_sentence_embedding_dimension()
        self.coin_net = nn.Sequential(
            nn.Linear(d*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k)
        )

    def embed_sentences(self, sents: List[str]) -> np.ndarray:
        return self.embedder.encode(sents, convert_to_numpy=True)

    def build_graph(self, emb: np.ndarray) -> nx.Graph:
        sim = cosine_similarity(emb)
        n = len(emb)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            nbrs = np.argsort(sim[i])[::-1][1:self.k+1]
            for j in nbrs:
                G.add_edge(i, j, weight=sim[i,j])
        return G

    def quantum_walk(self, G: nx.Graph, qv: np.ndarray, emb: np.ndarray) -> torch.Tensor:
        n, k = G.number_of_nodes(), self.k
        # Determine the device from the model parameters
        device = next(self.coin_net.parameters()).device
        
        # Initialize state on the correct device
        state = torch.ones(n,k, dtype=torch.cfloat, device=device) / np.sqrt(n*k)
        nbr_lists = [list(G.neighbors(i)) for i in range(n)]
        
        # Move input tensors to the correct device
        q_t = torch.from_numpy(qv).float().to(device)
        emb_t = torch.from_numpy(emb).float().to(device)
        
        for _ in range(self.walk_steps):
            coins = []
            for i in range(n):
                # inp is now created from tensors already on the correct device
                inp = torch.cat([emb_t[i], q_t]) 
                amps = self.coin_net(inp)
                # Ensure normalization factor isn't zero or NaN
                norm_factor = torch.norm(amps)
                if norm_factor == 0 or torch.isnan(norm_factor):
                    # Handle case where norm is zero or NaN - maybe assign uniform coin?
                    # Using a small epsilon or uniform distribution might be options.
                    # For now, creating a uniform complex coin as a fallback.
                    uniform_amps = torch.ones_like(amps) 
                    norm_factor = torch.norm(uniform_amps)
                    coin_complex = (uniform_amps.unsqueeze(1) * uniform_amps.unsqueeze(0)).to(torch.cfloat)
                    coin_complex /= torch.norm(coin_complex) # Normalize the complex coin matrix
                else:
                    c_real = amps.unsqueeze(1) * amps.unsqueeze(0)
                    # Normalize the real coin matrix before converting to complex
                    c_real_norm = torch.norm(c_real)
                    if c_real_norm > 0:
                         coin_complex = (c_real / c_real_norm).to(torch.cfloat) 
                    else: # Handle zero norm case again if needed
                         uniform_amps = torch.ones_like(amps) 
                         norm_factor = torch.norm(uniform_amps)
                         coin_complex = (uniform_amps.unsqueeze(1) * uniform_amps.unsqueeze(0)).to(torch.cfloat)
                         coin_complex /= torch.norm(coin_complex)
                coins.append(coin_complex)
            
            # Initialize new_state on the correct device
            new_state = torch.zeros_like(state)
            for i in range(n):
                # Ensure coin and state slice are compatible for matmul
                # coin[i] is (k, k), state[i] is (k,) -> result should be (k,)
                s_p = coins[i] @ state[i] 
                neighbors = nbr_lists[i][:k] # Get up to k neighbors
                for idx,j in enumerate(neighbors):
                    # Check if idx is within the bounds of s_p
                    if idx < len(s_p):
                        new_state[j,idx] += s_p[idx]
                    # else: Handle cases where node has fewer than k neighbors if necessary
            
            # Normalize the global state
            state_norm = torch.norm(new_state)
            if state_norm > 0:
                 state = new_state / state_norm
            else:
                 # Handle global state becoming zero if necessary (e.g., reinitialize)
                 print("Warning: Quantum walk state norm is zero.")
                 # Reinitialize or break?
                 state = torch.ones(n,k, dtype=torch.cfloat, device=device) / np.sqrt(n*k)

        # Final result computation
        return state.abs().sum(dim=1)

    def forward(self, question: str, sents: List[str]) -> List[Tuple[int,float]]:
        emb = self.embed_sentences(sents)
        qv  = self.embedder.encode([question], convert_to_numpy=True)[0]
        G   = self.build_graph(emb)
        logits = self.quantum_walk(G, qv, emb).detach().cpu().numpy()
        return sorted(enumerate(logits), key=lambda x: x[1], reverse=True)


# Cell 3: IR + training/test helpers
def load_hotpot(fn: str) -> List[Dict]:
    return json.load(open(fn))

def build_paragraph_index(data, model):
    paras = []
    for ex in data:
        for title, s in ex['context']:
            paras.append((title, s))
    seen, unique = set(), []
    for t,s in paras:
        if t not in seen:
            seen.add(t); unique.append((t,s))
    texts = [' '.join(s) for _,s in unique]
    embs  = model.encode(texts, convert_to_numpy=True)
    d = embs.shape[1]
    idx = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embs)
    idx.add(embs)
    return idx, unique

def retrieve_topk(idx, paras, model, qs, k=10):
    ids, questions = zip(*qs)
    qemb = model.encode(questions, convert_to_numpy=True)
    faiss.normalize_L2(qemb)
    _, I = idx.search(qemb, k)
    return {qid: [paras[i] for i in row] for qid,row in zip(ids, I)}

def prepare_train_examples(data):
    exs = []
    for ex in data:
        q = ex['question']
        sents, lbls = [], []
        for title, slist in ex['context']:
            for sid,s in enumerate(slist):
                sents.append(s)
                lbls.append(1 if [title,sid] in ex.get('supporting_facts',[]) else 0)
        exs.append({'question':q,'sentences':sents,'labels':lbls})
    return exs


# Cell 4: TRAIN with checkpointing (resume/save)

# Define constants and setup outside the main block if they might be needed by imports,
# although it's generally cleaner to pass parameters or use a config file.
train_file   = '/data/research/hotpot/hotpot_train_v1.1.json'
model_path   = 'coin_net.pth'
epochs, lr   = 5, 1e-3
checkpoint_dir = 'checkpoints'
last_ckpt = os.path.join(checkpoint_dir, 'last.ckpt')

# Only run training when the script is executed directly
if __name__ == "__main__":
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load data and initialize
    print("Loading training data...")
    train_data  = load_hotpot(train_file)
    examples    = prepare_train_examples(train_data)
    print("Initializing model...")
    retriever   = QuantumWalkRetriever() # Using default params k=5, hidden_dim=128, walk_steps=3
    optimizer   = torch.optim.Adam(retriever.coin_net.parameters(), lr=lr)

    # Resume from checkpoint if exists
    start_epoch = 0
    if os.path.exists(last_ckpt):
        print(f"Loading checkpoint from {last_ckpt}...")
        ckpt = torch.load(last_ckpt)
        retriever.coin_net.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming training from epoch {start_epoch}")
    else:
        print("No checkpoint found, starting training from scratch.")

    # Training loop
    print("Starting training...")
    for ep in range(start_epoch, epochs):
        total_loss = 0.0
        for ex in tqdm(examples, desc=f'Train Epoch {ep+1}/{epochs}'):
            labels = torch.tensor(ex['labels'], dtype=torch.float)
            if labels.sum() > 0:
                labels /= labels.sum()
            else:
                # Handle cases with no positive labels if necessary, e.g., skip or assign a default behavior
                # print(f"Warning: Skipping example with no positive labels in epoch {ep+1}")
                continue # Skipping for now
            
            emb  = retriever.embed_sentences(ex['sentences'])
            qv   = retriever.embedder.encode([ex['question']], convert_to_numpy=True)[0]
            G    = retriever.build_graph(emb)
            logits = retriever.quantum_walk(G, qv, emb)
            probs  = torch.softmax(logits, dim=0)
            loss   = F.kl_div(probs.log(), labels, reduction='batchmean')
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf loss encountered in epoch {ep+1}. Skipping update.")
                # Optionally add more debugging info here, like printing inputs/outputs
                continue

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        # Avoid division by zero if all examples were skipped
        num_processed = len([ex for ex in examples if torch.tensor(ex['labels']).sum() > 0]) 
        if num_processed > 0:
            avg_loss = total_loss / num_processed
            print(f'Epoch {ep+1}, Avg Loss: {avg_loss:.4f}')
        else:
            print(f'Epoch {ep+1}, No valid examples processed.')

        # Save checkpoint
        print(f"Saving checkpoint for epoch {ep+1}...")
        ckpt = {
            'epoch': ep,
            'model_state': retriever.coin_net.state_dict(),
            'optimizer_state': optimizer.state_dict()
        }
        torch.save(ckpt, last_ckpt)
        torch.save(retriever.coin_net.state_dict(), model_path)

    print("Training complete, final model saved.")