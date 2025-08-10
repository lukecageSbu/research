import os
import json
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class QuantumWalkRetriever(nn.Module):
    def __init__(self, embed_model='all-MiniLM-L6-v2', embedding_dim=384, k=8, hidden_dim=128, walk_steps=3):
        super().__init__()
        self.embedder = SentenceTransformer(embed_model)
        for p in self.embedder.parameters():
            p.requires_grad = False
        self.k = k
        self.walk_steps = walk_steps
        d = embedding_dim

        self.coin_net = nn.Sequential(
            nn.Linear(d * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k)
        )
        self.path_net = nn.Sequential(
            nn.Linear(d * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        nn.init.xavier_uniform_(self.path_net[2].weight)
        nn.init.zeros_(self.path_net[2].bias)

    def build_adj(self, n, neighbors):
        if neighbors is None or len(neighbors) == 0:
            idx = torch.arange(n)
            return torch.sparse_coo_tensor(torch.stack([idx, idx]), torch.ones(n), (n, n))

        rows, cols = [], []
        for i, nbrs in enumerate(neighbors):
            valid = [j for j in nbrs if 0 <= j < n]
            if not valid:
                rows.append(i); cols.append(i)
            else:
                for j in valid:
                    rows.append(i); cols.append(j)
        if not rows:
            rows = [0]; cols = [0]
        idx = torch.tensor([rows, cols])
        vals = torch.ones(len(rows))
        return torch.sparse_coo_tensor(idx, vals, (n, n))

    def forward(self, questions, sent_embs, neighbors, labels):
        if not questions:
            return []
        q_embs = self.embedder.encode(questions, convert_to_tensor=True)
        logits_list = []
        for qv, emb, nbrs, lbl in zip(q_embs, sent_embs, neighbors, labels):
            n = emb.size(0)
            if n == 0:
                continue
            A = self.build_adj(n, nbrs).to(emb.device)
            state = torch.ones(n, self.k) / (n * self.k) ** 0.5

            inp = torch.cat([emb, qv.unsqueeze(0).expand(n, -1)], dim=1)
            path_scores = torch.sigmoid(self.path_net(inp)).squeeze()
            if lbl.sum() > 0:
                supports = torch.nonzero(lbl).flatten()
                for s in supports:
                    if 0 <= s < n:
                        path_scores[s] += 1

            amps = self.coin_net(inp) * (1 + path_scores.unsqueeze(1))
            for _ in range(self.walk_steps):
                st = state * amps
                st = torch.sparse.mm(A.float(), st)
                norm = st.norm()
                state = st / norm if norm > 0 else st

            logits_list.append(state.abs().sum(dim=1))
        return logits_list


class HotpotDataset(Dataset):
    def __init__(self, data_file, emb_dir, prefix='train', k=8):
        with open(data_file) as f:
            data = json.load(f)
        # Load prefix-specific files
        emb_path = os.path.join(emb_dir, f"{prefix}_embeddings.npy")
        off_path = os.path.join(emb_dir, f"{prefix}_offsets.npy")
        idx_path = os.path.join(emb_dir, f"{prefix}_index.faiss")
        self.sent_embs = np.load(emb_path)
        self.offsets = np.load(off_path)
        self.index = faiss.read_index(idx_path)
        self.k = k
        self.examples = []
        for item, (start, end) in zip(data, self.offsets):
            embs = self.sent_embs[start:end]
            if embs.size == 0:
                continue
            neighbors = self._get_neighbors(embs)
            labels = self._get_labels(item, len(embs))
            self.examples.append((item['question'], torch.tensor(embs), neighbors, torch.tensor(labels, dtype=torch.float)))

    def _get_neighbors(self, embs):
        _, nbrs = self.index.search(embs.astype(np.float32), self.k + 1)
        nbrs = nbrs[:, 1:]
        return [list(row) for row in nbrs]

    def _get_labels(self, item, n):
        labels = []
        for title, sents in item['context']:
            for i in range(len(sents)):
                labels.append(1 if [title, i] in item.get('supporting_facts', []) else 0)
        # pad/truncate
        if len(labels) < n:
            labels += [0] * (n - len(labels))
        return labels[:n]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for q, emb, nbrs, lbl in loader:
            logits = model([q], [emb.to(device)], [nbrs], [lbl.to(device)])[0]
            if lbl.sum() == 0:
                continue
            prob = F.softmax(logits, dim=0)
            loss = F.kl_div(prob.log(), lbl / lbl.sum(), reduction='batchmean')
            total_loss += loss.item()
            k = int(lbl.sum().item())
            if set(prob.topk(k).indices.tolist()) == set((lbl>0).nonzero().flatten().tolist()):
                correct += 1
            count += 1
    return (total_loss / count) if count else float('inf'), (correct / count) if count else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--dev',   required=True)
    parser.add_argument('--emb',   required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--bs',     type=int, default=64)
    parser.add_argument('--lr',     type=float, default=1e-4)
    parser.add_argument('--k',      type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # pass prefix to load correct npy/faiss files
    train_ds = HotpotDataset(args.train, args.emb, prefix='train', k=args.k)
    dev_ds   = HotpotDataset(args.dev,   args.emb, prefix='dev',   k=args.k)

    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=args.bs)

    model = QuantumWalkRetriever(k=args.k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = float('inf')
    for ep in range(args.epochs):
        model.train()
        for q, emb, nbrs, lbl in train_loader:
            opt.zero_grad()
            logits = model([q], [emb.to(device)], [nbrs], [lbl.to(device)])[0]
            if lbl.sum() == 0:
                continue
            loss = F.kl_div(F.softmax(logits, dim=0).log(), lbl / lbl.sum(), reduction='batchmean')
            loss.backward()
            opt.step()
        val_loss, em = evaluate(model, dev_loader, device)
        print(f"Epoch {ep}: Val Loss={val_loss:.4f}, EM={em:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pt')

if __name__ == '__main__':
    main()
