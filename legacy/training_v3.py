import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Any
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import glob
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR  # Added for LR scheduling

class QuantumWalkRetriever(nn.Module):
    def __init__(self, embed_model_name='all-MiniLM-L6-v2', k=8, hidden_dim=128, walk_steps=3):
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
        # Add path scoring network
        self.path_net = nn.Sequential(
            nn.Linear(d*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Initialize path net weights
        for m in self.path_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

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

    def find_paths_to(self, G: nx.Graph, target: int, max_length: int = 3) -> List[List[int]]:
        """Find all paths of length <= max_length to the target node."""
        paths = []
        for source in range(G.number_of_nodes()):
            try:
                for path in nx.all_simple_paths(G, source=source, target=target, cutoff=max_length):
                    paths.append(path)
            except nx.NetworkXNoPath:
                continue
        return paths

    def score_path(self, path: List[int], emb: np.ndarray, qv: np.ndarray) -> torch.Tensor:
        """Score a path based on semantic coherence and question relevance."""
        if len(path) < 2:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Convert to tensors for gradient computation
        emb_t = torch.from_numpy(emb).float().to(next(self.parameters()).device)
        qv_t = torch.from_numpy(qv).float().to(next(self.parameters()).device)
        
        # Score based on edge weights (semantic similarity)
        edge_scores = []
        for i in range(len(path)-1):
            sim = F.cosine_similarity(emb_t[path[i]].unsqueeze(0), emb_t[path[i+1]].unsqueeze(0))
            edge_scores.append(sim)
        
        # Score based on question relevance
        question_scores = []
        for node in path:
            sim = F.cosine_similarity(emb_t[node].unsqueeze(0), qv_t.unsqueeze(0))
            question_scores.append(sim)
        
        # Combine scores
        edge_score = torch.mean(torch.stack(edge_scores))
        question_score = torch.mean(torch.stack(question_scores))
        path_score = edge_score * question_score
        
        return path_score

    def quantum_walk(self, G: nx.Graph, qv: np.ndarray, emb: np.ndarray, labels: torch.Tensor = None) -> torch.Tensor:
        n, k = G.number_of_nodes(), self.k
        device = next(self.parameters()).device
        state = torch.ones(n,k, dtype=torch.cfloat, device=device, requires_grad=True) / np.sqrt(n*k)
        nbr_lists = [list(G.neighbors(i)) for i in range(n)]
        q_t = torch.from_numpy(qv).float().to(device)
        emb_t = torch.from_numpy(emb).float().to(device)
        
        # Initialize path scores as a new tensor with gradients
        path_scores = torch.zeros(n, device=device, requires_grad=True)
        
        # If labels are provided, use them to find support facts
        if labels is not None:
            support_indices = labels.nonzero().squeeze().tolist()
            if isinstance(support_indices, int):
                support_indices = [support_indices]
            
            # Score paths to support facts
            for target in support_indices:
                paths = self.find_paths_to(G, target, max_length=self.walk_steps)
                for path in paths:
                    path_score = self.score_path(path, emb, qv)
                    # Create a new tensor for updated path scores
                    new_path_scores = path_scores.clone()
                    for node in path:
                        new_path_scores[node] = new_path_scores[node] + path_score
                    path_scores = new_path_scores
        
        for _ in range(self.walk_steps):
            coins = []
            for i in range(n):
                inp = torch.cat([emb_t[i], q_t])
                amps = self.coin_net(inp)
                
                # Incorporate path scores into coin operator
                if path_scores[i] > 0:
                    path_factor = torch.sigmoid(self.path_net(inp)) * path_scores[i]
                    amps = amps * (1 + path_factor)
                
                norm_factor = torch.norm(amps)
                if norm_factor == 0 or torch.isnan(norm_factor):
                    uniform_amps = torch.ones_like(amps)
                    norm_factor = torch.norm(uniform_amps)
                    coin_complex = (uniform_amps.unsqueeze(1) * uniform_amps.unsqueeze(0)).to(torch.cfloat)
                    coin_complex /= torch.norm(coin_complex)
                else:
                    c_real = amps.unsqueeze(1) * amps.unsqueeze(0)
                    c_real_norm = torch.norm(c_real)
                    if c_real_norm > 0:
                        coin_complex = (c_real / c_real_norm).to(torch.cfloat)
                    else:
                        uniform_amps = torch.ones_like(amps)
                        norm_factor = torch.norm(uniform_amps)
                        coin_complex = (uniform_amps.unsqueeze(1) * uniform_amps.unsqueeze(0)).to(torch.cfloat)
                        coin_complex /= torch.norm(coin_complex)
                coins.append(coin_complex)
            
            new_state = torch.zeros_like(state)
            for i in range(n):
                s_p = coins[i] @ state[i]
                neighbors = nbr_lists[i][:k]
                for idx,j in enumerate(neighbors):
                    if idx < len(s_p):
                        new_state[j,idx] += s_p[idx]
            
            state_norm = torch.norm(new_state)
            if state_norm > 0:
                state = new_state / state_norm
            else:
                state = torch.ones(n,k, dtype=torch.cfloat, device=device, requires_grad=True) / np.sqrt(n*k)

        return state.abs().sum(dim=1)

    def forward(self, batch, device='cuda'):
        """Process a batch of examples."""
        batch_logits = []
        for example in batch:
            questions = example['question']
            sentences = example['sentences']
            labels = torch.tensor(example['labels'], device=device).float()
            
            # Skip examples with no positive labels
            if labels.sum() == 0:
                continue
                
            # Normalize labels
            labels = labels / labels.sum()
            
            # Get embeddings and run retrieval
            emb = self.embedder.encode(sentences, convert_to_numpy=True)
            qv = self.embedder.encode([questions], convert_to_numpy=True)[0]
            G = self.build_graph(emb)
            
            # Run quantum walk with path awareness
            logits = self.quantum_walk(G, qv, emb, labels)
            batch_logits.append(logits)
        
        return batch_logits

class HotpotDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        self.examples = self._prepare_examples()

    def _prepare_examples(self):
        exs = []
        for ex in self.data:
            q = ex['question']
            sents, lbls = [], []
            for title, slist in ex['context']:
                for sid,s in enumerate(slist):
                    sents.append(s)
                    lbls.append(1 if [title,sid] in ex.get('supporting_facts',[]) else 0)
            exs.append({'question':q,'sentences':sents,'labels':lbls})
        return exs

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

# Add this function to handle evaluation during training
def evaluate_model(model, dataloader, device, num_batches=None):
    """Evaluates the model on the given dataloader and returns average loss and metrics."""
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_exact_match = 0
    total_examples = 0
    
    if num_batches is None:
        total_samples = len(dataloader)
    else:
        total_samples = min(num_batches, len(dataloader))
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, desc='Evaluating', total=total_samples)):
            if num_batches is not None and i >= num_batches:
                break
            
            for example in batch:
                questions = example['question']
                sentences = example['sentences']
                labels = torch.tensor(example['labels'], device=device).float()
                
                if labels.sum() == 0:
                    continue
                    
                labels = labels / labels.sum()
                
                emb = model.module.embedder.encode(sentences, convert_to_numpy=True)
                qv = model.module.embedder.encode([questions], convert_to_numpy=True)[0]
                G = model.module.build_graph(emb)
                
                logits = model.module.quantum_walk(G, qv, emb, labels)
                probs = torch.softmax(logits.float(), dim=0)
                loss = F.kl_div(probs.log(), labels, reduction='batchmean')
                
                # Calculate exact match
                k = int(labels.sum().item())
                pred_indices = set(probs.topk(k)[1].tolist())
                true_indices = set(labels.nonzero().squeeze().tolist())
                if pred_indices == true_indices:
                    total_exact_match += 1
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    total_loss += loss.item()
                    total_batches += 1
                    total_examples += 1
    
    avg_loss = total_loss / total_batches if total_batches > 0 else float('inf')
    exact_match = total_exact_match / total_examples if total_examples > 0 else 0.0
    
    return avg_loss, exact_match

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized inputs.
    Instead of stacking examples, it returns a list of individual examples.
    """
    return batch  # Just return the batch as a list of dictionaries without any stacking

def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint with proper handling of model architecture changes."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get current model state dict
        current_state_dict = model.state_dict()
        
        # Get checkpoint state dict
        checkpoint_state_dict = checkpoint['model_state_dict']
        
        # Handle missing keys (new architecture)
        for key in current_state_dict.keys():
            if key not in checkpoint_state_dict:
                print(f"Initializing new parameter: {key}")
                if 'path_net' in key:
                    # Initialize path_net weights
                    if 'weight' in key:
                        nn.init.xavier_uniform_(current_state_dict[key])
                    elif 'bias' in key:
                        nn.init.zeros_(current_state_dict[key])
        
        # Handle size mismatches
        for key in checkpoint_state_dict.keys():
            if key in current_state_dict:
                if checkpoint_state_dict[key].shape != current_state_dict[key].shape:
                    print(f"Resizing parameter: {key}")
                    if 'coin_net.2' in key:  # Handle k parameter change
                        if 'weight' in key:
                            # Resize weight matrix
                            new_weight = torch.zeros_like(current_state_dict[key])
                            min_k = min(checkpoint_state_dict[key].shape[0], current_state_dict[key].shape[0])
                            new_weight[:min_k] = checkpoint_state_dict[key][:min_k]
                            checkpoint_state_dict[key] = new_weight
                        elif 'bias' in key:
                            # Resize bias vector
                            new_bias = torch.zeros_like(current_state_dict[key])
                            min_k = min(checkpoint_state_dict[key].shape[0], current_state_dict[key].shape[0])
                            new_bias[:min_k] = checkpoint_state_dict[key][:min_k]
                            checkpoint_state_dict[key] = new_bias
        
        # Load state dict
        model.load_state_dict(checkpoint_state_dict, strict=False)
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load epoch
        start_epoch = checkpoint.get('epoch', 0) + 1
        
        print(f"Successfully loaded checkpoint from epoch {start_epoch-1}")
        return start_epoch
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True)
    parser.add_argument('--dev_file', type=str, required=True)
    parser.add_argument('--checkpoint_dir', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--eval_batch_size', type=int, default=4096)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--walk_steps', type=int, default=2)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=32)
    parser.add_argument('--lr', type=float, default=4e-4)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--gradient_clip_val', type=float, default=1.0)  # Added for gradient clipping
    parser.add_argument('--warmup_epochs', type=int, default=5)  # Added for LR warmup
    parser.add_argument('--eval_interval', type=int, default=1, 
                      help='Evaluate every N epochs')
    parser.add_argument('--eval_batches', type=int, default=50,
                      help='Number of batches to use for evaluation (None for all)')
    parser.add_argument('--start_from_scratch', action='store_true',
                      help='Start training from scratch, ignoring any checkpoints')
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    
    # Set device for random number generator
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    generator = torch.Generator(device=device)

    # Create datasets
    train_dataset = HotpotDataset(args.train_file)
    dev_dataset = HotpotDataset(args.dev_file)

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
    )
    dev_sampler = DistributedSampler(
        dev_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=42,
    )

    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn  # Add custom collate function
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        sampler=dev_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn  # Add custom collate function
    )

    # Initialize model and move to device
    model = QuantumWalkRetriever(
        k=args.k,
        hidden_dim=args.hidden_dim,
        walk_steps=args.walk_steps
    ).to(device)
        
    model = DDP(model, device_ids=[local_rank])
    
    # Ensure parameters that should be trained are set to requires_grad=True
    for param in model.module.coin_net.parameters():
        param.requires_grad = True
    for param in model.module.path_net.parameters():
        param.requires_grad = True
    
    # Make sure we're optimizing ALL trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if rank == 0:
        print(f"Number of trainable parameters: {sum(p.numel() for p in trainable_params)}")
    
    optimizer = optim.Adam([
        {'params': model.module.coin_net.parameters(), 'lr': args.lr},
        {'params': model.module.path_net.parameters(), 'lr': args.lr}
    ])
    
    # Add learning rate scheduler with warmup
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs-args.warmup_epochs)

    # Initialize GradScaler for AMP using the recommended API
    scaler = GradScaler(enabled=True)

    # Checkpoint loading logic
    start_epoch = 0
    if not args.start_from_scratch and os.path.exists(args.checkpoint_dir):
        checkpoint_files = glob.glob(os.path.join(args.checkpoint_dir, 'checkpoint_epoch_*.pt'))
        if checkpoint_files:
            try:
                latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
                print(f"Loading checkpoint: {latest_checkpoint}")
                start_epoch = load_checkpoint(model, optimizer, latest_checkpoint, device)
            except Exception as e:
                print(f"Error loading checkpoint: {e}. Starting from scratch.")
                start_epoch = 0
        else:
            print("No checkpoint files found. Starting from scratch.")
    else:
        print("Starting from scratch (--start_from_scratch flag set or no checkpoint directory)")

    # Learning rate warmup values
    warmup_lr_values = [args.lr * ((i+1) / args.warmup_epochs) for i in range(args.warmup_epochs)]

    # Training loop
    best_eval_loss = float('inf')
    for epoch in range(start_epoch, args.num_epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        
        total_train_loss = 0.0
        train_batches = 0
        
        for batch in tqdm(train_dataloader, desc=f'Epoch {epoch}'):
            # Process all examples in batch together to accumulate gradients
            optimizer.zero_grad()
            batch_loss = 0
            has_valid_examples = False
            
            for ex in batch:
                labels = torch.tensor(ex['labels'], device=device).float()
                if labels.sum() == 0:
                    continue
                
                labels = labels / labels.sum()
                
                # Forward pass with autocast
                with autocast(device_type='cuda'):
                    emb = model.module.embedder.encode(ex['sentences'], convert_to_numpy=True)
                    qv = model.module.embedder.encode([ex['question']], convert_to_numpy=True)[0]
                    G = model.module.build_graph(emb)
                    logits = model.module.quantum_walk(G, qv, emb, labels)
                    probs = torch.softmax(logits.float(), dim=0)
                    loss = F.kl_div(probs.log(), labels, reduction='batchmean')
                
                # Skip problematic examples
                if not torch.isfinite(loss):
                    continue
                
                # Accumulate the loss (normalized by number of examples)
                batch_loss += loss
                has_valid_examples = True
            
            # If we have at least one valid example, backprop the accumulated loss
            if has_valid_examples:
                # Scale and backprop
                scaler.scale(batch_loss).backward()
                
                # Unscale before gradient clipping
                scaler.unscale_(optimizer)
                
                # Apply gradient clipping 
                nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip_val)
                
                # Optimizer step with scaler
                scaler.step(optimizer)
                
                # Update scaler for next iteration
                scaler.update()
                
                # Track loss for reporting
                total_train_loss += batch_loss.item()
                train_batches += 1
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / train_batches if train_batches > 0 else float('inf')
        if rank == 0:
            print(f"Epoch {epoch} - Training Loss: {avg_train_loss:.6f}")
        
        # LR scheduling
        if epoch < args.warmup_epochs:
            # Manual warmup
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr_values[epoch]
        else:
            # Use cosine scheduler after warmup
            lr_scheduler.step()
            
        # Evaluation step
        if args.eval_interval > 0 and epoch % args.eval_interval == 0:
            # Evaluate on dev set
            dev_loss, dev_exact_match = evaluate_model(
                model, 
                dev_dataloader, 
                device=device,
                num_batches=args.eval_batches
            )
            
            if rank == 0:
                print(f"Epoch {epoch} - Validation Loss: {dev_loss:.6f}, Exact Match: {dev_exact_match:.6f}")
                
                # Track best model
                is_best = dev_loss < best_eval_loss
                if is_best:
                    best_eval_loss = dev_loss
                    # Save best model
                    best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                        'loss': dev_loss,
                        'exact_match': dev_exact_match,
                    }, best_model_path)
                    print(f"Rank {rank}: New best model saved to {best_model_path}")
        
        # Save checkpoint
        if rank == 0:
            save_path = os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'loss': dev_loss,
                'exact_match': dev_exact_match,
            }, save_path)
            print(f"Checkpoint saved to {save_path}")

if __name__ == '__main__':
    main() 