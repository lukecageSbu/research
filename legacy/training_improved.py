import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from typing import List, Dict, Tuple, Any

# Assuming faiss, sentence_transformers, networkx, sklearn are installed
# import faiss
from sentence_transformers import SentenceTransformer
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train a Quantum Walk Retriever for HotpotQA")
    parser.add_argument('--train_file', type=str, default='/data/research/hotpot/hotpot_train_v1.1.json', help='Path to HotpotQA training file')
    parser.add_argument('--dev_file', type=str, default='/data/research/hotpot/hotpot_dev_distractor_v1.json', help='Path to HotpotQA development file')
    parser.add_argument('--model_path', type=str, default='coin_net_improved.pth', help='Path to save the final model')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_improved', help='Directory to save checkpoints')
    parser.add_argument('--embed_model_name', type=str, default='all-MiniLM-L6-v2', help='Sentence Transformer model name')
    parser.add_argument('--k', type=int, default=5, help='Number of neighbors for graph construction')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension for the coin network')
    parser.add_argument('--walk_steps', type=int, default=3, help='Number of steps for the quantum walk')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate') # Adjusted default LR
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size') # Added batch size
    parser.add_argument('--eval_batch_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA even if available')
    return parser.parse_args()

# --- QuantumWalkRetriever Model ---
class QuantumWalkRetriever(nn.Module):
    def __init__(self, embed_model_name='all-MiniLM-L6-v2', k=5, hidden_dim=128, walk_steps=3):
        super().__init__()
        self.embedder = SentenceTransformer(embed_model_name)
        # Freeze embedder parameters
        for p in self.embedder.parameters(): p.requires_grad = False
        self.k, self.walk_steps = k, walk_steps
        d = self.embedder.get_sentence_embedding_dimension()
        self.coin_net = nn.Sequential(
            nn.Linear(d*2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, k)
        )
        self.embedder_dim = d

    def embed_sentences(self, sents: List[str], device: torch.device) -> torch.Tensor:
        # DEPRECATED - embedding is now batched in forward pass
        # Ensure embedder is on the correct device (usually CPU for SBERT, computation on GPU later)
        # Embeddings are returned as numpy, convert to tensor and move to device
        # embeddings = self.embedder.encode(sents, convert_to_tensor=False, show_progress_bar=False)
        # return torch.from_numpy(embeddings).float().to(device)
        raise DeprecationWarning("embed_sentences is deprecated. Use batch embedding within the forward pass.")
        pass # Add this line

    def build_graph(self, emb: np.ndarray) -> nx.Graph:
        # Keep graph building on CPU with numpy/networkx
        sim = cosine_similarity(emb)
        n = len(emb)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
             # Ensure we don't include self-loops and handle cases with fewer than k unique similarities
            nbrs_indices = np.argsort(sim[i])[::-1]
            valid_nbrs = [idx for idx in nbrs_indices if idx != i][:self.k]
            for j in valid_nbrs:
                G.add_edge(i, j, weight=sim[i,j])
        return G

    def quantum_walk(self, G: nx.Graph, qv_t: torch.Tensor, emb_t: torch.Tensor) -> torch.Tensor:
        # Perform walk on the specified device
        device = qv_t.device
        n = G.number_of_nodes()
        if n == 0: # Handle empty graphs
             return torch.empty(0, device=device)

        max_k = max(G.degree(i) for i in range(n)) # Max actual degree
        effective_k = min(self.k, max_k) # Use the smaller of configured k or max degree

        if effective_k == 0: # Handle graphs with isolated nodes
            return torch.ones(n, device=device) / n # Uniform probability if no edges

        # Initial state: uniform superposition
        state = torch.ones(n, effective_k, dtype=torch.cfloat, device=device) / np.sqrt(n * effective_k)

        nbr_lists = []
        for i in range(n):
             neighbors = list(G.neighbors(i))
             # Pad neighbors list if fewer than effective_k
             padded_neighbors = neighbors + [i] * (effective_k - len(neighbors)) # Pad with self-loops
             nbr_lists.append(padded_neighbors[:effective_k]) # Ensure exactly effective_k

        for _ in range(self.walk_steps):
            coins = torch.zeros(n, effective_k, effective_k, dtype=torch.cfloat, device=device)
            for i in range(n):
                # Project coin net output to match effective_k if necessary
                coin_input_dim = self.coin_net[0].in_features
                if emb_t[i].shape[0] * 2 != coin_input_dim: # Check dimension match
                    # Handle potential dimension mismatch (e.g., if embedder dim changed)
                    # This case should ideally not happen if initialized correctly
                     raise ValueError(f"Dimension mismatch: emb_t {emb_t[i].shape}, qv_t {qv_t.shape}, expected input {coin_input_dim}")

                inp = torch.cat([emb_t[i], qv_t])
                amps = self.coin_net(inp)[:effective_k] # Select first effective_k amplitudes

                # Normalize amplitudes (optional, helps stability)
                amps_norm = amps / (torch.norm(amps) + 1e-8) # Add epsilon for stability

                # Construct coin matrix (outer product)
                c_real = amps_norm.unsqueeze(1) * amps_norm.unsqueeze(0)

                # Normalize the coin matrix - crucial for unitary property (approximation)
                coin_op = (c_real.to(torch.cfloat)) / (torch.linalg.norm(c_real) + 1e-8)
                coins[i, :effective_k, :effective_k] = coin_op # Assign to the block


            new_state = torch.zeros_like(state)

            # Batched Coin Application (Conceptual - matmul needs loop or advanced batching)
            # This part is tricky to fully batch without specialized libraries (like PyG)
            # We loop through nodes for clarity, though matmul could be batched if states/coins are stacked carefully
            for i in range(n):
                if state[i].numel() == 0 or coins[i].numel() == 0: continue # Skip if empty
                try:
                     s_p = coins[i] @ state[i] # Apply coin: (k,k) @ (k,) -> (k,)
                except RuntimeError as e:
                     print(f"RuntimeError in coin application at node {i}: {e}")
                     print(f"Coin shape: {coins[i].shape}, State shape: {state[i].shape}")
                     # Handle error, maybe skip update or use identity coin
                     s_p = state[i] # Fallback
                     continue

                # Shift Operation (Gather/Scatter equivalent)
                node_neighbors = nbr_lists[i] # List of neighbor indices
                for idx, neighbor_node in enumerate(node_neighbors):
                     # Find where neighbor_node appears in other nodes' neighbor lists to map incoming state
                     try:
                         # Find the index corresponding to node `i` in neighbor `j`'s neighbor list
                         target_neighbor_idx = nbr_lists[neighbor_node].index(i) # Find the slot in neighbor's state
                         new_state[neighbor_node, target_neighbor_idx] += s_p[idx]
                     except ValueError:
                          # Node i is not in neighbor_node's top k list, amplitude effectively lost/reflected
                          # This can happen in non-regular graphs or if k is small
                          pass # Or implement reflection logic if needed
                     except IndexError:
                          print(f"IndexError: neighbor_node={neighbor_node}, target_neighbor_idx attempt failed.")
                          pass


            # Normalization per node/walker (ensures total probability is 1)
            # Normalize the entire state tensor to maintain probability distribution interpretation
            state_norm = torch.sqrt(torch.sum(new_state.abs()**2)) + 1e-8 # Global norm
            state = new_state / state_norm

        # Final probabilities: sum of squared magnitudes over the k dimension
        final_probs = torch.sum(state.abs()**2, dim=1)
        # Handle potential NaN/Inf from calculations
        final_probs = torch.nan_to_num(final_probs, nan=0.0, posinf=1.0, neginf=0.0)
        return final_probs # Return probabilities (N,)

    def forward(self, batch: Dict[str, Any], device: torch.device) -> List[torch.Tensor]:
        # Process a batch of examples
        questions = batch['questions']
        sentence_lists = batch['sentences'] # List of lists of strings
        batch_size = len(questions)
        # Note: Labels are handled in the training loop

        # --- Batch Question Embedding ---
        q_emb_np = self.embedder.encode(questions, convert_to_numpy=True, show_progress_bar=False)
        q_emb_t = torch.from_numpy(q_emb_np).float().to(device)

        # --- Batch Sentence Embedding ---
        # 1. Flatten sentence lists and track counts
        all_sents = []
        sentences_per_example = []
        example_indices_for_sents = [] # Track which example each sentence belongs to
        valid_example_indices = [] # Track indices of examples with sentences

        for i, sents in enumerate(sentence_lists):
            num_sents = len(sents)
            if num_sents > 0:
                 all_sents.extend(sents)
                 sentences_per_example.append(num_sents)
                 example_indices_for_sents.extend([i] * num_sents)
                 valid_example_indices.append(i)
            # Else: Example has no sentences, will be skipped later

        # 2. Embed all sentences in one go (if any exist)
        if not all_sents:
            # Handle case where the entire batch has no sentences
            return [torch.empty(0, device=device) for _ in range(batch_size)]

        all_sent_emb_np = self.embedder.encode(all_sents, convert_to_numpy=True, show_progress_bar=False)
        # Note: SBERT often returns float32 numpy arrays directly
        all_sent_emb_t = torch.from_numpy(all_sent_emb_np).float().to(device)


        # --- Process Each Example (Graph Build + Walk - Bottleneck) ---
        batch_logits = [torch.empty(0, device=device)] * batch_size # Initialize placeholders for all examples
        current_sent_idx = 0
        valid_example_counter = 0 # Counter for examples we are actually processing

        for i in range(batch_size):
             if i not in valid_example_indices:
                 # This example had no sentences, keep empty tensor placeholder
                 continue

             num_sents = sentences_per_example[valid_example_counter]

             # 3. Get the slice of embeddings for the current example
             sent_emb_np_slice = all_sent_emb_np[current_sent_idx : current_sent_idx + num_sents]
             sent_emb_t_slice = all_sent_emb_t[current_sent_idx : current_sent_idx + num_sents]

             if sent_emb_t_slice.shape[0] == 0: # Double check slice validity
                  # Should not happen if valid_example_indices logic is correct
                  current_sent_idx += num_sents
                  valid_example_counter += 1
                  continue


             # Build graph (CPU) using the numpy slice
             G = self.build_graph(sent_emb_np_slice)

             # Perform quantum walk (GPU/CPU) using the tensor slice
             current_q_emb = q_emb_t[i]
             # Pass tensor slice for embeddings
             logits = self.quantum_walk(G, current_q_emb, sent_emb_t_slice)
             batch_logits[i] = logits # Place logits in the correct position in the batch list

             # Update indices for next iteration
             current_sent_idx += num_sents
             valid_example_counter += 1


        return batch_logits # List of tensors, one per example (some might be empty)

# --- Data Loading ---
def load_hotpot(fn: str) -> List[Dict]:
    try:
        with open(fn, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} examples from {fn}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {fn}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {fn}")
        return []

def prepare_examples(data: List[Dict]) -> List[Dict]:
    exs = []
    skipped_no_sp = 0
    skipped_no_ctx = 0
    for ex in data:
        q = ex['question']
        sents, lbls = [], []
        sp_titles = {fact[0] for fact in ex.get('supporting_facts', [])}
        context_map = {title: slist for title, slist in ex['context']}

        if not ex['context']:
            skipped_no_ctx += 1
            continue

        has_sp_in_ctx = False
        for title, slist in ex['context']:
            is_sp_title = title in sp_titles
            for sid, s in enumerate(slist):
                sents.append(s.strip()) # Add sentence text
                is_sp_fact = is_sp_title and [title, sid] in ex.get('supporting_facts',[])
                lbls.append(1.0 if is_sp_fact else 0.0) # Use float for labels (0.0 or 1.0)
                if is_sp_fact:
                    has_sp_in_ctx = True

        # Keep example only if it has context and at least one supporting fact present in the context
        # No need to filter based on SP presence anymore if using BCE loss
        # if sents and has_sp_in_ctx:
        if sents: # Keep if sentences exist
             # ---- REMOVED LABEL NORMALIZATION for BCE ----
             # label_sum = sum(lbls)
             # if label_sum > 0:
             #      normalized_lbls = [l / label_sum for l in lbls]
             # else:
             #      normalized_lbls = lbls # Keep as zeros if sum is zero

             exs.append({'id': ex['_id'], 'question': q, 'sentences': sents, 'labels': lbls}) # Use raw 0/1 labels
        # elif sents and not has_sp_in_ctx: # Keep examples even if no SP in ctx? Decide based on task req. Let's keep them.
             # skipped_no_sp += 1
             #pass # If we decide to skip them
        # else: skipped implicitly if no sents


    print(f"Prepared {len(exs)} examples.")
    # if skipped_no_sp > 0: print(f"Skipped {skipped_no_sp} examples with context but no SP facts listed in context.") # Comment out if keeping all
    if skipped_no_ctx > 0: print(f"Skipped {skipped_no_ctx} examples with no context.")
    return exs

class HotpotDataset(Dataset):
    def __init__(self, examples: List[Dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]

def collate_fn(batch: List[Dict]) -> Dict[str, Any]:
    # Custom collate function to handle variable length sentence lists
    questions = [item['question'] for item in batch]
    sentence_lists = [item['sentences'] for item in batch]
    labels_lists = [item['labels'] for item in batch]
    ids = [item['id'] for item in batch]

    # No padding needed here as graph/walk handles variable size per example
    # If we were batching graph operations, padding would be needed.

    return {
        'ids': ids,
        'questions': questions,
        'sentences': sentence_lists,
        'labels': labels_lists # Keep as list of lists/tensors
    }

# --- Evaluation ---
def evaluate(model: QuantumWalkRetriever, dataloader: DataLoader, loss_fn: nn.Module, device: torch.device):
    model.eval()
    total_loss = 0.0
    total_examples = 0
    skipped_batches = 0
    # --- Placeholder for Metrics ---
    all_predictions = []
    all_true_labels = []
    # --- End Placeholder ---

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            try:
                batch_logits_list = model(batch, device) # Returns list of tensors
                batch_labels = batch['labels'] # List of lists/tensors

                batch_loss = 0.0
                valid_examples_in_batch = 0

                for i, logits in enumerate(batch_logits_list):
                    labels = torch.tensor(batch_labels[i], dtype=torch.float).to(device)

                    # Ensure logits and labels are compatible for loss calculation
                    if logits.numel() == 0 or labels.numel() == 0 or logits.shape[0] != labels.shape[0]:
                        # Skip example if shapes mismatch or empty (e.g., no sentences)
                        # print(f"Skipping eval example {batch['ids'][i]} due to shape mismatch or empty tensors. Logits: {logits.shape}, Labels: {labels.shape}")
                        continue

                    # --- ADJUSTED Loss Calculation for BCE ---
                    # Use raw logits directly with BCEWithLogitsLoss
                    loss = loss_fn(logits, labels)
                    # --- End Adjustment ---


                    if not torch.isnan(loss) and not torch.isinf(loss):
                        batch_loss += loss.item()
                        valid_examples_in_batch += 1
                        # --- Placeholder for Metrics Collection ---
                        # Apply sigmoid to logits to get probabilities if needed for metrics
                        # preds = torch.sigmoid(logits).detach().cpu().numpy()
                        # true = labels.detach().cpu().numpy()
                        # # Decide on prediction strategy (e.g., threshold > 0.5)
                        # predicted_labels = (preds > 0.5).astype(int)
                        # all_predictions.append(predicted_labels)
                        # all_true_labels.append(true)
                        # --- End Placeholder ---

                    # else:
                        # print(f"Warning: NaN/Inf loss encountered for example {batch['ids'][i]}. Logits: {logits}, Labels: {labels}")


                if valid_examples_in_batch > 0:
                    total_loss += batch_loss
                    total_examples += valid_examples_in_batch
                else:
                    skipped_batches += 1

            except Exception as e:
                 print(f"Error during evaluation batch: {e}")
                 skipped_batches += 1
                 continue # Skip batch on error

    if total_examples > 0:
        avg_loss = total_loss / total_examples
        print(f"Evaluation Avg Loss: {avg_loss:.4f} ({total_examples} examples evaluated)")
    else:
        avg_loss = float('inf')
        print("Evaluation failed: No examples could be processed.")

    # --- Placeholder for Metric Calculation ---
    # if all_predictions and all_true_labels:
         # Flatten lists if needed
         # true_flat = np.concatenate(all_true_labels)
         # pred_flat = np.concatenate(all_predictions)
         # Calculate metrics (Precision, Recall, F1) using sklearn.metrics
         # from sklearn.metrics import precision_recall_fscore_support
         # precision, recall, f1, _ = precision_recall_fscore_support(true_flat, pred_flat, average='binary') # or 'micro'/'macro'
         # print(f"Evaluation Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    # else:
         # print("Could not calculate evaluation metrics (no valid examples found).")
    # --- End Placeholder ---


    if skipped_batches > 0:
         print(f"Skipped {skipped_batches} evaluation batches due to errors or empty results.")

    return avg_loss


# --- Main Training Loop ---
def main():
    args = parse_args()

    # Setup device
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    last_ckpt_path = os.path.join(args.checkpoint_dir, 'last.ckpt')

    # Load data
    print("Loading and preparing training data...")
    train_data = load_hotpot(args.train_file)
    train_examples = prepare_examples(train_data)
    train_dataset = HotpotDataset(train_examples)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print("Loading and preparing development data...")
    dev_data = load_hotpot(args.dev_file)
    dev_examples = prepare_examples(dev_data)
    dev_dataset = HotpotDataset(dev_examples)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    # Initialize model, optimizer
    print("Initializing model...")
    retriever = QuantumWalkRetriever(
        embed_model_name=args.embed_model_name,
        k=args.k,
        hidden_dim=args.hidden_dim,
        walk_steps=args.walk_steps
    ).to(device)
    # Only optimize coin_net parameters
    optimizer = optim.Adam(retriever.coin_net.parameters(), lr=args.lr)

    # Define loss function - Switched to BCEWithLogitsLoss
    loss_fn = nn.BCEWithLogitsLoss(reduction='sum') # Sum losses, average manually per epoch
    print(f"Using loss function: {type(loss_fn).__name__}")


    # Resume from checkpoint if exists
    start_epoch = 0
    if os.path.exists(last_ckpt_path):
        print(f"Loading checkpoint from {last_ckpt_path}")
        try:
            ckpt = torch.load(last_ckpt_path, map_location=device)
            retriever.coin_net.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt['epoch'] + 1
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting from scratch.")
            start_epoch = 0


    # Training loop
    print("Starting training...")
    best_eval_loss = float('inf')

    for ep in range(start_epoch, args.epochs):
        retriever.train()
        epoch_loss = 0.0
        examples_processed = 0
        skipped_batches_train = 0

        for batch in tqdm(train_dataloader, desc=f'Train Epoch {ep+1}/{args.epochs}'):
            optimizer.zero_grad()
            try:
                batch_logits_list = retriever(batch, device) # List of tensors (logits per example)
                batch_labels = batch['labels'] # List of lists/tensors

                current_batch_loss = 0.0
                valid_examples_in_batch = 0

                for i, logits in enumerate(batch_logits_list):
                     labels = torch.tensor(batch_labels[i], dtype=torch.float).to(device)

                     # Check for compatibility
                     if logits.numel() == 0 or labels.numel() == 0 or logits.shape[0] != labels.shape[0]:
                         # print(f"Skipping train example {batch['ids'][i]} due to shape mismatch or empty. Logits: {logits.shape}, Labels: {labels.shape}")
                         continue

                     # --- REMOVED Label sum check for KLDiv ---
                     # if isinstance(loss_fn, nn.KLDivLoss) and not torch.isclose(labels.sum(), torch.tensor(1.0)):
                     #     # print(f"Skipping train example {batch['ids'][i]} for KLDiv: labels sum to {labels.sum()}")
                     #     continue # Skip examples where KLDiv is undefined/unstable


                     # --- ADJUSTED Loss Calculation for BCE ---
                     # Pass raw logits directly to BCEWithLogitsLoss
                     loss = loss_fn(logits, labels)
                     # --- End Adjustment ---


                     if not torch.isnan(loss) and not torch.isinf(loss):
                         current_batch_loss += loss
                         valid_examples_in_batch += 1


                if valid_examples_in_batch > 0 and isinstance(current_batch_loss, torch.Tensor):
                    # Average loss over valid examples in the batch before backward pass?
                    # Or sum is fine if optimizer handles scaling? Let's stick with sum for loss_fn reduction='sum'
                    # The gradients will be accumulated.
                    current_batch_loss.backward() # Backward pass on the accumulated loss for the batch
                    optimizer.step()
                    epoch_loss += current_batch_loss.item() # Add sum of losses for the batch
                    examples_processed += valid_examples_in_batch
                else:
                    skipped_batches_train += 1

            except Exception as e:
                 print(f"\nError during training batch: {e}")
                 print(f"Problematic Batch IDs: {batch.get('ids', 'N/A')}")
                 skipped_batches_train += 1
                 # Optionally add more debugging here (e.g., print shapes)
                 continue # Skip batch on error

        if examples_processed > 0:
             avg_epoch_loss = epoch_loss / examples_processed
             print(f'Epoch {ep+1}, Avg Train Loss: {avg_epoch_loss:.4f}')
        else:
             print(f'Epoch {ep+1}: No examples successfully processed.')


        if skipped_batches_train > 0:
             print(f"Skipped {skipped_batches_train} training batches due to errors or empty results.")


        # Evaluation step
        print(f"\n--- Evaluating Epoch {ep+1} ---")
        eval_loss = evaluate(retriever, dev_dataloader, loss_fn, device)

        # Save checkpoint
        ckpt = {
            'epoch': ep,
            'model_state': retriever.coin_net.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'eval_loss': eval_loss
        }
        torch.save(ckpt, last_ckpt_path)
        print(f"Checkpoint saved to {last_ckpt_path}")

        # Save model if it's the best so far based on eval loss
        if eval_loss < best_eval_loss:
             best_eval_loss = eval_loss
             torch.save(retriever.coin_net.state_dict(), args.model_path)
             print(f"New best model saved to {args.model_path} (Eval Loss: {eval_loss:.4f})")
        print("--- End Epoch Evaluation ---")


    print("\nTraining complete.")
    if os.path.exists(args.model_path):
         print(f"Final best model saved to {args.model_path}")
    else:
         print("No best model saved (evaluation might have failed or not improved).")


if __name__ == '__main__':
    main() 