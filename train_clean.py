#!/usr/bin/env python3
"""
Clean training script for HotpotQA GNN model.
Uses config file properly and reuses embedding functions.
"""

import os
# Set tokenizer parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import yaml
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader as PyGDataLoader
import argparse
from tqdm.auto import tqdm
import time

# Import our models and utilities
from models.encoder import create_hotpot_graph_encoder
from models.scorer import create_hotpot_prefix_scorer
from data.build_graph import build_graphs_from_embeddings
from utils.batching import extract_graph_labels, create_dataloader, analyze_batch_statistics
from utils.seeds import set_seed, get_reproducible_generator

# Import embedding functions from make_embeddings
from data.make_embeddings import precompute_embeddings, process_hotpot_file
from sentence_transformers import SentenceTransformer

# Import evaluation metrics
from metrics import evaluate_predictions, compute_f1_at_k, compute_precision_at_k, compute_recall_at_k

class HotpotGNNModel(torch.nn.Module):
    """GNN Model for HotpotQA"""
    
    def __init__(self, config: dict, phase: int = 1):
        super().__init__()
        self.config = config
        self.encoder = create_hotpot_graph_encoder(config, phase=phase)
        self.scorer = create_hotpot_prefix_scorer(config)
        self.hidden_dim = config.get("model", {}).get("hidden_dim", 256)
    
    def forward(self, batch):
        # Encode nodes
        H = self.encoder(batch.x, batch.edge_index, batch.edge_attr)
        
        # Extract labels
        labels_list = extract_graph_labels(batch.y, batch.batch, batch.is_para_mask)
        
        # Score candidates for each graph
        logits_list = []
        graph_ptr = batch.ptr
        
        for graph_idx in range(batch.num_graphs):
            start_node = graph_ptr[graph_idx].item()
            end_node = graph_ptr[graph_idx + 1].item()
            
            H_graph = H[start_node:end_node]
            graph_mask = batch.batch == graph_idx
            is_para_mask_graph = batch.is_para_mask[graph_mask]
            
            # Find question node and paragraph candidates
            q_idx = 0 if not is_para_mask_graph[0] else None
            para_indices = torch.where(is_para_mask_graph)[0]
            
            if len(para_indices) > 0:
                logits = self.scorer(H_graph, q_idx, [], para_indices)
                logits_list.append(logits)
        
        return logits_list, labels_list


def create_or_load_embeddings(config: dict, force_recreate: bool = False):
    """Create embeddings if they don't exist or if force_recreate is True"""
    
    data_config = config["data"]
    embeddings_dir = "embeddings_5pct"
    train_file = config["paths"]["train_file"]
    dev_file = config["paths"]["dev_file"]
    
    # Check if embeddings exist
    train_emb_file = os.path.join(embeddings_dir, "train_question_embeddings.npy")
    
    if not os.path.exists(train_emb_file) or force_recreate:
        print("🚀 Creating embeddings...")
        dataset_percentage = data_config["dataset_fraction"] * 100  # Convert 0.05 to 5.0
        model_name = data_config["embedding_model"]
        
        precompute_embeddings(
            train_file=train_file,
            dev_file=dev_file,
            output_dir=embeddings_dir,
            dataset_percentage=dataset_percentage,
            model_name=model_name
        )
    else:
        print(f"✅ Using existing embeddings from {embeddings_dir}")
    
    return embeddings_dir


def load_data_from_embeddings(embeddings_dir: str, config: dict, split: str):
    """Load data from precomputed embeddings"""
    import pickle
    import json
    import numpy as np
    
    print(f"🔄 Loading {split} data from {embeddings_dir}...")
    
    # Load embeddings
    q_emb_file = os.path.join(embeddings_dir, f"{split}_question_embeddings.npy")
    p_emb_file = os.path.join(embeddings_dir, f"{split}_paragraph_embeddings.pkl")
    meta_file = os.path.join(embeddings_dir, f"{split}_metadata.json")
    
    print(f"  📊 Loading question embeddings from {q_emb_file}")
    question_embeddings = np.load(q_emb_file)
    print(f"     Shape: {question_embeddings.shape}")
    
    print(f"  📊 Loading paragraph embeddings from {p_emb_file}")
    with open(p_emb_file, 'rb') as f:
        paragraph_embeddings = pickle.load(f)
    print(f"     Count: {len(paragraph_embeddings)} questions")
    
    print(f"  📊 Loading metadata from {meta_file}")
    with open(meta_file, 'r') as f:
        metadata_list = json.load(f)
    print(f"     Count: {len(metadata_list)} examples")
    
    # Build graphs with progress tracking (handled by build_graphs_from_embeddings)
    print(f"🔧 Converting embeddings to graphs for {split} split...")
    graphs = build_graphs_from_embeddings(
        question_embeddings,
        paragraph_embeddings, 
        metadata_list,
        config
    )
    
    # Show graph statistics
    total_nodes = sum(g.x.size(0) for g in graphs)
    total_edges = sum(g.edge_index.size(1) for g in graphs)
    avg_nodes = total_nodes / len(graphs) if graphs else 0
    avg_edges = total_edges / len(graphs) if graphs else 0
    
    print(f"📈 {split.upper()} Graph Statistics:")
    print(f"   Total graphs: {len(graphs)}")
    print(f"   Avg nodes per graph: {avg_nodes:.1f}")
    print(f"   Avg edges per graph: {avg_edges:.1f}")
    print(f"   Total nodes: {total_nodes}")
    print(f"   Total edges: {total_edges}")
    
    return graphs


def train_epoch(model, dataloader, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        
        # Forward pass
        with autocast(device_type="cuda"):
            logits_list, labels_list = model(batch)
        
        if not logits_list:
            continue
        
        # Calculate loss
        batch_loss = 0.0
        valid_examples = 0
        
        for logits, labels in zip(logits_list, labels_list):
            if logits.numel() == 0 or labels.numel() == 0 or logits.size(0) != labels.size(0):
                continue
            
            loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
            
            if not (torch.isnan(loss) or torch.isinf(loss)):
                batch_loss += loss
                valid_examples += 1
        
        if valid_examples > 0:
            batch_loss = batch_loss / valid_examples
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += batch_loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else float("inf")


def evaluate_model(model, dataloader, device):
    """Evaluate the model with proper metrics"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Collect predictions and targets for metrics
    all_predictions = []
    all_targets = []
    
    print("  🔍 Evaluating...")
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            
            logits_list, labels_list = model(batch)
            
            if not logits_list:
                continue
            
            batch_loss = 0.0
            valid_examples = 0
            
            for logits, labels in zip(logits_list, labels_list):
                if logits.numel() == 0 or labels.numel() == 0 or logits.size(0) != labels.size(0):
                    continue
                
                loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction="mean")
                
                if not (torch.isnan(loss) or torch.isinf(loss)):
                    batch_loss += loss
                    valid_examples += 1
                
                # Store predictions and targets for metrics
                try:
                    # Convert logits to probabilities
                    probs = torch.sigmoid(logits).cpu().numpy()
                    target_labels = labels.cpu().numpy()
                    
                    all_predictions.append(probs)
                    all_targets.append(target_labels)
                except Exception:
                    pass  # Skip if error in metrics collection
            
            if valid_examples > 0:
                batch_loss = batch_loss / valid_examples
                total_loss += batch_loss.item()
                num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    
    # Calculate metrics if we have predictions
    metrics = {}
    if all_predictions and all_targets:
        try:
            from metrics import evaluate_predictions
            import numpy as np
            
            # Calculate evaluation metrics using the original list format
            if all_predictions and all_targets:
                metrics = evaluate_predictions(all_predictions, all_targets)
                
        except Exception as e:
            print(f"Warning: Could not compute metrics: {e}")
            metrics = {}
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_5pct", help="Checkpoint directory")
    parser.add_argument("--force_recreate_embeddings", action="store_true", help="Force recreate embeddings")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--analyze_batches", action="store_true", help="Analyze batch statistics")
    args = parser.parse_args()
    
    # Set seed for reproducibility
    print(f"🌱 Setting random seed: {args.seed}")
    set_seed(args.seed, deterministic=False)
    
    # Load configuration
    print(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Add paths to config (these should really be in the config file)
    config["paths"] = {
        "train_file": "hotpot/hotpot_train_v1.1.json",
        "dev_file": "hotpot/hotpot_dev_distractor_v1.json"
    }
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create or load embeddings
    embeddings_dir = create_or_load_embeddings(config, args.force_recreate_embeddings)
    
    # Load datasets
    print("\n" + "="*60)
    print("📚 LOADING DATASETS")
    print("="*60)
    train_graphs = load_data_from_embeddings(embeddings_dir, config, "train")
    print()
    dev_graphs = load_data_from_embeddings(embeddings_dir, config, "dev")
    
    # Create data loaders using config and utils
    training_config = config["training"]
    data_config = config["data"]
    
    print("🔄 Creating optimized data loaders...")
    train_dataloader = create_dataloader(
        train_graphs, 
        batch_size=training_config["batch_size"], 
        shuffle=True,
        num_workers=data_config["num_workers"]
    )
    dev_dataloader = create_dataloader(
        dev_graphs, 
        batch_size=training_config["batch_size"], 
        shuffle=False,
        num_workers=data_config["num_workers"]
    )
    
    # Analyze batch statistics if requested
    if args.analyze_batches:
        print("\n📊 BATCH STATISTICS ANALYSIS")
        print("="*60)
        train_stats = analyze_batch_statistics(train_dataloader)
        print("TRAINING BATCHES:")
        for key, value in train_stats.items():
            if key not in ['node_count_distribution', 'edge_count_distribution']:
                print(f"  {key}: {value}")
        
        dev_stats = analyze_batch_statistics(dev_dataloader)
        print("\nDEV BATCHES:")
        for key, value in dev_stats.items():
            if key not in ['node_count_distribution', 'edge_count_distribution']:
                print(f"  {key}: {value}")
        print("="*60)
    
    # Create model
    print("🤖 Creating GNN model...")
    model = HotpotGNNModel(config, phase=1).to(device)
    
    # Count model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 Model Parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Setup training components using config
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=training_config["lr"], 
        weight_decay=training_config["weight_decay"]
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=training_config["epochs"])
    scaler = GradScaler()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print(f"\n" + "="*60)
    print(f"🚀 STARTING TRAINING")
    print("="*60)
    print(f"Epochs: {training_config['epochs']}")
    print(f"Batch size: {training_config['batch_size']}")
    print(f"Learning rate: {training_config['lr']}")
    print(f"Train graphs: {len(train_graphs)}")
    print(f"Dev graphs: {len(dev_graphs)}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print("="*60)
    
    best_eval_loss = float("inf")
    
    for epoch in range(training_config["epochs"]):
        print(f"\n📊 EPOCH {epoch + 1}/{training_config['epochs']}")
        print("-" * 40)
        
        # Training
        start_time = time.time()
        train_loss = train_epoch(model, train_dataloader, optimizer, scaler, device)
        train_time = time.time() - start_time
        print(f"🏋️  Training Loss: {train_loss:.6f} (took {train_time:.1f}s)")
        
        # Evaluation
        start_time = time.time()
        eval_loss, eval_metrics = evaluate_model(model, dev_dataloader, device)
        eval_time = time.time() - start_time
        print(f"📏 Evaluation Loss: {eval_loss:.6f} (took {eval_time:.1f}s)")
        
        # Log best metrics if available
        if eval_metrics and 'f1_at_2' in eval_metrics:
            print(f"🎯 Key Metrics: F1@2={eval_metrics['f1_at_2']:.4f}, MAP={eval_metrics.get('map', 0.0):.4f}")
        
        # Step scheduler
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        print(f"📉 Learning rate: {current_lr:.2e} → {new_lr:.2e}")
        
        # Save best model
        if eval_loss < best_eval_loss:
            improvement = best_eval_loss - eval_loss
            best_eval_loss = eval_loss
            checkpoint_path = os.path.join(args.checkpoint_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'eval_loss': eval_loss,
                'eval_metrics': eval_metrics,
                'config': config,
                'seed': args.seed,
                'train_loss': train_loss
            }, checkpoint_path)
            print(f"🏆 NEW BEST MODEL! Improved by {improvement:.6f} → {best_eval_loss:.6f}")
        else:
            print(f"📊 Best so far: {best_eval_loss:.6f}")
        
        # Save latest checkpoint
        latest_path = os.path.join(args.checkpoint_dir, "latest_model.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'eval_loss': eval_loss,
            'eval_metrics': eval_metrics,
            'config': config,
            'seed': args.seed,
            'train_loss': train_loss
        }, latest_path)
    
    print(f"\n🎉 Training complete! Best eval loss: {best_eval_loss:.6f}")


if __name__ == "__main__":
    main() 