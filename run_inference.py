#!/usr/bin/env python3
"""
Complete HotpotQA Multi-Hop Inference System
Single file with all functionality for frontier-chain search inference.
"""

import os
import json
import yaml
import torch
import numpy as np
import pickle
import argparse
from tqdm.auto import tqdm
from typing import List, Dict, Any, Tuple
from torch_geometric.data import Data

# Import our modules
from data.build_graph import build_graphs_from_embeddings
from data.make_embeddings import precompute_embeddings
from models.encoder import create_hotpot_graph_encoder
from models.scorer import create_hotpot_prefix_scorer
from metrics import evaluate_predictions, compute_f1_at_k


def infer_frontier_chain(graph_data: Data, encoder, scorer, cfg: Dict[str, Any]) -> List[int]:
    """
    Perform frontier-chain search to find the best reasoning chain.
    
    Args:
        graph_data: torch_geometric.data.Data with graph structure
        encoder: Graph encoder model
        scorer: Prefix-conditioned scorer model
        cfg: Configuration dictionary with inference parameters
        
    Returns:
        best_chain: List of node indices forming the best reasoning chain
    """
    # Extract inference parameters
    chain_width = cfg.get('chain_width', 4)
    max_hops = cfg.get('max_hops', 3)
    top_k_expand = cfg.get('top_k_expand', 5)
    stop_threshold = cfg.get('stop_threshold', 0.25)
    
    # Encode the graph once
    with torch.no_grad():
        H = encoder(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    
    # Get question node index
    q_idx = getattr(graph_data, 'q_idx', None)
    
    # Get paragraph mask to identify valid paragraph nodes
    is_para_mask = getattr(graph_data, 'is_para_mask', None)
    if is_para_mask is not None:
        para_indices = torch.where(is_para_mask)[0].tolist()
    else:
        # Assume all nodes except last (question) are paragraphs
        num_nodes = graph_data.x.size(0)
        para_indices = list(range(num_nodes - (1 if q_idx is not None else 0)))
    
    # Initialize parallel reasoning chains with empty prefixes
    active_chains = [([], 0.0)]
    
    for hop in range(max_hops):
        new_candidates = []
        
        for prefix_ids, prefix_score in active_chains:
            # Get frontier nodes (paragraphs not in current prefix)
            frontier_ids = [idx for idx in para_indices if idx not in prefix_ids]
            
            if not frontier_ids:
                # No more nodes to expand
                new_candidates.append((prefix_ids, prefix_score))
                continue
            
            # Score frontier nodes
            frontier_tensor = torch.tensor(frontier_ids, device=H.device)
            logits = scorer(H, q_idx, prefix_ids, frontier_tensor)
            probs = torch.softmax(logits, dim=0)
            
            # Get top-k candidates for chain expansion
            top_k = min(top_k_expand, len(frontier_ids))
            top_probs, top_indices = torch.topk(probs, top_k)
            
            for i in range(top_k):
                prob = top_probs[i].item()
                node_idx = frontier_ids[top_indices[i].item()]
                
                # Check stopping threshold
                if prob < stop_threshold and len(prefix_ids) > 0:
                    # Don't expand this chain further
                    new_candidates.append((prefix_ids, prefix_score))
                else:
                    # Extend the reasoning chain
                    new_prefix = prefix_ids + [node_idx]
                    new_score = prefix_score + torch.log(top_probs[i]).item()
                    new_candidates.append((new_prefix, new_score))
        
        # Keep top chain_width candidates globally
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        active_chains = new_candidates[:chain_width]
        
        # Let it explore the full search space without early stopping
    
    # Return the best reasoning chain
    best_chain = active_chains[0][0] if active_chains else []
    return best_chain


def load_trained_model(checkpoint_path: str, config: dict, device: torch.device):
    """Load a trained model from checkpoint"""
    
    # Create model
    encoder = create_hotpot_graph_encoder(config, phase=1)
    scorer = create_hotpot_prefix_scorer(config)
    
    # Load checkpoint if it exists
    if os.path.exists(checkpoint_path):
        print(f"📥 Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Extract state dicts from the combined model if needed
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            
            # Split the state dict for encoder and scorer
            encoder_state = {}
            scorer_state = {}
            
            for key, value in state_dict.items():
                if key.startswith('encoder.'):
                    encoder_state[key[8:]] = value
                elif key.startswith('scorer.'):
                    scorer_state[key[7:]] = value
            
            encoder.load_state_dict(encoder_state)
            scorer.load_state_dict(scorer_state)
            
        print(f"✅ Model loaded successfully")
    else:
        print(f"⚠️ No checkpoint found, using random initialization")
    
    encoder.to(device)
    scorer.to(device)
    encoder.eval()
    scorer.eval()
    
    return encoder, scorer


def create_embeddings_if_needed(data_file: str, embeddings_dir: str, config: dict):
    """Create embeddings if they don't exist"""
    
    if os.path.exists(embeddings_dir):
        print(f"✅ Using existing embeddings from {embeddings_dir}")
        return embeddings_dir
    
    print(f"🔄 Creating embeddings for {data_file}...")
    
    # Create temporary files for embedding generation
    temp_dir = os.path.dirname(embeddings_dir) if os.path.dirname(embeddings_dir) else "."
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)
    
    precompute_embeddings(
        train_file=data_file,
        dev_file=data_file,  # Use same file for both
        output_dir=embeddings_dir,
        model_name=config.get("embedding_model", "all-MiniLM-L6-v2"),
        dataset_percentage=100.0
    )
    
    return embeddings_dir


def load_and_build_graphs(data_file: str, embeddings_dir: str, config: dict, split: str = "train"):
    """Load data and convert to graphs"""
    
    print(f"📊 Loading {split} data...")
    
    # Load embeddings
    q_emb_file = os.path.join(embeddings_dir, f"{split}_question_embeddings.npy")
    p_emb_file = os.path.join(embeddings_dir, f"{split}_paragraph_embeddings.pkl")
    meta_file = os.path.join(embeddings_dir, f"{split}_metadata.json")
    
    question_embeddings = np.load(q_emb_file)
    
    with open(p_emb_file, 'rb') as f:
        paragraph_embeddings = pickle.load(f)
        
    with open(meta_file, 'r') as f:
        metadata_list = json.load(f)
    
    print(f"   📈 Questions: {len(question_embeddings)}")
    print(f"   📈 Metadata: {len(metadata_list)}")
    
    # Build graphs
    print(f"🔧 Building graphs...")
    graphs = build_graphs_from_embeddings(
        question_embeddings,
        paragraph_embeddings, 
        metadata_list,
        config
    )
    
    print(f"✅ Built {len(graphs)} graphs")
    return graphs, metadata_list


def run_inference_on_graphs(graphs: List[Data], encoder, scorer, config: dict, device: torch.device):
    """Run inference on all graphs"""
    
    # Get inference parameters from config file
    inference_section = config.get('inference', {})
    inference_config = {
        'chain_width': inference_section.get('chain_width', 8),
        'max_hops': inference_section.get('max_hops', 4),
        'top_k_expand': inference_section.get('top_k_expand', 10),
        'stop_threshold': inference_section.get('stop_threshold', 0.01)
    }
    
    print(f"🔍 Running inference with config: {inference_config}")
    
    predictions = []
    
    for i, graph in enumerate(tqdm(graphs, desc="Inference")):
        graph = graph.to(device)
        
        try:
            # Import improved inference from infer.py
            from infer import infer_frontier_chain as improved_infer
            
            # Extract improved inference config
            max_hops = inference_config.get('max_hops', 3)  # Reduced from 5 to 3
            chain_width = inference_config.get('chain_width', 3)  # Reduced for focus
            adaptive_stopping = inference_config.get('adaptive_stopping', True)
            confidence_threshold = inference_config.get('confidence_threshold', 0.5)
            plateau_patience = inference_config.get('plateau_patience', 2)
            score_improvement_threshold = inference_config.get('score_improvement_threshold', 0.05)
            
            pred_chain = improved_infer(
                graph_data=graph,
                model_tuple=(encoder, scorer),
                max_hops=max_hops,
                chain_width=chain_width,
                adaptive_stopping=adaptive_stopping,
                confidence_threshold=confidence_threshold,
                plateau_patience=plateau_patience,
                score_improvement_threshold=score_improvement_threshold
            )
            predictions.append(pred_chain)
            
        except Exception as e:
            print(f"❌ Error in example {i}: {e}")
            predictions.append([])
    
    return predictions


def evaluate_results(data: List[Dict], predictions: List[List[int]], verbose: bool = True):
    """Evaluate inference results"""
    
    results = []
    
    for i, (example, pred_chain) in enumerate(zip(data, predictions)):
        
        # Get ground truth supporting facts
        supporting_facts = example.get('supporting_facts', [])
        
        # Simple evaluation - count overlap (in practice would need proper paragraph mapping)
        gt_count = len(supporting_facts)
        pred_count = len(pred_chain)
        
        result = {
            'example_id': example.get('_id', f'example_{i}'),
            'question': example.get('question', ''),
            'answer': example.get('answer', ''),
            'gt_supporting_facts': gt_count,
            'pred_chain': pred_chain,
            'pred_length': pred_count,
            'multi_hop': pred_count > 1
        }
        results.append(result)
        
        if verbose:
            question = example.get('question', '')[:100]
            if len(example.get('question', '')) > 100:
                question += "..."
                
            print(f"\n📝 Example {i+1}: {example.get('_id', f'ex_{i}')}")
            print(f"   Q: {question}")
            print(f"   A: {example.get('answer', 'N/A')}")
            print(f"   GT facts: {gt_count}")
            print(f"   Predicted: {pred_chain} (length={pred_count})")
            
            if pred_count > 1:
                print(f"   🎉 MULTI-HOP: {pred_count}-hop reasoning")
            elif pred_count == 1:
                print(f"   ⚠️ Single-hop")
            else:
                print(f"   ❌ Empty prediction")
    
    # Summary statistics
    multi_hop_count = sum(1 for r in results if r['multi_hop'])
    single_hop_count = sum(1 for r in results if r['pred_length'] == 1)
    empty_count = sum(1 for r in results if r['pred_length'] == 0)
    avg_length = np.mean([r['pred_length'] for r in results]) if results else 0
    max_length = max([r['pred_length'] for r in results]) if results else 0
    
    print(f"\n📈 INFERENCE SUMMARY:")
    print(f"   Total examples: {len(results)}")
    print(f"   Multi-hop (>1): {multi_hop_count}/{len(results)} ({multi_hop_count/len(results)*100:.1f}%)")
    print(f"   Single-hop (=1): {single_hop_count}/{len(results)} ({single_hop_count/len(results)*100:.1f}%)")
    print(f"   Empty (=0): {empty_count}/{len(results)} ({empty_count/len(results)*100:.1f}%)")
    print(f"   Average length: {avg_length:.2f}")
    print(f"   Max length: {max_length}")
    
    return results


def main():
    """Main inference function"""
    
    parser = argparse.ArgumentParser(description="HotpotQA Multi-Hop Inference")
    parser.add_argument("--data_file", type=str, default="data/sample_5_objects.json", 
                       help="Input data file (JSON)")
    parser.add_argument("--config", type=str, default="config.yaml", 
                       help="Configuration file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints_5pct/best_model.pth", 
                       help="Model checkpoint")
    parser.add_argument("--output", type=str, default="inference_output.json", 
                       help="Output results file")
    parser.add_argument("--embeddings_dir", type=str, default=None, 
                       help="Embeddings directory (auto-generated if not provided)")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cpu", "cuda"], help="Device to use")
    parser.add_argument("--verbose", action="store_true", 
                       help="Print detailed results")
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ Using device: {device}")
    
    # Load configuration
    print(f"📋 Loading config from {args.config}")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load input data
    print(f"📂 Loading data from {args.data_file}")
    with open(args.data_file, 'r') as f:
        data = json.load(f)
    
    print(f"✅ Loaded {len(data)} examples")
    
    # Setup embeddings directory
    if args.embeddings_dir is None:
        args.embeddings_dir = f"temp_inference/embeddings"
    
    # Create embeddings if needed
    embeddings_dir = create_embeddings_if_needed(args.data_file, args.embeddings_dir, config)
    
    # Load and build graphs
    graphs, metadata = load_and_build_graphs(args.data_file, embeddings_dir, config)
    
    # Load trained model
    encoder, scorer = load_trained_model(args.checkpoint, config, device)
    
    # Run inference
    print(f"\n🚀 STARTING INFERENCE")
    print("=" * 50)
    
    predictions = run_inference_on_graphs(graphs, encoder, scorer, config, device)
    
    # Evaluate results
    print(f"\n📊 EVALUATING RESULTS")
    print("=" * 50)
    
    results = evaluate_results(data, predictions, verbose=args.verbose)
    
    # Save results
    print(f"\n💾 Saving results to {args.output}")
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Inference complete!")
    
    # Success message
    multi_hop_count = sum(1 for r in results if r['multi_hop'])
    if multi_hop_count > 0:
        print(f"🎉 SUCCESS: Found multi-hop reasoning in {multi_hop_count} examples!")
    else:
        print(f"⚠️ No multi-hop reasoning found. Consider adjusting inference parameters.")


if __name__ == "__main__":
    main() 