"""
Inference module implementing frontier-chain search for multi-hop reasoning.
"""

import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from torch_geometric.data import Data
from models.encoder import GraphEncoder
from models.scorer import PrefixScorer


def infer_frontier_chain(graph_data: Any, model_tuple: Tuple, 
                        max_hops: int = 3, chain_width: int = 3,
                        adaptive_stopping: bool = True,
                        confidence_threshold: float = 0.5,
                        plateau_patience: int = 2,
                        score_improvement_threshold: float = 0.05) -> List[int]:
    """
    Improved frontier-based multi-hop inference with adaptive stopping.
    
    Args:
        graph_data: PyTorch Geometric graph
        model_tuple: (encoder, scorer) models
        max_hops: Maximum number of hops (reduced to 3 for supporting facts)
        chain_width: Number of parallel chains to maintain
        adaptive_stopping: Whether to stop early based on confidence
        confidence_threshold: Minimum confidence to continue expansion
        plateau_patience: Stop if score doesn't improve for N hops
        score_improvement_threshold: Minimum improvement to continue
        
    Returns:
        List of paragraph indices representing the reasoning chain
    """
    encoder, scorer = model_tuple
    device = next(encoder.parameters()).device
    
    # Move graph to device
    graph_data = graph_data.to(device)
    
    # Encode graph
    with torch.no_grad():
        H = encoder(graph_data.x, graph_data.edge_index, graph_data.edge_attr)
    
    # Find question node (first non-paragraph node)
    is_para_mask = graph_data.is_para_mask
    q_idx = None
    for i, is_para in enumerate(is_para_mask):
        if not is_para:
            q_idx = i
            break
    
    # Get paragraph indices
    num_nodes = len(is_para_mask)
    if q_idx is not None:
        para_indices = [i for i in range(num_nodes) if i != q_idx and is_para_mask[i]]
    else:
        para_indices = list(range(num_nodes - (1 if q_idx is not None else 0)))
    
    # Initialize parallel reasoning chains with empty prefixes
    # Each chain item: (prefix_ids, score, hop_scores)
    active_chains = [([], 0.0, [])]
    
    for hop in range(max_hops):
        new_candidates = []
        best_hop_score = -float('inf')
        
        for prefix_ids, prefix_score, hop_scores in active_chains:
            # Get frontier nodes (paragraphs not in current prefix)
            frontier_ids = [idx for idx in para_indices if idx not in prefix_ids]
            
            if not frontier_ids:
                # No more nodes to expand - keep current chain
                new_candidates.append((prefix_ids, prefix_score, hop_scores))
                continue
            
            # Score all frontier candidates
            frontier_scores = scorer(H, q_idx, prefix_ids, frontier_ids)
            
            if frontier_scores.numel() == 0:
                new_candidates.append((prefix_ids, prefix_score, hop_scores))
                continue
            
            # Convert to probabilities
            frontier_probs = torch.sigmoid(frontier_scores).detach().cpu().numpy()
            
            # Expand with top candidates
            for i, frontier_idx in enumerate(frontier_ids):
                candidate_score = frontier_probs[i]
                new_prefix = prefix_ids + [frontier_idx]
                new_total_score = prefix_score + candidate_score
                new_hop_scores = hop_scores + [candidate_score]
                
                new_candidates.append((new_prefix, new_total_score, new_hop_scores))
                best_hop_score = max(best_hop_score, candidate_score)
        
        if not new_candidates:
            break
            
        # Sort by total score and keep top candidates
        new_candidates.sort(key=lambda x: x[1], reverse=True)
        active_chains = new_candidates[:chain_width]
        
        # ADAPTIVE STOPPING CRITERIA
        if adaptive_stopping and len(active_chains) > 0:
            best_chain = active_chains[0]
            
            # 1. Confidence threshold: Stop if best candidate score is too low
            if best_hop_score < confidence_threshold:
                print(f"  🛑 Early stopping at hop {hop+1}: Low confidence ({best_hop_score:.3f} < {confidence_threshold})")
                break
            
            # 2. Plateau detection: Stop if score hasn't improved significantly
            if len(best_chain[2]) >= plateau_patience:
                recent_scores = best_chain[2][-plateau_patience:]
                if len(recent_scores) >= 2:
                    max_recent = max(recent_scores)
                    min_recent = min(recent_scores)
                    if max_recent - min_recent < score_improvement_threshold:
                        print(f"  🛑 Early stopping at hop {hop+1}: Score plateau detected")
                        break
            
            # 3. Length-based stopping: For supporting facts, 2-3 paragraphs is often enough
            if len(best_chain[0]) >= 2 and best_hop_score < confidence_threshold * 1.2:
                print(f"  🛑 Early stopping at hop {hop+1}: Sufficient length with declining confidence")
                break
    
    # Return the best reasoning chain
    if active_chains:
        best_chain = active_chains[0][0]
        print(f"  🎯 Selected chain length: {len(best_chain)} (max_hops={max_hops})")
        return best_chain
    else:
        print(f"  ⚠️ No valid chains found")
        return []


def infer_batch(graphs: List[Any], model_tuple: Tuple, device: torch.device,
                config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Run inference on a batch of graphs with improved configuration.
    
    Args:
        graphs: List of PyTorch Geometric graphs
        model_tuple: (encoder, scorer) models  
        device: Torch device
        config: Configuration dictionary
        
    Returns:
        List of inference results
    """
    results = []
    
    # Extract inference config with new adaptive parameters
    inference_config = config.get('inference', {})
    max_hops = inference_config.get('max_hops', 3)  # Reduced default
    chain_width = inference_config.get('chain_width', 3)  # Reduced for focus
    adaptive_stopping = inference_config.get('adaptive_stopping', True)
    confidence_threshold = inference_config.get('confidence_threshold', 0.5)
    plateau_patience = inference_config.get('plateau_patience', 2)
    score_improvement_threshold = inference_config.get('score_improvement_threshold', 0.05)
    
    print(f"🔧 Inference config: max_hops={max_hops}, adaptive={adaptive_stopping}, confidence_threshold={confidence_threshold}")
    
    for i, graph in enumerate(graphs):
        try:
            chain = infer_frontier_chain(
                graph, model_tuple, 
                max_hops=max_hops,
                chain_width=chain_width,
                adaptive_stopping=adaptive_stopping,
                confidence_threshold=confidence_threshold,
                plateau_patience=plateau_patience,
                score_improvement_threshold=score_improvement_threshold
            )
            
            results.append({
                'pred_chain': chain,
                'graph_idx': i,
                'chain_length': len(chain)
            })
            
        except Exception as e:
            print(f"Error in inference for graph {i}: {e}")
            results.append({
                'pred_chain': [],
                'graph_idx': i, 
                'chain_length': 0
            })
    
    return results 