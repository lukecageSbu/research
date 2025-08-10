"""
Graph pruning utilities for top-k neighbors and threshold-based pruning.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional


def top_k_pruning(similarity_matrix: torch.Tensor, k: int, 
                  keep_self_loops: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prune graph by keeping only top-k most similar neighbors per node.
    
    Args:
        similarity_matrix: [n, n] matrix of similarities
        k: Number of neighbors to keep per node
        keep_self_loops: Whether to keep self-connections
        
    Returns:
        edge_index: [2, num_edges] edge connectivity
        edge_weights: [num_edges] edge weights (similarities)
    """
    n = similarity_matrix.size(0)
    
    # Get top-k neighbors for each node
    if keep_self_loops:
        # Don't exclude diagonal when finding top-k
        topk_values, topk_indices = torch.topk(similarity_matrix, min(k, n), dim=1)
    else:
        # Exclude diagonal by setting to -inf, then restore after topk
        sim_no_diag = similarity_matrix.clone()
        sim_no_diag.fill_diagonal_(-float('inf'))
        topk_values, topk_indices = torch.topk(sim_no_diag, min(k, n-1), dim=1)
    
    # Build edge list
    row_indices = []
    col_indices = []
    edge_weights = []
    
    for i in range(n):
        for j in range(topk_indices.size(1)):
            neighbor = topk_indices[i, j].item()
            weight = topk_values[i, j].item()
            
            # Skip invalid neighbors (can happen if k > actual neighbors)
            if weight == -float('inf'):
                continue
                
            row_indices.append(i)
            col_indices.append(neighbor)
            edge_weights.append(weight)
    
    # Convert to tensors
    edge_index = torch.tensor([row_indices, col_indices], dtype=torch.long)
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    return edge_index, edge_weights


def threshold_pruning(similarity_matrix: torch.Tensor, threshold: float,
                     min_degree: int = 1, keep_self_loops: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prune graph by keeping edges above similarity threshold with min degree fallback.
    
    Args:
        similarity_matrix: [n, n] matrix of similarities
        threshold: Minimum similarity to keep edge
        min_degree: Minimum degree per node (fallback if threshold too high)
        keep_self_loops: Whether to keep self-connections
        
    Returns:
        edge_index: [2, num_edges] edge connectivity  
        edge_weights: [num_edges] edge weights (similarities)
    """
    n = similarity_matrix.size(0)
    
    # Find edges above threshold
    if keep_self_loops:
        mask = similarity_matrix >= threshold
    else:
        mask = (similarity_matrix >= threshold) & (~torch.eye(n, dtype=torch.bool))
    
    # Check if any node has degree < min_degree
    degrees = mask.sum(dim=1)
    low_degree_nodes = (degrees < min_degree).nonzero(as_tuple=True)[0]
    
    # For low-degree nodes, add top neighbors to reach min_degree
    for node in low_degree_nodes:
        current_degree = degrees[node].item()
        needed = min_degree - current_degree
        
        if needed > 0:
            # Get similarities for this node
            node_sims = similarity_matrix[node]
            
            # Exclude already connected nodes
            node_sims_masked = node_sims.clone()
            node_sims_masked[mask[node]] = -float('inf')
            
            if not keep_self_loops:
                node_sims_masked[node] = -float('inf')
            
            # Get top additional neighbors
            if (node_sims_masked > -float('inf')).sum() > 0:
                _, top_indices = torch.topk(node_sims_masked, min(needed, n))
                
                for idx in top_indices:
                    if node_sims_masked[idx] > -float('inf'):
                        mask[node, idx] = True
    
    # Build edge list from mask
    row_indices, col_indices = mask.nonzero(as_tuple=True)
    edge_weights = similarity_matrix[row_indices, col_indices]
    
    edge_index = torch.stack([row_indices, col_indices], dim=0)
    
    return edge_index, edge_weights


def symmetrize_edges(edge_index: torch.Tensor, edge_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetrize edges by adding reverse edges with the same weights.
    
    Args:
        edge_index: [2, num_edges] edge connectivity
        edge_weights: [num_edges] edge weights
        
    Returns:
        sym_edge_index: [2, num_symmetric_edges] symmetrized edge connectivity
        sym_edge_weights: [num_symmetric_edges] symmetrized edge weights
    """
    # Create reverse edges
    reverse_edge_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
    
    # Combine original and reverse edges
    all_edges = torch.cat([edge_index, reverse_edge_index], dim=1)
    all_weights = torch.cat([edge_weights, edge_weights], dim=0)
    
    # Remove duplicates by converting to set of tuples
    edges_set = set()
    unique_edges = []
    unique_weights = []
    
    for i in range(all_edges.size(1)):
        edge = (all_edges[0, i].item(), all_edges[1, i].item())
        if edge not in edges_set:
            edges_set.add(edge)
            unique_edges.append([all_edges[0, i], all_edges[1, i]])
            unique_weights.append(all_weights[i])
    
    if unique_edges:
        sym_edge_index = torch.tensor(unique_edges).t()
        sym_edge_weights = torch.tensor(unique_weights)
    else:
        sym_edge_index = torch.empty((2, 0), dtype=torch.long)
        sym_edge_weights = torch.empty(0, dtype=torch.float)
    
    return sym_edge_index, sym_edge_weights


def prune_graph(similarity_matrix: torch.Tensor, strategy: str = 'top_k',
                k: Optional[int] = None, threshold: Optional[float] = None,
                min_degree: int = 1, keep_self_loops: bool = True,
                symmetrize: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified graph pruning function supporting multiple strategies.
    
    Args:
        similarity_matrix: [n, n] matrix of similarities
        strategy: 'top_k' or 'threshold'
        k: Number of neighbors for top_k strategy
        threshold: Similarity threshold for threshold strategy
        min_degree: Minimum degree per node (for threshold strategy)
        keep_self_loops: Whether to keep self-connections
        symmetrize: Whether to symmetrize the resulting edges
        
    Returns:
        edge_index: [2, num_edges] edge connectivity
        edge_weights: [num_edges] edge weights
    """
    if strategy == 'top_k':
        if k is None:
            raise ValueError("k must be specified for top_k strategy")
        edge_index, edge_weights = top_k_pruning(similarity_matrix, k, keep_self_loops)
    elif strategy == 'threshold':
        if threshold is None:
            raise ValueError("threshold must be specified for threshold strategy")
        edge_index, edge_weights = threshold_pruning(similarity_matrix, threshold, min_degree, keep_self_loops)
    else:
        raise ValueError(f"Unknown pruning strategy: {strategy}")
    
    if symmetrize:
        edge_index, edge_weights = symmetrize_edges(edge_index, edge_weights)
    
    return edge_index, edge_weights 