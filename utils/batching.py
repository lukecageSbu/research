"""
Batching utilities for variable-size graphs in multi-hop reasoning.
Handles HotpotQA data with PyTorch Geometric DataLoaders.
"""

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from typing import List, Dict, Any, Optional, Callable
import numpy as np


def create_dataloader(graphs: List[Data], batch_size: int = 16, shuffle: bool = True,
                     num_workers: int = 0, collate_fn: Optional[Callable] = None) -> DataLoader:
    """
    Create PyTorch Geometric DataLoader for variable-size graphs.
    
    Args:
        graphs: List of torch_geometric.data.Data objects
        batch_size: Number of graphs per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        collate_fn: Custom collation function (optional)
        
    Returns:
        dataloader: PyTorch Geometric DataLoader
    """
    return DataLoader(
        graphs,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        follow_batch=['x', 'edge_attr'],  # Track batching for node and edge features
        collate_fn=collate_fn
    )


def analyze_batch_statistics(dataloader: DataLoader) -> Dict[str, Any]:
    """
    Analyze batching statistics for debugging and optimization.
    
    Args:
        dataloader: PyTorch Geometric DataLoader
        
    Returns:
        stats: Dictionary with batching statistics
    """
    batch_sizes = []
    node_counts = []
    edge_counts = []
    
    for batch in dataloader:
        batch_sizes.append(batch.num_graphs)
        node_counts.append(batch.x.size(0))
        edge_counts.append(batch.edge_index.size(1))
    
    return {
        'num_batches': len(batch_sizes),
        'avg_graphs_per_batch': np.mean(batch_sizes),
        'avg_nodes_per_batch': np.mean(node_counts),
        'avg_edges_per_batch': np.mean(edge_counts),
        'max_nodes_per_batch': max(node_counts),
        'max_edges_per_batch': max(edge_counts),
        'min_nodes_per_batch': min(node_counts),
        'min_edges_per_batch': min(edge_counts),
        'node_count_distribution': node_counts,
        'edge_count_distribution': edge_counts
    }


def graph_level_pooling(x: torch.Tensor, batch: torch.Tensor, 
                       method: str = 'mean') -> torch.Tensor:
    """
    Perform graph-level pooling to get graph representations from node features.
    
    Args:
        x: Node features [total_nodes, feature_dim]
        batch: Batch assignment [total_nodes] indicating which graph each node belongs to
        method: Pooling method ('mean', 'max', 'sum', 'attention')
        
    Returns:
        graph_representations: [num_graphs, feature_dim]
    """
    from torch_scatter import scatter
    
    if method == 'mean':
        return scatter(x, batch, dim=0, reduce='mean')
    elif method == 'max':
        return scatter(x, batch, dim=0, reduce='max')
    elif method == 'sum':
        return scatter(x, batch, dim=0, reduce='add')
    else:
        raise ValueError(f"Unsupported pooling method: {method}")


def extract_graph_predictions(logits: torch.Tensor, batch: torch.Tensor, 
                             is_para_mask: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract per-graph predictions from batched node predictions.
    
    Args:
        logits: Node logits [total_nodes, num_classes]
        batch: Batch assignment [total_nodes]
        is_para_mask: Mask for paragraph nodes [total_nodes]
        
    Returns:
        graph_predictions: List of per-graph prediction tensors
    """
    predictions = []
    unique_graphs = torch.unique(batch)
    
    for graph_id in unique_graphs:
        # Get nodes belonging to this graph
        graph_mask = (batch == graph_id)
        
        # Also apply paragraph mask (exclude question nodes)
        para_mask_for_graph = graph_mask & is_para_mask
        
        # Extract predictions for paragraphs in this graph
        graph_logits = logits[para_mask_for_graph]
        predictions.append(graph_logits)
    
    return predictions


def extract_graph_labels(labels: torch.Tensor, batch: torch.Tensor, 
                        is_para_mask: torch.Tensor) -> List[torch.Tensor]:
    """
    Extract per-graph labels from batched labels.
    
    CRITICAL: Labels tensor only contains paragraph labels, not question node labels.
    The batch tensor and is_para_mask include question nodes, but labels don't.
    
    Args:
        labels: Paragraph labels [total_paragraphs] - NO question node labels
        batch: Batch assignment [total_nodes] - includes question nodes  
        is_para_mask: Mask for paragraph nodes [total_nodes] - includes question nodes
        
    Returns:
        graph_labels: List of per-graph label tensors
    """
    # Create mapping from paragraph positions to label positions
    paragraph_positions = torch.where(is_para_mask)[0]  # Indices of paragraph nodes
    
    if len(paragraph_positions) != len(labels):
        raise ValueError(f"Mismatch: {len(paragraph_positions)} paragraph positions but {len(labels)} labels")
    
    graph_labels = []
    unique_graphs = torch.unique(batch)
    
    label_offset = 0
    for graph_id in unique_graphs:
        # Get paragraph nodes for this graph
        graph_mask = (batch == graph_id)
        para_mask_for_graph = graph_mask & is_para_mask
        num_paragraphs = para_mask_for_graph.sum().item()
        
        # Extract labels for this graph's paragraphs
        graph_labels_tensor = labels[label_offset:label_offset + num_paragraphs]
        graph_labels.append(graph_labels_tensor)
        
        label_offset += num_paragraphs
    
    return graph_labels


def pad_graphs_to_fixed_size(graphs: List[Data], max_nodes: int, 
                            pad_value: float = 0.0) -> List[Data]:
    """
    Pad graphs to fixed size (alternative batching strategy).
    
    Args:
        graphs: List of Data objects
        max_nodes: Maximum number of nodes to pad to
        pad_value: Value to use for padding
        
    Returns:
        padded_graphs: List of padded Data objects with additional masks
    """
    padded_graphs = []
    
    for graph in graphs:
        n_nodes = graph.x.size(0)
        
        if n_nodes > max_nodes:
            raise ValueError(f"Graph has {n_nodes} nodes, exceeds max_nodes={max_nodes}")
        
        # Pad node features
        pad_size = max_nodes - n_nodes
        if pad_size > 0:
            # Pad with zeros
            padded_x = torch.cat([
                graph.x,
                torch.full((pad_size, graph.x.size(1)), pad_value)
            ], dim=0)
            
            # Create node mask (True for real nodes, False for padding)
            node_mask = torch.cat([
                torch.ones(n_nodes, dtype=torch.bool),
                torch.zeros(pad_size, dtype=torch.bool)
            ])
            
            # Pad labels
            padded_y = torch.cat([
                graph.y,
                torch.zeros(pad_size, dtype=graph.y.dtype)
            ])
            
            # Update is_para_mask if present
            if hasattr(graph, 'is_para_mask'):
                padded_is_para_mask = torch.cat([
                    graph.is_para_mask,
                    torch.zeros(pad_size, dtype=torch.bool)
                ])
            else:
                padded_is_para_mask = node_mask.clone()
            
            # Create new graph with padded features
            padded_graph = Data(
                x=padded_x,
                edge_index=graph.edge_index,  # Edges unchanged
                edge_attr=graph.edge_attr,
                y=padded_y,
                is_para_mask=padded_is_para_mask,
                node_mask=node_mask,  # New field to track real vs padding
                q_idx=graph.q_idx if hasattr(graph, 'q_idx') else None
            )
        else:
            # No padding needed
            padded_graph = graph
            padded_graph.node_mask = torch.ones(n_nodes, dtype=torch.bool)
        
        padded_graphs.append(padded_graph)
    
    return padded_graphs


def dynamic_batching_by_size(graphs: List[Data], batch_size: int = 16,
                           size_tolerance: int = 2) -> List[List[Data]]:
    """
    Group graphs by similar sizes for more efficient batching.
    
    Args:
        graphs: List of Data objects
        batch_size: Target batch size
        size_tolerance: Maximum difference in node count within a batch
        
    Returns:
        batches: List of graph batches grouped by similar sizes
    """
    # Sort graphs by number of nodes
    graphs_with_sizes = [(g, g.x.size(0)) for g in graphs]
    graphs_with_sizes.sort(key=lambda x: x[1])
    
    batches = []
    current_batch = []
    current_size = None
    
    for graph, size in graphs_with_sizes:
        # Start new batch if size difference too large or batch full
        if (current_size is not None and 
            abs(size - current_size) > size_tolerance) or len(current_batch) >= batch_size:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
        
        current_batch.append(graph)
        current_size = size
    
    # Add final batch
    if current_batch:
        batches.append(current_batch)
    
    return batches


def compute_batch_memory_usage(batch: Data) -> Dict[str, float]:
    """
    Estimate memory usage of a batch for optimization.
    
    Args:
        batch: Batched Data object
        
    Returns:
        memory_stats: Dictionary with memory usage estimates (in MB)
    """
    def tensor_memory_mb(tensor):
        return tensor.numel() * tensor.element_size() / (1024 * 1024)
    
    stats = {
        'node_features_mb': tensor_memory_mb(batch.x),
        'edge_index_mb': tensor_memory_mb(batch.edge_index),
        'edge_features_mb': tensor_memory_mb(batch.edge_attr),
        'labels_mb': tensor_memory_mb(batch.y),
        'batch_tensor_mb': tensor_memory_mb(batch.batch),
    }
    
    stats['total_mb'] = sum(stats.values())
    
    return stats


class HotpotQABatchingStrategy:
    """
    Specialized batching strategy for HotpotQA multi-hop reasoning.
    """
    
    def __init__(self, batch_size: int = 16, max_nodes_per_batch: int = 500,
                 prefer_similar_sizes: bool = True):
        """
        Initialize batching strategy.
        
        Args:
            batch_size: Number of graphs per batch
            max_nodes_per_batch: Maximum total nodes per batch (memory limit)
            prefer_similar_sizes: Whether to group similar-sized graphs
        """
        self.batch_size = batch_size
        self.max_nodes_per_batch = max_nodes_per_batch
        self.prefer_similar_sizes = prefer_similar_sizes
    
    def create_batches(self, graphs: List[Data]) -> List[List[Data]]:
        """
        Create optimal batches for HotpotQA graphs.
        
        Args:
            graphs: List of graph Data objects
            
        Returns:
            batches: List of graph batches
        """
        if self.prefer_similar_sizes:
            # Use dynamic batching by size
            batches = dynamic_batching_by_size(graphs, self.batch_size)
        else:
            # Simple sequential batching
            batches = [graphs[i:i+self.batch_size] 
                      for i in range(0, len(graphs), self.batch_size)]
        
        # Filter batches that exceed memory limits
        filtered_batches = []
        for batch in batches:
            total_nodes = sum(g.x.size(0) for g in batch)
            if total_nodes <= self.max_nodes_per_batch:
                filtered_batches.append(batch)
            else:
                # Split large batch
                filtered_batches.extend(self._split_large_batch(batch))
        
        return filtered_batches
    
    def _split_large_batch(self, batch: List[Data]) -> List[List[Data]]:
        """Split a batch that exceeds memory limits."""
        small_batches = []
        current_batch = []
        current_nodes = 0
        
        for graph in batch:
            graph_nodes = graph.x.size(0)
            
            if current_nodes + graph_nodes > self.max_nodes_per_batch:
                if current_batch:
                    small_batches.append(current_batch)
                current_batch = [graph]
                current_nodes = graph_nodes
            else:
                current_batch.append(graph)
                current_nodes += graph_nodes
        
        if current_batch:
            small_batches.append(current_batch)
        
        return small_batches 