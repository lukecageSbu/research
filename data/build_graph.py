"""
Graph construction module for multi-hop reasoning.
Builds graphs from HotpotQA examples with node and edge features.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from torch_geometric.data import Data
from sklearn.metrics.pairwise import cosine_similarity
import sys
import os
import re

# Add utils to path for importing helper functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.entities import compute_overlap_features
from utils.prune import prune_graph


def compute_similarities(emb_q: np.ndarray, emb_p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine similarities between question-paragraphs and paragraph-paragraph.
    
    Args:
        emb_q: Question embedding [d_txt]
        emb_p: Paragraph embeddings [n, d_txt]
        
    Returns:
        sim_qp: Question-paragraph similarities [n]
        sim_pp: Paragraph-paragraph similarities [n, n]
    """
    # Question-paragraph similarities
    sim_qp = cosine_similarity(emb_q.reshape(1, -1), emb_p).flatten()
    
    # Paragraph-paragraph similarities
    sim_pp = cosine_similarity(emb_p, emb_p)
    
    return sim_qp, sim_pp



def build_node_features(emb_q: np.ndarray, emb_p: np.ndarray, sim_qp: np.ndarray, 
                       overlap_qp: np.ndarray, use_question_node: bool = True) -> torch.Tensor:
    """
    Build node features following the plan template.
    
    Args:
        emb_q: Question embedding [d_txt]
        emb_p: Paragraph embeddings [n, d_txt]
        sim_qp: Question-paragraph similarities [n]
        overlap_qp: Question-paragraph overlaps [n]
        use_question_node: Whether to include question node
        
    Returns:
        x: Node features [n+1, d_node] or [n, d_node]
    """
    n, d_txt = emb_p.shape
    node_features = []
    
    # Paragraph node features: [emb_p[i] || sim_qp[i] || overlap_qp[i]]
    for i in range(n):
        para_feat = np.concatenate([
            emb_p[i],           # Paragraph embedding
            [sim_qp[i]],        # Q-P similarity
            [overlap_qp[i]]     # Q-P overlap
        ])
        node_features.append(para_feat)
    
    # Optional question node feature: [emb_q || 1 || 1]
    if use_question_node:
        question_feat = np.concatenate([
            emb_q,              # Question embedding
            [1.0],              # Constant 1
            [1.0]               # Constant 1
        ])
        node_features.append(question_feat)
    
    return torch.tensor(np.array(node_features), dtype=torch.float)


def build_edges_and_features(sim_pp: np.ndarray, overlap_pp: np.ndarray, sim_qp: np.ndarray,
                            cfg: Dict[str, Any], use_question_node: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build edge connectivity and features following the plan template.
    
    Args:
        sim_pp: Paragraph-paragraph similarities [n, n]
        overlap_pp: Paragraph-paragraph overlaps [n, n]
        sim_qp: Question-paragraph similarities [n]
        cfg: Configuration with graph construction parameters
        use_question_node: Whether to include question node
        
    Returns:
        edge_index: Edge connectivity [2, num_edges]
        edge_attr: Edge features [num_edges, d_edge]
    """
    n = sim_pp.shape[0]
    
    # Get graph construction parameters
    strategy = cfg.get('pruning_strategy', 'top_k')
    top_k_neighbors = cfg.get('top_k_neighbors', 10)
    threshold = cfg.get('similarity_threshold', 0.7)
    keep_self_loops = cfg.get('keep_self_loops', True)
    top_k_question = cfg.get('top_k_question', 6)
    
    # Build paragraph-paragraph edges using pruning strategy
    sim_pp_tensor = torch.tensor(sim_pp, dtype=torch.float)
    pp_edge_index, pp_edge_weights = prune_graph(
        similarity_matrix=sim_pp_tensor,
        strategy=strategy,
        k=top_k_neighbors,
        threshold=threshold,
        keep_self_loops=keep_self_loops,
        symmetrize=True
    )
    
    # Build edge features for paragraph-paragraph edges
    pp_edge_features = []
    for i in range(pp_edge_index.size(1)):
        src, dst = pp_edge_index[0, i].item(), pp_edge_index[1, i].item()
        edge_feat = [
            sim_pp[src, dst],       # sim_pp[i,j]
            overlap_pp[src, dst],   # overlap_pp[i,j]
            sim_qp[src],            # sim_qp[i]
            sim_qp[dst]             # sim_qp[j]
        ]
        pp_edge_features.append(edge_feat)
    
    edge_index_list = [pp_edge_index]
    edge_features_list = pp_edge_features
    
    # Add question node edges if enabled
    if use_question_node:
        q_idx = n  # Question node index
        
        # Add q_idx -> top_kq paragraph edges by sim_qp
        top_k_indices = np.argsort(sim_qp)[::-1][:top_k_question]
        
        for para_idx in top_k_indices:
            # Question -> Paragraph edge
            qp_edge = torch.tensor([[q_idx], [para_idx]], dtype=torch.long)
            edge_index_list.append(qp_edge)
            
            # Edge features: [0, 0, sim_qp[para_idx], sim_qp[para_idx]]
            # (no paragraph-paragraph similarity for Q->P edges)
            qp_edge_feat = [
                0.0,                    # No sim_pp for Q->P edge
                0.0,                    # No overlap_pp for Q->P edge
                sim_qp[para_idx],       # sim_qp[i] (source is Q)
                sim_qp[para_idx]        # sim_qp[j] (destination para)
            ]
            edge_features_list.append(qp_edge_feat)
            
            # Paragraph -> Question edge (symmetrize)
            pq_edge = torch.tensor([[para_idx], [q_idx]], dtype=torch.long)
            edge_index_list.append(pq_edge)
            edge_features_list.append(qp_edge_feat)  # Same features
    
    # Combine all edges
    if len(edge_index_list) > 1:
        edge_index = torch.cat(edge_index_list, dim=1)
    else:
        edge_index = edge_index_list[0]
    
    edge_attr = torch.tensor(edge_features_list, dtype=torch.float)
    
    return edge_index, edge_attr


def build_labels_and_masks(support_indices: List[int], n_paragraphs: int, 
                          use_question_node: bool = True) -> Tuple[torch.Tensor, torch.Tensor, Optional[int]]:
    """
    Build node labels and masks.
    
    Args:
        support_indices: List of supporting paragraph indices
        n_paragraphs: Number of paragraphs
        use_question_node: Whether question node is included
        
    Returns:
        y: Node labels [n] (0/1 per paragraph)
        is_para_mask: Boolean mask [n+1] or [n]
        q_idx: Question node index (or None)
    """
    # Node labels: 0/1 per paragraph
    y = torch.zeros(n_paragraphs, dtype=torch.long)
    for idx in support_indices:
        if 0 <= idx < n_paragraphs:
            y[idx] = 1
    
    # Paragraph mask: True for paragraphs, False for question node
    if use_question_node:
        is_para_mask = torch.cat([
            torch.ones(n_paragraphs, dtype=torch.bool),   # Paragraphs
            torch.zeros(1, dtype=torch.bool)              # Question node
        ])
        q_idx = n_paragraphs
    else:
        is_para_mask = torch.ones(n_paragraphs, dtype=torch.bool)
        q_idx = None
    
    return y, is_para_mask, q_idx





def build_graph(example: Dict[str, Any], cfg: Dict[str, Any]) -> Data:
    """
    Optimized graph building with pre-computed overlap features.
    
    Args:
        example: Dictionary with pre-computed cached_overlaps
        cfg: Configuration dictionary
        
    Returns:
        torch_geometric.data.Data object
    """
    # Extract data from example
    question = example['question']
    paragraphs = example['paragraphs']
    emb_q = example['emb_q']  # [d_txt]
    emb_p = example['emb_p']  # [n, d_txt]
    support_indices = example.get('support_indices', [])
    cached_overlaps = example.get('cached_overlaps', None)
    
    # Configuration
    use_question_node = cfg.get('use_question_node', True)
    
    n_paragraphs = len(paragraphs)
    
    # 1. Compute similarities (fast)
    sim_qp, sim_pp = compute_similarities(emb_q, emb_p)
    
    # 2. Use cached overlap features or compute if needed
    if cached_overlaps is not None:
        overlap_qp = np.array(cached_overlaps['qp_entity_overlap'])
        overlap_pp = np.array(cached_overlaps['pp_entity_overlap'])
    else:
        # Fallback to individual computation
        overlap_features = compute_overlap_features(question, paragraphs, cfg.get('entity_method', 'rules'))
        overlap_qp = np.array(overlap_features['qp_entity_overlap'])
        overlap_pp = np.array(overlap_features['pp_entity_overlap'])
    
    # 3. Build node features: [emb_p[i] || sim_qp[i] || overlap_qp[i]]
    x = build_node_features(emb_q, emb_p, sim_qp, overlap_qp, use_question_node)
    
    # 4. Build edges and edge features
    edge_index, edge_attr = build_edges_and_features(
        sim_pp, overlap_pp, sim_qp, cfg, use_question_node
    )
    
    # 5. Build labels and masks
    y, is_para_mask, q_idx = build_labels_and_masks(
        support_indices, n_paragraphs, use_question_node
    )
    
    # Create PyTorch Geometric Data object
    graph_data = Data(
        x=x,                    # Node features
        edge_index=edge_index,  # Edge connectivity
        edge_attr=edge_attr,    # Edge features
        y=y,                    # Node labels (paragraph-level)
        is_para_mask=is_para_mask,  # Paragraph mask
        q_idx=q_idx             # Question node index
    )
    
    # Add metadata
    graph_data.example_id = example.get('id', '')
    graph_data.question = question
    graph_data.num_paragraphs = n_paragraphs
    graph_data.support_indices = support_indices
    
    return graph_data


def build_graphs_from_embeddings(question_embeddings: np.ndarray, 
                                paragraph_embeddings: List[np.ndarray],
                                metadata: List[Dict[str, Any]], 
                                cfg: Dict[str, Any]) -> List[Data]:
    """
    Optimized batch graph building from precomputed embeddings and metadata.
    
    Args:
        question_embeddings: [num_questions, d_txt] question embeddings
        paragraph_embeddings: List of [n_paragraphs, d_txt] paragraph embeddings
        metadata: List of metadata dictionaries
        cfg: Configuration dictionary
        
    Returns:
        graphs: List of torch_geometric.data.Data objects
    """
    from tqdm import tqdm
    import time
    
    graphs = []
    
    print(f"Building {len(metadata)} graphs from embeddings...")
    
    # OPTIMIZATION 1: Batch precompute overlap features for all texts
    print("🚀 Precomputing overlap features in batch...")
    start_time = time.time()
    
    # Extract all unique texts for batch processing
    all_questions = [meta['question'] for meta in metadata]
    all_paragraphs = []
    text_to_overlap = {}  # Cache overlap computations
    
    # Batch compute entity overlaps (much faster than individual)
    entity_method = cfg.get('entity_method', 'rules')
    
    # Use optimized overlap computation
    from utils.entities import batch_compute_overlap_features
    overlap_cache = batch_compute_overlap_features(all_questions, 
                                                 [meta['paragraphs'] for meta in metadata],
                                                 entity_method)
    
    overlap_time = time.time() - start_time
    print(f"   ✅ Overlap computation completed in {overlap_time:.1f}s")
    
    # OPTIMIZATION 2: Build graphs with cached overlaps
    start_time = time.time()
    
    for i, meta in enumerate(tqdm(metadata, desc="Building graphs")):
        # Prepare example dictionary with cached overlaps
        example = {
            'id': meta['id'],
            'question': meta['question'],
            'paragraphs': meta['paragraphs'],
            'emb_q': question_embeddings[i],
            'emb_p': paragraph_embeddings[i],
            'support_indices': meta['support_indices'],
            'answer': meta.get('answer', ''),
            'type': meta.get('type', ''),
            'level': meta.get('level', ''),
            # Pre-computed overlaps
            'cached_overlaps': overlap_cache[i] if i < len(overlap_cache) else None
        }
        
        # Build graph with cached data
        graph = build_graph(example, cfg)
        graphs.append(graph)
    
    graph_time = time.time() - start_time
    total_time = overlap_time + graph_time
    
    print(f"✅ Successfully built {len(graphs)} graphs in {total_time:.1f}s")
    print(f"   📊 Average: {total_time/len(graphs)*1000:.1f}ms per graph")
    return graphs 