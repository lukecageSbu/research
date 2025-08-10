"""
Prefix-conditioned scorer for multi-hop reasoning and frontier-chain search.
Implements context-aware scoring for next paragraph selection in reasoning chains.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Union, Tuple, Dict
import math


class ContextPooling(nn.Module):
    """
    Context pooling module that can use mean pooling or attention pooling.
    """
    
    def __init__(self, hidden_dim: int, pooling_type: str = 'mean', 
                 num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize context pooling.
        
        Args:
            hidden_dim: Hidden dimension
            pooling_type: 'mean', 'max', 'sum', 'attention'
            num_heads: Number of attention heads (for attention pooling)
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.pooling_type = pooling_type
        self.num_heads = num_heads
        
        if pooling_type == 'attention':
            # Multi-head attention for context pooling
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            # Learnable query for context attention
            self.context_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        elif pooling_type not in ['mean', 'max', 'sum']:
            raise ValueError(f"Unknown pooling_type: {pooling_type}")
    
    def forward(self, context_nodes: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Pool context nodes into a single context vector.
        
        Args:
            context_nodes: Context node representations [num_context, hidden]
            mask: Optional mask for valid context nodes [num_context]
            
        Returns:
            context: Pooled context vector [hidden]
        """
        if context_nodes.size(0) == 0:
            return torch.zeros(self.hidden_dim, device=context_nodes.device, dtype=context_nodes.dtype)
        
        if self.pooling_type == 'mean':
            if mask is not None:
                # Masked mean pooling
                masked_nodes = context_nodes * mask.unsqueeze(-1)
                context = masked_nodes.sum(dim=0) / mask.sum().clamp(min=1)
            else:
                context = torch.mean(context_nodes, dim=0)
                
        elif self.pooling_type == 'max':
            context, _ = torch.max(context_nodes, dim=0)
            
        elif self.pooling_type == 'sum':
            context = torch.sum(context_nodes, dim=0)
            
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            context_nodes_batch = context_nodes.unsqueeze(0)  # [1, num_context, hidden]
            query = self.context_query.expand(1, -1, -1)  # [1, 1, hidden]
            
            # Apply attention
            attn_output, _ = self.attention(
                query, context_nodes_batch, context_nodes_batch,
                key_padding_mask=~mask if mask is not None else None
            )
            context = attn_output.squeeze(0).squeeze(0)  # [hidden]
        
        return context


class NodeScorer(nn.Module):
    """
    MLP-based node scorer that takes node and context representations.
    """
    
    def __init__(self, hidden_dim: int, num_layers: int = 2, dropout: float = 0.1,
                 activation: str = 'relu', use_layer_norm: bool = True):
        """
        Initialize node scorer.
        
        Args:
            hidden_dim: Hidden dimension
            num_layers: Number of MLP layers
            dropout: Dropout probability
            activation: Activation function
            use_layer_norm: Whether to use layer normalization
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Build MLP layers
        layers = []
        input_dim = hidden_dim * 2  # [node_repr || context]
        
        for i in range(num_layers):
            output_dim = hidden_dim if i < num_layers - 1 else 1
            
            layers.append(nn.Linear(input_dim, output_dim))
            
            if i < num_layers - 1:  # Not the last layer
                if use_layer_norm:
                    layers.append(nn.LayerNorm(output_dim))
                
                if activation == 'relu':
                    layers.append(nn.ReLU())
                elif activation == 'gelu':
                    layers.append(nn.GELU())
                else:
                    raise ValueError(f"Unknown activation: {activation}")
                
                layers.append(nn.Dropout(dropout))
            
            input_dim = output_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, node_reprs: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Score nodes given context.
        
        Args:
            node_reprs: Node representations [num_candidates, hidden]
            context: Context vector [hidden]
            
        Returns:
            scores: Compatibility scores [num_candidates]
        """
        batch_size = node_reprs.size(0)
        
        # Expand context to match candidates
        context_expanded = context.unsqueeze(0).expand(batch_size, -1)  # [num_candidates, hidden]
        
        # Concatenate node representations with context
        combined = torch.cat([node_reprs, context_expanded], dim=1)  # [num_candidates, 2*hidden]
        
        # Score with MLP
        scores = self.mlp(combined).squeeze(-1)  # [num_candidates]
        
        return scores


class ConditioningGNN(nn.Module):
    """
    Optional conditioning GNN that updates representations based on current reasoning path.
    This implements the "path token" concept from the plan.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize conditioning GNN.
        
        Args:
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Path token - learnable embedding that gets updated with context
        self.path_token = nn.Parameter(torch.randn(hidden_dim))
        
        # Attention to update path token with context
        self.path_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Graph attention to propagate path information
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, H: torch.Tensor, edge_index: torch.Tensor,
                context_ids: List[int]) -> torch.Tensor:
        """
        Apply conditioning GNN to update representations.
        
        Args:
            H: Node representations [num_nodes, hidden]
            edge_index: Edge connectivity [2, num_edges]
            context_ids: Indices of nodes in current context
            
        Returns:
            H_conditioned: Updated node representations [num_nodes, hidden]
        """
        batch_size = H.size(0)
        
        # Get context representations
        if context_ids:
            context_reprs = H[context_ids]  # [num_context, hidden]
        else:
            context_reprs = self.path_token.unsqueeze(0)  # [1, hidden]
        
        # Update path token with context
        path_token_batch = self.path_token.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        context_batch = context_reprs.unsqueeze(0)  # [1, num_context, hidden]
        
        updated_path, _ = self.path_attention(
            path_token_batch, context_batch, context_batch
        )
        updated_path = self.layer_norm1(updated_path + path_token_batch)
        updated_path = updated_path.squeeze(0).squeeze(0)  # [hidden]
        
        # Propagate path information to all nodes
        H_batch = H.unsqueeze(0)  # [1, num_nodes, hidden]
        path_batch = updated_path.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden]
        path_expanded = path_batch.expand(1, batch_size, -1)  # [1, num_nodes, hidden]
        
        H_updated, _ = self.graph_attention(H_batch, path_expanded, path_expanded)
        H_conditioned = self.layer_norm2(H_updated + H_batch)
        H_conditioned = self.dropout(H_conditioned)
        
        return H_conditioned.squeeze(0)  # [num_nodes, hidden]


class PrefixScorer(nn.Module):
    """
    Enhanced prefix-conditioned scorer for multi-hop reasoning.
    Supports multiple context pooling strategies and optional conditioning GNN.
    """
    
    def __init__(self, hidden_dim: int, context_pool: str = 'mean', 
                 scorer_layers: int = 2, num_heads: int = 4, dropout: float = 0.1,
                 activation: str = 'relu'):
        """
        Initialize the prefix scorer.
        
        Args:
            hidden_dim: Hidden dimension of node representations
            context_pool: Context pooling method ('mean', 'max', 'sum', 'attention')
            scorer_layers: Number of layers in scoring MLP
            num_heads: Number of attention heads
            dropout: Dropout probability
            activation: Activation function for MLP
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.context_pool = context_pool
        
        # Context pooling module
        self.context_pooling = ContextPooling(
            hidden_dim=hidden_dim,
            pooling_type=context_pool,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Node scorer MLP
        self.node_scorer = NodeScorer(
            hidden_dim=hidden_dim,
            num_layers=scorer_layers,
            dropout=dropout,
            activation=activation
        )
        
        # Optional conditioning GNN
        # Conditioning GNN - CORE component for multi-hop reasoning
        self.conditioning_gnn = ConditioningGNN(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout
            )
    
    def context(self, H: torch.Tensor, q_idx: Optional[int], 
                prefix_ids: List[int]) -> torch.Tensor:
        """
        Compute context vector from question and prefix nodes.
        
        Args:
            H: Node representations [N, hidden]
            q_idx: Index of question node (None if no question node)
            prefix_ids: List of node indices in current prefix
            
        Returns:
            c_t: Context vector [hidden]
        """
        # Collect indices for context: {question} ∪ prefix_nodes
        context_indices = []
        
        if q_idx is not None:
            context_indices.append(q_idx)
            
        context_indices.extend(prefix_ids)
        
        if not context_indices:
            # Empty context - return zero vector
            return torch.zeros(self.hidden_dim, device=H.device, dtype=H.dtype)
        
        # Get context node representations
        context_nodes = H[context_indices]  # [num_context, hidden]
        
        # Pool context nodes
        c_t = self.context_pooling(context_nodes)
        
        return c_t
    
    def forward(self, H: torch.Tensor, q_idx: Optional[int], prefix_ids: List[int], 
                candidate_ids: torch.Tensor, edge_index: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Score candidate nodes for the next step in the reasoning chain.
        
        Args:
            H: Node representations [N, hidden]
            q_idx: Index of question node (None if no question node)
            prefix_ids: List of node indices in current prefix
            candidate_ids: Tensor of candidate node indices [M]
            edge_index: Edge connectivity [2, num_edges] (for conditioning GNN)
            
        Returns:
            logits: Compatibility scores for candidates [M]
        """
        # Apply conditioning GNN if enabled
        # Apply conditioning GNN (always enabled)
        if edge_index is not None:
            context_ids = ([q_idx] if q_idx is not None else []) + prefix_ids
            H = self.conditioning_gnn(H, edge_index, context_ids)
        
        # Get context vector
        c_t = self.context(H, q_idx, prefix_ids)  # [hidden]
        
        # Get candidate representations
        candidate_reprs = H[candidate_ids]  # [M, hidden]
        
        # Score candidates with context
        logits = self.node_scorer(candidate_reprs, c_t)  # [M]
        
        return logits
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_top_k_candidates(self, H: torch.Tensor, q_idx: Optional[int], 
                            prefix_ids: List[int], candidate_ids: torch.Tensor,
                            k: int = 5, temperature: float = 1.0,
                            edge_index: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get top-k frontier candidates with their scores for chain expansion.
        
        Args:
            H: Node representations [N, hidden]
            q_idx: Index of question node
            prefix_ids: List of node indices in current reasoning chain prefix
            candidate_ids: Tensor of frontier candidate node indices [M]
            k: Number of top frontier candidates to return
            temperature: Temperature for score scaling
            edge_index: Edge connectivity (for conditioning GNN)
            
        Returns:
            top_k_candidates: Top-k frontier candidate indices [k]
            top_k_scores: Top-k frontier scores [k]
        """
        # Get scores
        logits = self.forward(H, q_idx, prefix_ids, candidate_ids, edge_index)
        
        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / temperature
        
        # Get top-k frontier candidates
        top_k_scores, top_k_indices = torch.topk(logits, min(k, len(candidate_ids)))
        top_k_candidates = candidate_ids[top_k_indices]
        
        return top_k_candidates, top_k_scores
    
    def frontier_chain_score(self, H: torch.Tensor, q_idx: Optional[int], 
                             prefix_ids: List[int], candidate_ids: torch.Tensor,
                             top_k_expand: int = 5, edge_index: Optional[torch.Tensor] = None) -> List[Tuple[int, float]]:
        """
        Score candidates for frontier-chain search inference.
        
        Args:
            H: Node representations [N, hidden]
            q_idx: Index of question node
            prefix_ids: List of node indices in current prefix
            candidate_ids: Tensor of candidate node indices [M]
            top_k_expand: Number of top candidates for frontier expansion
            edge_index: Edge connectivity (for conditioning GNN)
            
        Returns:
            scored_candidates: List of (candidate_id, score) tuples for frontier expansion
        """
        # Get top candidates for frontier expansion
        top_candidates, top_scores = self.get_top_k_candidates(
            H, q_idx, prefix_ids, candidate_ids, k=top_k_expand, edge_index=edge_index
        )
        
        # Convert to list of tuples
        scored_candidates = [
            (candidate.item(), score.item()) 
            for candidate, score in zip(top_candidates, top_scores)
        ]
        
        return scored_candidates


def create_hotpot_prefix_scorer(config: dict) -> PrefixScorer:
    """
    Factory function to create a PrefixScorer optimized for HotpotQA frontier-chain search.
    
    Args:
        config: Configuration dictionary with scorer and model parameters
        
    Returns:
        scorer: Configured PrefixScorer instance for multi-hop reasoning
        
    Example:
        config = {
            'scorer': {
                'type': 'mlp',
                'context_pool': 'mean',
                'conditioning_layer': False
            },
            'model': {
                'hidden_dim': 256,
                'heads': 4,
                'dropout': 0.15
            }
        }
        scorer = create_hotpot_prefix_scorer(config)
    """
    scorer_config = config.get('scorer', {})
    model_config = config.get('model', {})
    
    # Extract parameters
    hidden_dim = model_config.get('hidden_dim', 256)
    context_pool = scorer_config.get('context_pool', 'mean')
    scorer_layers = scorer_config.get('layers', 2)
    num_heads = model_config.get('heads', 4)
    dropout = model_config.get('dropout', 0.15)
    activation = scorer_config.get('activation', 'relu')
    
    return PrefixScorer(
        hidden_dim=hidden_dim,
        context_pool=context_pool,
        scorer_layers=scorer_layers,
        num_heads=num_heads,
        dropout=dropout,
        activation=activation
    ) 