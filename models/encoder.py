"""
Graph-aware encoder module starting with GraphSAGE, then upgrading to GATv2.
Phase 1: Simple GraphSAGE implementation for pipeline validation.
Phase 2: Advanced GATv2 with attention and edge features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, LayerNorm
from typing import Optional, Union
import math


class GraphSAGELayer(nn.Module):
    """
    Single GraphSAGE layer with residual connection and layer normalization.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1, activation: str = 'relu'):
        """
        Initialize GraphSAGE layer.
        
        Args:
            hidden_dim: Hidden dimension
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # GraphSAGE convolution
        self.sage_conv = SAGEConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            aggr='mean',  # Can be 'mean', 'max', 'add'
            normalize=True,
            bias=True
        )
        
        # Layer normalization
        self.layer_norm = LayerNorm(hidden_dim)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GraphSAGE layer.
        
        Args:
            x: Node features [N, hidden_dim]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            out: Updated node features [N, hidden_dim]
        """
        # Store residual
        residual = x
        
        # Apply GraphSAGE convolution
        sage_out = self.sage_conv(x, edge_index)
        
        # Apply activation and dropout
        sage_out = self.activation(sage_out)
        sage_out = self.dropout(sage_out)
        
        # Residual connection + layer norm
        out = self.layer_norm(sage_out + residual)
        
        return out


class SimpleGraphEncoder(nn.Module):
    """
    Simple GraphSAGE-based encoder for multi-hop reasoning.
    Phase 1 implementation to validate the entire pipeline.
    """
    
    def __init__(self, d_node: int, d_edge: int, hidden: int, layers: int, 
                 dropout: float = 0.1, activation: str = 'relu'):
        """
        Initialize the simple graph encoder.
        
        Args:
            d_node: Input node feature dimension (e.g., 386 for SentenceTransformer + features)
            d_edge: Input edge feature dimension (e.g., 4 - will be ignored for now)
            hidden: Hidden dimension for graph layers
            layers: Number of GraphSAGE layers (2-3 recommended)
            dropout: Dropout probability
            activation: Activation function ('relu', 'gelu')
        """
        super().__init__()
        
        self.d_node = d_node
        self.d_edge = d_edge  # Stored but not used in Phase 1
        self.hidden = hidden
        self.layers = layers
        self.dropout = dropout
        
        # Input projection to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(d_node, hidden),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GraphSAGE layers
        self.graph_layers = nn.ModuleList([
            GraphSAGELayer(
                hidden_dim=hidden,
                dropout=dropout,
                activation=activation
            ) for _ in range(layers)
        ])
        
        # Output layer normalization
        self.output_norm = LayerNorm(hidden)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights with Xavier uniform initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through the graph encoder.
        
        Args:
            x: Node features [N, d_node]
            edge_index: Edge connectivity [2, num_edges]
            edge_attr: Edge features [num_edges, d_edge] (ignored in Phase 1)
            
        Returns:
            H: Updated node representations [N, hidden]
        """
        # Input validation
        if x.size(-1) != self.d_node:
            raise ValueError(f"Expected node features of size {self.d_node}, got {x.size(-1)}")
        
        # Note: edge_attr is ignored in this simple implementation
        # We'll add edge feature integration in Phase 2 (GATv2)
        
        # Project input to hidden dimension
        h = self.input_projection(x)
        
        # Apply GraphSAGE layers
        for graph_layer in self.graph_layers:
            h = graph_layer(h, edge_index)
        
        # Final layer normalization
        h = self.output_norm(h)
        
        return h
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_layer_outputs(self, x: torch.Tensor, edge_index: torch.Tensor) -> list:
        """
        Get intermediate layer outputs for analysis.
        
        Args:
            x: Node features [N, d_node]
            edge_index: Edge connectivity [2, num_edges]
            
        Returns:
            layer_outputs: List of outputs from each layer
        """
        layer_outputs = []
        
        # Input projection
        h = self.input_projection(x)
        layer_outputs.append(h)
        
        # Apply each layer and store output
        for graph_layer in self.graph_layers:
            h = graph_layer(h, edge_index)
            layer_outputs.append(h)
        
        # Final normalization
        h = self.output_norm(h)
        layer_outputs.append(h)
        
        return layer_outputs


# Phase 2: Advanced GATv2 Implementation (for later)
class EdgeFeatureGATv2Conv(nn.Module):
    """
    GATv2 layer with edge feature integration for multi-hop reasoning.
    This will be used in Phase 2 after GraphSAGE validation.
    """
    
    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, 
                 edge_dim: Optional[int] = None, dropout: float = 0.0, 
                 add_self_loops: bool = False, bias: bool = True):
        """
        Initialize edge-aware GATv2 convolution.
        """
        super().__init__()
        
        # Import GATv2Conv here to avoid issues if torch_geometric not fully installed
        try:
            from torch_geometric.nn import GATv2Conv
        except ImportError:
            raise ImportError("torch_geometric not found. Install with: pip install torch-geometric")
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.edge_dim = edge_dim
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        
        # Core GATv2 layer
        self.gatv2_conv = GATv2Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            heads=heads,
            dropout=dropout,
            add_self_loops=add_self_loops,
            edge_dim=edge_dim,
            bias=bias
        )
        
        # Edge feature preprocessing (if provided)
        if edge_dim is not None:
            self.edge_encoder = nn.Sequential(
                nn.Linear(edge_dim, edge_dim * heads),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            self.edge_encoder = None
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None, 
                return_attention_weights: bool = False):
        """Forward pass with edge feature integration."""
        # Preprocess edge features if provided
        processed_edge_attr = None
        if edge_attr is not None and self.edge_encoder is not None:
            processed_edge_attr = self.edge_encoder(edge_attr)
        elif edge_attr is not None:
            processed_edge_attr = edge_attr
        
        # Apply GATv2 convolution
        return self.gatv2_conv(
            x, edge_index, edge_attr=processed_edge_attr,
            return_attention_weights=return_attention_weights
        )


class AdvancedGraphEncoder(nn.Module):
    """
    Advanced GATv2-based encoder with edge features and attention.
    Phase 2 implementation after GraphSAGE validation.
    """
    
    def __init__(self, d_node: int, d_edge: int, hidden: int, layers: int, 
                 heads: int, dropout: float = 0.1, activation: str = 'relu'):
        """
        Initialize the advanced graph encoder.
        This will be implemented in Phase 2.
        """
        super().__init__()
        
        self.d_node = d_node
        self.d_edge = d_edge
        self.hidden = hidden
        self.layers = layers
        self.heads = heads
        self.dropout = dropout
        
        # TODO: Implement in Phase 2
        # For now, raise an informative error
        raise NotImplementedError(
            "AdvancedGraphEncoder (GATv2) will be implemented in Phase 2. "
            "Use SimpleGraphEncoder (GraphSAGE) for Phase 1."
        )


# Main GraphEncoder class - switches between implementations
class GraphEncoder(nn.Module):
    """
    Main GraphEncoder class that switches between implementations.
    Phase 1: Uses SimpleGraphEncoder (GraphSAGE)
    Phase 2: Will use AdvancedGraphEncoder (GATv2)
    """
    
    def __init__(self, d_node: int, d_edge: int, hidden: int, layers: int, 
                 heads: int = 4, dropout: float = 0.1, activation: str = 'relu',
                 encoder_type: str = 'sage'):
        """
        Initialize the graph encoder.
        
        Args:
            d_node: Input node feature dimension
            d_edge: Input edge feature dimension
            hidden: Hidden dimension
            layers: Number of graph layers
            heads: Number of attention heads (used for GATv2 in Phase 2)
            dropout: Dropout probability
            activation: Activation function
            encoder_type: 'sage' for GraphSAGE (Phase 1), 'gatv2' for GATv2 (Phase 2)
        """
        super().__init__()
        
        self.encoder_type = encoder_type
        
        if encoder_type == 'sage':
            # Phase 1: Simple GraphSAGE implementation
            self.encoder = SimpleGraphEncoder(
                d_node=d_node,
                d_edge=d_edge,
                hidden=hidden,
                layers=layers,
                dropout=dropout,
                activation=activation
            )
        elif encoder_type == 'gatv2':
            # Phase 2: Advanced GATv2 implementation
            self.encoder = AdvancedGraphEncoder(
                d_node=d_node,
                d_edge=d_edge,
                hidden=hidden,
                layers=layers,
                heads=heads,
                dropout=dropout,
                activation=activation
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Use 'sage' or 'gatv2'.")
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, 
                edge_attr: Optional[torch.Tensor] = None,
                return_attention_weights: bool = False) -> Union[torch.Tensor, tuple]:
        """Forward pass through the encoder."""
        if self.encoder_type == 'sage':
            # GraphSAGE doesn't return attention weights
            output = self.encoder(x, edge_index, edge_attr)
            if return_attention_weights:
                return output, None
            return output
        else:
            # GATv2 can return attention weights
            return self.encoder(x, edge_index, edge_attr, return_attention_weights)
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters."""
        return self.encoder.count_parameters()
    
    def get_layer_outputs(self, x: torch.Tensor, edge_index: torch.Tensor, 
                         edge_attr: Optional[torch.Tensor] = None) -> list:
        """Get intermediate layer outputs for analysis."""
        if hasattr(self.encoder, 'get_layer_outputs'):
            if self.encoder_type == 'sage':
                return self.encoder.get_layer_outputs(x, edge_index)
            else:
                return self.encoder.get_layer_outputs(x, edge_index, edge_attr)
        else:
            raise NotImplementedError("get_layer_outputs not implemented for this encoder type")


def create_hotpot_graph_encoder(config: dict, phase: int = 1) -> GraphEncoder:
    """
    Factory function to create a GraphEncoder optimized for HotpotQA.
    
    Args:
        config: Configuration dictionary with model parameters
        phase: 1 for GraphSAGE (simple), 2 for GATv2 (advanced)
        
    Returns:
        encoder: Configured GraphEncoder instance
        
    Example:
        # Phase 1: Simple GraphSAGE
        config = {
            'model': {
                'hidden_dim': 256,
                'layers': 2,
                'dropout': 0.15
            }
        }
        encoder = create_hotpot_graph_encoder(config, phase=1)
        
        # Phase 2: Advanced GATv2 (after Phase 1 validation)
        encoder = create_hotpot_graph_encoder(config, phase=2)
    """
    model_config = config.get('model', {})
    
    # Default parameters optimized for HotpotQA
    d_node = 386  # SentenceTransformer (384) + similarity (1) + overlap (1)
    d_edge = 4    # [sim_pp, overlap_pp, sim_qp_src, sim_qp_dst]
    hidden = model_config.get('hidden_dim', 256)
    layers = model_config.get('layers', 2)
    heads = model_config.get('heads', 4)
    dropout = model_config.get('dropout', 0.15)
    activation = model_config.get('activation', 'relu')
    
    # Choose encoder type based on phase
    encoder_type = 'sage' if phase == 1 else 'gatv2'
    
    return GraphEncoder(
        d_node=d_node,
        d_edge=d_edge,
        hidden=hidden,
        layers=layers,
        heads=heads,
        dropout=dropout,
        activation=activation,
        encoder_type=encoder_type
    ) 