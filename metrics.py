"""
Evaluation metrics for multi-hop reasoning on HotpotQA.
"""

import numpy as np
from typing import List, Dict, Any, Union
import torch


def compute_f1_at_k(predictions: Union[np.ndarray, torch.Tensor], 
                   targets: Union[np.ndarray, torch.Tensor], k: int) -> float:
    """
    Compute F1@k metric for supporting fact prediction.
    
    Args:
        predictions: Prediction scores/probabilities [num_paragraphs]
        targets: Binary targets [num_paragraphs] 
        k: Number of top predictions to consider
        
    Returns:
        f1_score: F1@k score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    # Get top-k predictions
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0
    
    top_k_indices = np.argsort(predictions)[::-1][:k]
    
    # Convert to sets for easier intersection
    predicted_set = set(top_k_indices)
    true_set = set(np.where(targets == 1)[0])
    
    if len(predicted_set) == 0 and len(true_set) == 0:
        return 1.0  # Perfect if both empty
    
    if len(predicted_set) == 0 or len(true_set) == 0:
        return 0.0
    
    # Calculate precision, recall, F1
    intersection = predicted_set.intersection(true_set)
    precision = len(intersection) / len(predicted_set) if len(predicted_set) > 0 else 0.0
    recall = len(intersection) / len(true_set) if len(true_set) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def compute_precision_at_k(predictions: Union[np.ndarray, torch.Tensor], 
                          targets: Union[np.ndarray, torch.Tensor], k: int) -> float:
    """
    Compute Precision@k metric.
    
    Args:
        predictions: Prediction scores/probabilities [num_paragraphs]
        targets: Binary targets [num_paragraphs]
        k: Number of top predictions to consider
        
    Returns:
        precision: Precision@k score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0
    
    top_k_indices = np.argsort(predictions)[::-1][:k]
    
    predicted_set = set(top_k_indices)
    true_set = set(np.where(targets == 1)[0])
    
    if len(predicted_set) == 0:
        return 0.0
    
    intersection = predicted_set.intersection(true_set)
    precision = len(intersection) / len(predicted_set)
    return precision


def compute_recall_at_k(predictions: Union[np.ndarray, torch.Tensor], 
                       targets: Union[np.ndarray, torch.Tensor], k: int) -> float:
    """
    Compute Recall@k metric.
    
    Args:
        predictions: Prediction scores/probabilities [num_paragraphs]
        targets: Binary targets [num_paragraphs]
        k: Number of top predictions to consider
        
    Returns:
        recall: Recall@k score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0
    
    top_k_indices = np.argsort(predictions)[::-1][:k]
    
    predicted_set = set(top_k_indices)
    true_set = set(np.where(targets == 1)[0])
    
    if len(true_set) == 0:
        return 1.0 if len(predicted_set) == 0 else 0.0
    
    intersection = predicted_set.intersection(true_set)
    recall = len(intersection) / len(true_set)
    return recall


def compute_map(predictions: Union[np.ndarray, torch.Tensor], 
               targets: Union[np.ndarray, torch.Tensor]) -> float:
    """
    Compute Mean Average Precision (MAP).
    
    Args:
        predictions: Prediction scores/probabilities [num_paragraphs]
        targets: Binary targets [num_paragraphs]
        
    Returns:
        map_score: Mean Average Precision score
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    
    if len(predictions) == 0 or len(targets) == 0:
        return 0.0
    
    # Sort by prediction scores (descending)
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_targets = targets[sorted_indices]
    
    # Calculate precision at each relevant position
    precisions = []
    num_relevant = 0
    
    for i, target in enumerate(sorted_targets):
        if target == 1:
            num_relevant += 1
            precision_at_i = num_relevant / (i + 1)
            precisions.append(precision_at_i)
    
    if len(precisions) == 0:
        return 0.0
    
    return np.mean(precisions)


def evaluate_predictions(all_predictions: List[np.ndarray], 
                        all_targets: List[np.ndarray]) -> Dict[str, float]:
    """
    Evaluate predictions across multiple examples.
    
    Args:
        all_predictions: List of prediction arrays, one per example
        all_targets: List of target arrays, one per example
        
    Returns:
        metrics: Dictionary of computed metrics
    """
    metrics = {}
    
    # Compute metrics for each k
    for k in [1, 2, 3]:
        f1_scores = []
        precision_scores = []
        recall_scores = []
        
        for preds, targets in zip(all_predictions, all_targets):
            if len(targets) > 0:  # Only evaluate if there are targets
                f1_scores.append(compute_f1_at_k(preds, targets, k))
                precision_scores.append(compute_precision_at_k(preds, targets, k))
                recall_scores.append(compute_recall_at_k(preds, targets, k))
        
        if f1_scores:
            metrics[f'f1_at_{k}'] = np.mean(f1_scores)
            metrics[f'precision_at_{k}'] = np.mean(precision_scores)
            metrics[f'recall_at_{k}'] = np.mean(recall_scores)
    
    # Compute MAP
    map_scores = []
    for preds, targets in zip(all_predictions, all_targets):
        if len(targets) > 0:
            map_scores.append(compute_map(preds, targets))
    
    if map_scores:
        metrics['map'] = np.mean(map_scores)
    
    return metrics


def compute_chain_exact_match(predicted_chain: List[int], 
                             true_chain: List[int]) -> float:
    """
    Compute exact match for reasoning chains.
    
    Args:
        predicted_chain: Predicted reasoning chain
        true_chain: True reasoning chain
        
    Returns:
        exact_match: 1.0 if exact match, 0.0 otherwise
    """
    if len(predicted_chain) != len(true_chain):
        return 0.0
    
    return 1.0 if predicted_chain == true_chain else 0.0 