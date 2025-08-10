#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
import torch

def main():
    parser = argparse.ArgumentParser(description='Debug HotpotQA dataset loading.')
    parser.add_argument('--train_file', type=str, default='data/hotpot_train_v1.1.json',
                        help='Path to training data file')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings_v4',
                        help='Directory containing embeddings')
    parser.add_argument('--dataset_percentage', type=float, default=10,
                        help='Percentage of dataset to use (1-100)')
    parser.add_argument('--k', type=int, default=2,
                        help='K parameter for neighbors')
    args = parser.parse_args()
    
    debug_dataset_loading(args.train_file, args.embeddings_dir, args.dataset_percentage, args.k)

def debug_dataset_loading(data_file, embeddings_dir, dataset_percentage, k):
    print(f"Loading data file: {data_file}")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Calculate how many examples to use based on percentage
    total_examples = len(data)
    num_examples = int(total_examples * (dataset_percentage / 100))
    data = data[:num_examples]
    print(f"Using {len(data)} out of {total_examples} examples ({dataset_percentage}%)")
    
    # Load precomputed embeddings and metadata
    prefix = "train"
    print(f"Loading embeddings from {embeddings_dir}")
    sent_embs = np.load(os.path.join(embeddings_dir, f"{prefix}_embeddings.npy"))
    doc_offsets = np.load(os.path.join(embeddings_dir, f"{prefix}_offsets.npy"))
    
    # Filter out empty examples
    print("Filtering out empty examples...")
    valid_indices = []
    empty_indices = []
    for idx in range(len(data)):
        if idx >= len(doc_offsets):
            print(f"Warning: Index {idx} is out of bounds for doc_offsets with length {len(doc_offsets)}")
            continue
            
        start, end = doc_offsets[idx]
        if start < end:  # Only keep non-empty examples
            valid_indices.append(idx)
        else:
            empty_indices.append((idx, start, end))
    
    print(f"Found {len(empty_indices)} empty examples in offset data")
    if empty_indices:
        print("First 10 empty examples:")
        for idx, start, end in empty_indices[:10]:
            print(f"  Index {idx}: start={start}, end={end}")
            if idx < len(data):
                question = data[idx].get('question', 'N/A')
                answer = data[idx].get('answer', 'N/A')
                print(f"  Question: {question}")
                print(f"  Answer: {answer}")
                
                # Check context length
                context_length = 0
                if 'context' in data[idx]:
                    for title, sentences in data[idx]['context']:
                        context_length += len(sentences)
                print(f"  Raw context length: {context_length} sentences")
    
    # Continue with the original filtering logic to see what happens
    data = [data[i] for i in valid_indices]
    print(f"After filtering, we have {len(data)} examples left")
    
    # Load Faiss index
    print("Loading Faiss index...")
    
    # Check if k-NN neighbors file exists
    neighbors_file = os.path.join(embeddings_dir, f"{prefix}_neighbors_k{k}.npy")
    if os.path.exists(neighbors_file):
        print(f"Found precomputed neighbors file: {neighbors_file}")
    else:
        print(f"No precomputed neighbors file found for k={k}")
    
    # Try preparing examples
    print("Preparing examples (debugging _prepare_examples)...")
    
    # Check for valid/invalid starts and ends
    print("Validating start/end indices...")
    
    # For the first 100 valid examples, check their offsets
    for i, idx in enumerate(valid_indices[:100]):
        start, end = doc_offsets[idx]
        example_data = data[i]  # Note: data has been filtered, so we use i instead of idx
        context_length = 0
        if 'context' in example_data:
            for title, sentences in example_data['context']:
                context_length += len(sentences)
        
        # Compute the embedding length and compare with context length
        embedding_length = end - start
        print(f"Example {i} (orig_idx={idx}): start={start}, end={end}, embedding_length={embedding_length}, context_length={context_length}")
        
        # Check if embeddings actually exist for this range
        if start >= len(sent_embs) or end > len(sent_embs):
            print(f"  WARNING: Offset indices out of bounds for embeddings array (len={len(sent_embs)})")
        
        # Try accessing the embeddings
        try:
            example_embs = sent_embs[start:end]
            print(f"  Successfully accessed embeddings with shape {example_embs.shape}")
        except Exception as e:
            print(f"  ERROR accessing embeddings: {e}")
        
        if i >= 10:  # Only show details for first 10 examples
            continue

if __name__ == "__main__":
    main() 