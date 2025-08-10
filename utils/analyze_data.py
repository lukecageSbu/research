#!/usr/bin/env python

import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description='Analyze HotpotQA dataset and embeddings.')
    parser.add_argument('--train_file', type=str, default='data/hotpot_train_v1.1.json',
                        help='Path to training data file')
    parser.add_argument('--dev_file', type=str, default='data/hotpot_dev_distractor_v1.json',
                        help='Path to dev data file')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings_v4',
                        help='Directory containing embeddings')
    args = parser.parse_args()
    
    # Analyze training data
    print("Analyzing training data...")
    analyze_dataset(args.train_file, args.embeddings_dir, is_train=True)
    
    # Analyze dev data
    print("\nAnalyzing dev data...")
    analyze_dataset(args.dev_file, args.embeddings_dir, is_train=False)

def analyze_dataset(data_file, embeddings_dir, is_train=True):
    # Load data file
    print(f"Loading data from {data_file}")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    print(f"Dataset contains {len(data)} examples")
    
    # Load embeddings and offsets
    prefix = "train" if is_train else "dev"
    embeddings_file = os.path.join(embeddings_dir, f"{prefix}_embeddings.npy")
    offsets_file = os.path.join(embeddings_dir, f"{prefix}_offsets.npy")
    
    print(f"Loading embeddings from {embeddings_file}")
    embeddings = np.load(embeddings_file)
    
    print(f"Loading offsets from {offsets_file}")
    offsets = np.load(offsets_file)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Offsets shape: {offsets.shape}")
    
    # Check for empty examples (where start >= end)
    empty_examples = []
    non_empty_examples = []
    
    for idx, (start, end) in enumerate(offsets):
        example_size = end - start
        if example_size <= 0:
            empty_examples.append(idx)
        else:
            non_empty_examples.append((idx, example_size))
    
    print(f"Empty examples: {len(empty_examples)} ({len(empty_examples)/len(data)*100:.2f}%)")
    print(f"Non-empty examples: {len(non_empty_examples)} ({len(non_empty_examples)/len(data)*100:.2f}%)")
    
    # Analyze first and last 100 examples to check for patterns
    print("\nAnalyzing first 100 examples:")
    analyze_example_range(data, offsets, 0, 100)
    
    print("\nAnalyzing last 100 examples:")
    analyze_example_range(data, offsets, len(data)-100, len(data))
    
    # Analyze distribution of example sizes
    sizes = [end - start for start, end in offsets]
    plot_size_distribution(sizes, prefix)
    
    # Analyze specific examples
    if empty_examples:
        print("\nExamining first 5 empty examples:")
        for idx in empty_examples[:5]:
            if idx < len(data):
                print(f"\nExample {idx} (start={offsets[idx][0]}, end={offsets[idx][1]}):")
                print(f"Question: {data[idx].get('question', 'N/A')}")
                print(f"Answer: {data[idx].get('answer', 'N/A')}")
                context_length = 0
                supporting_facts = 0
                
                if 'context' in data[idx]:
                    for title, sentences in data[idx]['context']:
                        context_length += len(sentences)
                
                if 'supporting_facts' in data[idx]:
                    supporting_facts = len(data[idx]['supporting_facts'])
                
                print(f"Context: {context_length} sentences across {len(data[idx].get('context', []))} documents")
                print(f"Supporting facts: {supporting_facts}")
    
    # Check potential mismatch between offsets and data
    if len(offsets) != len(data):
        print(f"\nWARNING: Mismatch between offsets ({len(offsets)}) and data ({len(data)})")

def analyze_example_range(data, offsets, start_idx, end_idx):
    """Analyze a range of examples to find patterns of empty vs non-empty examples."""
    empty_count = 0
    sentence_counts = []
    
    for idx in range(start_idx, min(end_idx, len(data))):
        if idx >= len(offsets):
            continue
            
        start, end = offsets[idx]
        example_size = end - start
        
        if example_size <= 0:
            empty_count += 1
        
        # Count sentences in context
        context_sentence_count = 0
        if 'context' in data[idx]:
            for title, sentences in data[idx]['context']:
                context_sentence_count += len(sentences)
        
        sentence_counts.append(context_sentence_count)
    
    print(f"Empty examples in range: {empty_count}/{min(end_idx, len(data))-start_idx} ({empty_count/(min(end_idx, len(data))-start_idx)*100:.2f}%)")
    print(f"Average sentence count: {sum(sentence_counts)/len(sentence_counts) if sentence_counts else 0:.2f}")
    print(f"Min sentence count: {min(sentence_counts) if sentence_counts else 0}")
    print(f"Max sentence count: {max(sentence_counts) if sentence_counts else 0}")

def plot_size_distribution(sizes, prefix):
    """Plot the distribution of example sizes."""
    plt.figure(figsize=(10, 6))
    plt.hist(sizes, bins=50)
    plt.title(f'{prefix.capitalize()} Example Size Distribution')
    plt.xlabel('Number of Sentences')
    plt.ylabel('Frequency')
    plt.savefig(f'{prefix}_size_distribution.png')
    
    # Print statistics
    print("\nSize statistics:")
    print(f"Min size: {min(sizes)}")
    print(f"Max size: {max(sizes)}")
    print(f"Mean size: {np.mean(sizes):.2f}")
    print(f"Median size: {np.median(sizes):.2f}")
    
    # Count zeros and negative values
    zeros = sum(1 for s in sizes if s == 0)
    negatives = sum(1 for s in sizes if s < 0)
    print(f"Zero-sized examples: {zeros} ({zeros/len(sizes)*100:.2f}%)")
    print(f"Negative-sized examples: {negatives} ({negatives/len(sizes)*100:.2f}%)")

if __name__ == "__main__":
    main() 