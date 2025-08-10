"""
Create paragraph-level embeddings for HotpotQA questions.
Produces emb_q (question embeddings) and emb_p (paragraph embeddings) per question.
"""

import os
import json
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Any, Tuple
import pickle


def extract_paragraphs_from_context(context: List[List]) -> List[str]:
    """
    Extract paragraph texts from HotpotQA context format.
    
    Args:
        context: List of [title, sentences_list] pairs
        
    Returns:
        paragraphs: List of paragraph texts (title + concatenated sentences)
    """
    paragraphs = []
    for title, sentences in context:
        # Concatenate title with all sentences in the paragraph
        paragraph_text = f"{title}. " + " ".join(sentences)
        paragraphs.append(paragraph_text)
    return paragraphs


def create_question_embeddings(questions: List[str], embedder: SentenceTransformer, 
                              batch_size: int = 64) -> np.ndarray:
    """
    Create embeddings for questions.
    
    Args:
        questions: List of question texts
        embedder: SentenceTransformer model
        batch_size: Batch size for encoding
        
    Returns:
        embeddings: [num_questions, d_txt] question embeddings
    """
    print(f"Encoding {len(questions)} questions...")
    embeddings = []
    
    for i in tqdm(range(0, len(questions), batch_size), desc="Question embeddings"):
        batch = questions[i:i+batch_size]
        with torch.no_grad():
            batch_emb = embedder.encode(batch, convert_to_tensor=True)
            if torch.cuda.is_available():
                batch_emb = batch_emb.cpu()
            embeddings.append(batch_emb.numpy())
    
    return np.vstack(embeddings)


def create_paragraph_embeddings(all_paragraphs: List[List[str]], embedder: SentenceTransformer,
                               batch_size: int = 64) -> List[np.ndarray]:
    """
    Create embeddings for paragraphs, maintaining per-question structure.
    
    Args:
        all_paragraphs: List of paragraph lists (one list per question)
        embedder: SentenceTransformer model
        batch_size: Batch size for encoding
        
    Returns:
        paragraph_embeddings: List of [n_paragraphs, d_txt] arrays (one per question)
    """
    # Flatten all paragraphs for efficient batch processing
    flat_paragraphs = []
    question_boundaries = []
    current_idx = 0
    
    for question_paragraphs in all_paragraphs:
        flat_paragraphs.extend(question_paragraphs)
        question_boundaries.append((current_idx, current_idx + len(question_paragraphs)))
        current_idx += len(question_paragraphs)
    
    print(f"Encoding {len(flat_paragraphs)} paragraphs...")
    
    # Batch encode all paragraphs
    all_embeddings = []
    for i in tqdm(range(0, len(flat_paragraphs), batch_size), desc="Paragraph embeddings"):
        batch = flat_paragraphs[i:i+batch_size]
        with torch.no_grad():
            batch_emb = embedder.encode(batch, convert_to_tensor=True)
            if torch.cuda.is_available():
                batch_emb = batch_emb.cpu()
            all_embeddings.append(batch_emb.numpy())
    
    # Concatenate all embeddings
    flat_embeddings = np.vstack(all_embeddings)
    
    # Split back into per-question structure
    paragraph_embeddings = []
    for start_idx, end_idx in question_boundaries:
        question_embs = flat_embeddings[start_idx:end_idx]
        paragraph_embeddings.append(question_embs)
    
    return paragraph_embeddings


def process_hotpot_file(file_path: str, embedder: SentenceTransformer, 
                       dataset_percentage: float = 100.0) -> Tuple[np.ndarray, List[np.ndarray], List[Dict]]:
    """
    Process a HotpotQA file to create embeddings.
    
    Args:
        file_path: Path to HotpotQA JSON file
        embedder: SentenceTransformer model
        dataset_percentage: Percentage of dataset to use (1-100)
        
    Returns:
        question_embeddings: [num_questions, d_txt] question embeddings
        paragraph_embeddings: List of [n_paragraphs, d_txt] paragraph embeddings per question
        metadata: List of processed examples with additional info
    """
    print(f"Loading data from {file_path}...")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Apply dataset percentage
    num_examples = int(len(data) * dataset_percentage / 100)
    data = data[:num_examples]
    print(f"Using {num_examples} examples ({dataset_percentage}%) from {file_path}")
    
    # Extract questions and paragraphs
    questions = []
    all_paragraphs = []
    metadata = []
    
    for example in tqdm(data, desc="Processing examples"):
        # Extract question
        question = example['question']
        questions.append(question)
        
        # Extract paragraphs from context
        paragraphs = extract_paragraphs_from_context(example['context'])
        all_paragraphs.append(paragraphs)
        
        # Create metadata entry
        meta = {
            'id': example.get('_id', ''),
            'question': question,
            'paragraphs': paragraphs,
            'supporting_facts': example.get('supporting_facts', []),
            'answer': example.get('answer', ''),
            'type': example.get('type', ''),
            'level': example.get('level', ''),
            'num_paragraphs': len(paragraphs)
        }
        
        # Convert supporting facts to paragraph indices
        support_indices = []
        paragraph_titles = [context[0] for context in example['context']]
        for title, sent_idx in example.get('supporting_facts', []):
            try:
                para_idx = paragraph_titles.index(title)
                if para_idx not in support_indices:
                    support_indices.append(para_idx)
            except ValueError:
                print(f"Warning: Supporting fact title '{title}' not found in context")
        
        meta['support_indices'] = support_indices
        metadata.append(meta)
    
    # Create embeddings
    question_embeddings = create_question_embeddings(questions, embedder)
    paragraph_embeddings = create_paragraph_embeddings(all_paragraphs, embedder)
    
    return question_embeddings, paragraph_embeddings, metadata


def save_embeddings(output_dir: str, prefix: str, question_embeddings: np.ndarray,
                   paragraph_embeddings: List[np.ndarray], metadata: List[Dict]):
    """
    Save embeddings and metadata to files.
    
    Args:
        output_dir: Output directory
        prefix: File prefix (e.g., 'train', 'dev')
        question_embeddings: Question embeddings array
        paragraph_embeddings: List of paragraph embedding arrays
        metadata: Metadata for each example
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save question embeddings
    question_file = os.path.join(output_dir, f"{prefix}_question_embeddings.npy")
    np.save(question_file, question_embeddings)
    print(f"Saved question embeddings: {question_embeddings.shape} -> {question_file}")
    
    # Save paragraph embeddings (as a list of arrays)
    paragraph_file = os.path.join(output_dir, f"{prefix}_paragraph_embeddings.pkl")
    with open(paragraph_file, 'wb') as f:
        pickle.dump(paragraph_embeddings, f)
    print(f"Saved paragraph embeddings: {len(paragraph_embeddings)} questions -> {paragraph_file}")
    
    # Save metadata
    metadata_file = os.path.join(output_dir, f"{prefix}_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"Saved metadata: {len(metadata)} examples -> {metadata_file}")
    
    # Print statistics
    print(f"\n📊 {prefix.upper()} Statistics:")
    print(f"  Questions: {len(question_embeddings)}")
    print(f"  Embedding dimension: {question_embeddings.shape[1]}")
    print(f"  Average paragraphs per question: {np.mean([len(p) for p in paragraph_embeddings]):.1f}")
    print(f"  Total paragraphs: {sum(len(p) for p in paragraph_embeddings)}")


def precompute_embeddings(train_file: str, dev_file: str, output_dir: str, 
                         dataset_percentage: float = 100.0, model_name: str = 'all-MiniLM-L6-v2'):
    """
    Main function to precompute embeddings for HotpotQA dataset.
    
    Args:
        train_file: Path to training JSON file
        dev_file: Path to development JSON file  
        output_dir: Output directory for embeddings
        dataset_percentage: Percentage of dataset to use (1-100)
        model_name: SentenceTransformer model name
    """
    print("🚀 Starting HotpotQA embedding precomputation...")
    print(f"Model: {model_name}")
    print(f"Dataset percentage: {dataset_percentage}%")
    
    # Initialize embedder
    print("Loading SentenceTransformer model...")
    embedder = SentenceTransformer(model_name)
    embedder.eval()
    if torch.cuda.is_available():
        embedder = embedder.cuda()
        print("Using GPU for embedding computation")
    else:
        print("Using CPU for embedding computation")
    
    # Process training file
    print("\n📚 Processing training data...")
    train_q_emb, train_p_emb, train_meta = process_hotpot_file(train_file, embedder, dataset_percentage)
    save_embeddings(output_dir, "train", train_q_emb, train_p_emb, train_meta)
    
    # Process development file
    print("\n📚 Processing development data...")
    dev_q_emb, dev_p_emb, dev_meta = process_hotpot_file(dev_file, embedder, dataset_percentage)
    save_embeddings(output_dir, "dev", dev_q_emb, dev_p_emb, dev_meta)
    
    print("\n✅ Embedding precomputation complete!")
    print(f"📁 Output directory: {output_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Precompute paragraph-level embeddings for HotpotQA")
    parser.add_argument('--train_file', type=str, required=True,
                       help='Path to HotpotQA training JSON file')
    parser.add_argument('--dev_file', type=str, required=True,
                       help='Path to HotpotQA development JSON file')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for embeddings')
    parser.add_argument('--dataset_percentage', type=float, default=100.0,
                       help='Percentage of dataset to use (1-100)')
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2',
                       help='SentenceTransformer model name')
    
    args = parser.parse_args()
    
    precompute_embeddings(
        train_file=args.train_file,
        dev_file=args.dev_file, 
        output_dir=args.output_dir,
        dataset_percentage=args.dataset_percentage,
        model_name=args.model_name
    ) 