import os
import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
import subprocess
import sys
from collections import OrderedDict

# Add src directory to path to import training modules
src_dir = os.path.join(os.path.dirname(__file__), 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Try importing required components from training_v6.py
try:
    from training_v6 import QuantumWalkRetriever, HotpotDataset, custom_collate_fn
except ImportError as e:
    print(f"Error importing from src/training_v6.py: {e}")
    print("Please ensure 'src/training_v6.py' exists and is runnable, and run this script from the project root directory.")
    sys.exit(1)

def load_model(checkpoint_path, model_k, hidden_dim, walk_steps, device):
    """Loads the QuantumWalkRetriever model from a checkpoint."""
    print(f"Loading model with k={model_k}, hidden_dim={hidden_dim}, walk_steps={walk_steps}")
    # Ensure embedding_dim matches the pre-trained embedder
    embedding_dim = 384 # Assuming 'all-MiniLM-L6-v2'
    model = QuantumWalkRetriever(
        embedding_dim=embedding_dim,
        k=model_k,
        hidden_dim=hidden_dim,
        walk_steps=walk_steps
    )

    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu') # Load to CPU first

    # Handle potential DDP 'module.' prefix
    state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    is_ddp = any(key.startswith('module.') for key in state_dict.keys())
    print(f"Checkpoint appears to be from DDP: {is_ddp}")

    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k # remove `module.` prefix if present
        new_state_dict[name] = v

    # Load the state dict
    load_result = model.load_state_dict(new_state_dict, strict=False)
    print(f"Model load results (missing/unexpected): {load_result}")

    model.to(device)
    model.eval()
    print("Model loaded successfully and set to evaluation mode.")
    return model

def predict_supporting_facts(model, dataloader, device, top_pred_k):
    """Generates supporting fact predictions for the dataset."""
    predictions = {"answer": {}, "sp": {}}
    original_data_map = {example['question']: example for example in dataloader.dataset.data}

    with torch.no_grad():
        batch_num = 0
        for batch in tqdm(dataloader, desc="Generating Predictions"):
            batch_num += 1
            # Prepare batch data (similar to training/eval loop)
            questions = [ex['question'] for ex in batch]
            sent_embs = [torch.from_numpy(ex['sent_embs']).to(device) for ex in batch]
            neighbors = [torch.from_numpy(ex['neighbors']).to(device).long() for ex in batch]
            # Labels might be needed if model forward pass requires them, pass dummy if not
            labels = [torch.zeros(ex['sent_embs'].shape[0], device=device).float() for ex in batch]
            contexts = [ex['context'] for ex in batch] # Get context for mapping indices
            supporting_facts_gold = [ex['supporting_facts'] for ex in batch] # For reference/debugging

            # Filter out examples within the batch that might be empty individually
            valid_indices = [i for i, emb in enumerate(sent_embs) if emb.size(0) > 0]
            if not valid_indices:
                print(f"Skipping empty batch {batch_num}")
                continue

            # Filter batch data based on valid indices
            questions_valid = [questions[i] for i in valid_indices]
            sent_embs_valid = [sent_embs[i] for i in valid_indices]
            neighbors_valid = [neighbors[i] for i in valid_indices]
            labels_valid = [labels[i] for i in valid_indices] # Use filtered labels
            contexts_valid = [contexts[i] for i in valid_indices]
            # Get original example IDs for valid examples
            original_ids = [dataloader.dataset.data[dataloader.dataset.examples[batch[i]['idx']].get('orig_idx', batch[i]['idx'])].get('_id')
                            for i in valid_indices if 'idx' in batch[i]] # Need example index from dataset
            if not original_ids or len(original_ids) != len(questions_valid):
                 # Fallback using question text if _id is missing (less reliable)
                 print("Warning: Could not reliably get original IDs, using question text. Ensure HotpotDataset stores '_id'.")
                 original_ids = [original_data_map[q]['_id'] for q in questions_valid if q in original_data_map]
                 if len(original_ids) != len(questions_valid):
                     print(f"Error: ID mismatch even with fallback ({len(original_ids)} vs {len(questions_valid)}). Skipping batch.")
                     continue


            # Model forward pass
            logits_list = model(questions_valid, sent_embs_valid, neighbors_valid, labels_valid) # Pass labels if needed

            if len(logits_list) != len(questions_valid):
                 print(f"Warning: Mismatch between model output ({len(logits_list)}) and valid inputs ({len(questions_valid)}). Skipping batch.")
                 continue

            # Process results for each example in the batch
            for i, logits in enumerate(logits_list):
                q_id = original_ids[i]
                context = contexts_valid[i]
                num_sentences = logits.size(0)

                if num_sentences == 0:
                    predictions["sp"][q_id] = []
                    predictions["answer"][q_id] = "no answer" # Or some indicator
                    continue

                # Convert logits to probabilities - Use sigmoid for BCEWithLogitsLoss outputs
                probs = torch.sigmoid(logits.float()) # Changed from softmax

                # Select top_pred_k indices
                k_to_select = min(top_pred_k, num_sentences)
                pred_scores, pred_indices_local = torch.topk(probs, k_to_select)
                pred_indices_local = pred_indices_local.cpu().tolist()

                # Map local indices back to [title, sent_idx]
                predicted_sp_list = []
                global_sent_idx = 0
                for title, sentences in context:
                    for sent_idx, _ in enumerate(sentences):
                        if global_sent_idx in pred_indices_local:
                            predicted_sp_list.append([title, sent_idx])
                        global_sent_idx += 1
                        if len(predicted_sp_list) == len(pred_indices_local): # Optimization
                            break
                    if len(predicted_sp_list) == len(pred_indices_local): # Optimization
                            break

                # Store predictions
                predictions["sp"][q_id] = predicted_sp_list
                predictions["answer"][q_id] = "dummy answer" # Official script needs this key

    return predictions

def main(args):
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load Model
    model = load_model(
        args.checkpoint_path,
        args.model_k,
        args.hidden_dim,
        args.walk_steps,
        device
    )

    # Load Dev Dataset
    print("Loading dev dataset...")
    # Note: The 'k' for HotpotDataset is for loading k-NN neighbors file
    # Ensure this matches the k used when precomputing neighbors (e.g., k=4 from run script)
    dev_dataset = HotpotDataset(
        data_file=args.dev_file,
        embeddings_dir=args.embeddings_dir,
        is_train=False,
        dataset_percentage=100, # Evaluate on full dev set
        k=args.data_loading_k # k used for neighbor file name
    )
    print(f"Dev dataset loaded with {len(dev_dataset)} examples")

    # Add original index to examples for ID mapping
    for i, ex in enumerate(dev_dataset.examples):
        ex['idx'] = i # Store the index within the prepared examples list


    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=custom_collate_fn
    )

    # Generate Predictions
    predictions = predict_supporting_facts(model, dev_dataloader, device, args.top_pred_k)

    # Save Predictions
    print(f"Saving predictions to: {args.output_file}")
    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)
    with open(args.output_file, 'w') as f:
        json.dump(predictions, f, indent=4)
    print("Predictions saved.")

    # Run Official Evaluation Script
    # Construct absolute path to the evaluation script
    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    eval_script_path = os.path.join(workspace_root, 'hotpot', 'hotpot_evaluate_v1.py')
    # eval_script_path = os.path.join('hotpot', 'hotpot_evaluate_v1.py') # Old relative path
    gold_file_path = args.dev_file
    print(f"Running official evaluation script: {eval_script_path}")
    # Ensure the command uses the same python executable that is running this script
    python_executable = sys.executable 
    command = [
        python_executable, # Use the same python interpreter
        eval_script_path,
        args.output_file,
        gold_file_path
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("--- Official Evaluation Results ---")
        print(result.stdout)
        if result.stderr:
            print("--- Evaluation Script Errors/Warnings ---")
            print(result.stderr)
    except FileNotFoundError:
        print(f"Error: Evaluation script not found at {eval_script_path}")
        print("Ensure the script exists and you are running from the project root.")
    except subprocess.CalledProcessError as e:
        print(f"Error running evaluation script: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate QuantumWalkRetriever on HotpotQA Supporting Facts")

    # --- File Paths ---
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints_v6_pathnet_final/best_model.pt',
                        help='Path to the trained model checkpoint (.pt file)')
    parser.add_argument('--dev_file', type=str, default='data/hotpot_dev_distractor_v1.json',
                        help='Path to the HotpotQA dev JSON file')
    parser.add_argument('--embeddings_dir', type=str, default='embeddings_v4_k4',
                        help='Directory containing precomputed dev embeddings and neighbors')
    parser.add_argument('--output_file', type=str, default='predictions/predictions_v6_pathnet_final.json',
                        help='Path to save the generated predictions JSON file')

    # --- Model Hyperparameters (MUST match training) ---
    parser.add_argument('--model_k', type=int, default=4,
                        help='Internal k parameter of the QuantumWalkRetriever model')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension used in the model')
    parser.add_argument('--walk_steps', type=int, default=4,
                        help='Number of walk steps used in the model')

    # --- Data Loading & Evaluation Parameters ---
    parser.add_argument('--data_loading_k', type=int, default=4,
                         help='k value used for finding/loading k-NN neighbors file (e.g., dev_neighbors_k{k}.npy)')
    parser.add_argument('--eval_batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for DataLoader')
    parser.add_argument('--top_pred_k', type=int, default=2,
                        help='Number of supporting facts to predict per question')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu', 'cuda:0', 'cuda:1'], # Add specific GPUs if needed
                        help='Device to use for evaluation (auto detects CUDA)')

    args = parser.parse_args()

    # --- Basic Validation ---
    if not os.path.exists(args.checkpoint_path):
         print(f"Error: Checkpoint path not found: {args.checkpoint_path}")
         sys.exit(1)
    if not os.path.exists(args.dev_file):
         print(f"Error: Dev file path not found: {args.dev_file}")
         sys.exit(1)
    if not os.path.exists(args.embeddings_dir):
         print(f"Error: Embeddings directory not found: {args.embeddings_dir}")
         sys.exit(1)
    neighbor_file = os.path.join(args.embeddings_dir, f"dev_neighbors_k{args.data_loading_k}.npy")
    if not os.path.exists(neighbor_file):
        print(f"Error: Expected neighbor file not found: {neighbor_file}")
        print(f"Ensure '--data_loading_k' ({args.data_loading_k}) matches the k used for precomputing neighbors.")
        sys.exit(1)


    main(args) 