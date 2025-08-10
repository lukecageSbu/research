import torch
import os
from training_v4 import QuantumWalkRetriever, HotpotDataset
from torch.utils.data import DataLoader

def test_model_initialization():
    print("Testing model initialization...")
    model = QuantumWalkRetriever(embedding_dim=384, k=8, hidden_dim=128, walk_steps=3)
    print("Model initialized successfully!")
    return model

def test_data_loading():
    print("\nTesting data loading...")
    # Use a small subset of data for testing
    dataset = HotpotDataset(
        data_file="data/hotpot_train_v1.1.json",
        embeddings_dir="embeddings_v4",
        is_train=True,
        dataset_percentage=1  # Use only 1% of data for testing
    )
    print(f"Dataset loaded with {len(dataset)} examples")
    
    # Test data loader
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=4
    )
    print("DataLoader created successfully")
    return dataset, dataloader

def test_forward_pass(model, dataloader):
    print("\nTesting forward pass...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Get a single batch
    batch = next(iter(dataloader))
    
    # Prepare inputs
    questions = [ex['question'] for ex in batch]
    sent_embs = [torch.from_numpy(ex['sent_embs']).to(device) for ex in batch]
    neighbors = [torch.from_numpy(ex['neighbors']).to(device).long() for ex in batch]
    labels = [torch.tensor(ex['labels'], device=device).float() for ex in batch]
    
    # Forward pass
    with torch.no_grad():
        logits_list = model(questions, sent_embs, neighbors, labels)
    
    print(f"Forward pass successful! Got {len(logits_list)} outputs")
    return logits_list

def test_training_step(model, dataloader):
    print("\nTesting training step...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW([
        {'params': model.coin_net.parameters(), 'lr': 4e-4},
        {'params': model.path_net.parameters(), 'lr': 4e-4}
    ], weight_decay=0.01)
    
    # Get a single batch
    batch = next(iter(dataloader))
    
    # Prepare inputs
    questions = [ex['question'] for ex in batch]
    sent_embs = [torch.from_numpy(ex['sent_embs']).to(device) for ex in batch]
    neighbors = [torch.from_numpy(ex['neighbors']).to(device).long() for ex in batch]
    labels = [torch.tensor(ex['labels'], device=device).float() for ex in batch]
    
    # Training step
    optimizer.zero_grad()
    logits_list = model(questions, sent_embs, neighbors, labels)
    
    # Calculate loss
    batch_loss = 0
    for logits, lbl in zip(logits_list, labels):
        if lbl.sum() == 0:
            continue
        lbl = lbl / lbl.sum()
        probs = torch.softmax(logits.float(), dim=0)
        loss = torch.nn.functional.kl_div(probs.log(), lbl, reduction='batchmean')
        batch_loss += loss
    
    # Backward pass
    batch_loss.backward()
    optimizer.step()
    
    print("Training step completed successfully!")
    return batch_loss.item()

def main():
    print("Starting sanity tests...")
    
    # Test 1: Model initialization
    model = test_model_initialization()
    
    # Test 2: Data loading
    dataset, dataloader = test_data_loading()
    
    # Test 3: Forward pass
    logits_list = test_forward_pass(model, dataloader)
    
    # Test 4: Training step
    loss = test_training_step(model, dataloader)
    print(f"\nFinal loss: {loss:.4f}")
    
    print("\nAll sanity tests completed successfully!")

if __name__ == "__main__":
    main() 