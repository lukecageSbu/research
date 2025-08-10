# Quantum Walk Retriever for HotpotQA

This project implements a Quantum Walk-based retrieval model for the HotpotQA dataset, using distributed training with PyTorch.

## Project Structure

```
.
├── src/               # Source code files
│   ├── training_v3.py       # Main training script (latest version)
│   ├── training.py          # Original training implementation
│   ├── training_improved.py # Improved training implementation
│   └── evaluate.py         # Evaluation script
├── models/            # Saved model files
│   ├── coin_net.pth        # Original model weights
│   ├── coin_net_v3.pth     # Version 3 model weights
│   └── coin_net_improved.pth # Improved model weights
├── data/              # Data directory
│   └── hotpot/            # HotpotQA dataset
├── logs/              # Training logs
├── notebooks/         # Jupyter notebooks
│   ├── experiment.ipynb   # Experimental notebooks
│   └── training.ipynb    # Training notebooks
├── checkpoints/       # Model checkpoints
├── checkpoints_v3/    # Version 3 checkpoints
├── checkpoints_improved/  # Improved version checkpoints
├── venv/              # Python virtual environment
└── run_training.sh    # Training script
```

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install torch torchvision torchaudio
pip install sentence-transformers faiss-gpu networkx scikit-learn
pip install tqdm numpy
```

3. Prepare data:
- Download HotpotQA dataset and place in `data/hotpot/`
- Ensure the following files exist:
  - `data/hotpot/hotpot_train_v1.1.json`
  - `data/hotpot/hotpot_dev_distractor_v1.json`

## Training

The project uses distributed training with PyTorch's DistributedDataParallel (DDP). The main training script is `run_training.sh`, which:
- Kills any existing training processes
- Creates necessary directories
- Runs training using `torchrun` with 4 GPUs
- Saves logs with timestamps
- Saves model checkpoints

To start training:
```bash
chmod +x run_training.sh  # Make script executable
./run_training.sh        # Run training
```

Current training configuration:
- 4 GPUs (RTX A6000)
- Batch size: 64 (training), 128 (evaluation)
- Learning rate: 1e-4
- Model: Quantum Walk Retriever with SentenceTransformer embeddings

## Model Architecture

The Quantum Walk Retriever consists of:
1. Sentence embedding model (SentenceTransformer)
2. Graph construction module
3. Quantum walk implementation
4. Coin network for controlling walk dynamics

## Monitoring

- Training logs are saved in `logs/` with timestamps
- Model checkpoints are saved in `checkpoints_v3/`
- Best model is saved as `models/coin_net_v3.pth`
- Monitor training with:
```bash
tail -f logs/training_*.log  # Latest log file
nvidia-smi                   # GPU usage
```

## Evaluation

Use `evaluate.py` to evaluate the model. The script calculates:
- Mean Reciprocal Rank (MRR)
- Recall@K
- Support Exact Match (EM)

## Development Notes

- `training_v3.py` is the latest version with improved metrics and distributed training
- Checkpoints are saved after each epoch
- Training can be resumed from the last checkpoint
- The script handles CUDA out-of-memory by adjusting batch sizes
- Uses BCEWithLogitsLoss for binary classification of supporting facts

## Hardware Requirements

- 4x NVIDIA RTX A6000 GPUs (48GB VRAM each)
- Recommended: 64GB+ System RAM
- Storage: ~100GB for dataset and checkpoints