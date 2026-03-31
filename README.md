# nanoGPT

Exploratory repo for GPU-accelerated sparse neural network inference. A minimal GPT serves as the testbed for building a Rust tool that sparsifies networks and streams them into the GPU — targeting production models like **Llama**, **DeepSeek**, and **Qwen**.

Based on [Andrej Karpathy's "Zero to Hero"](https://karpathy.ai/zero-to-hero.html) lecture series.

## Project Structure

```
nanoGPT/
├── model.py          # GPT architecture (attention, feed-forward, transformer blocks)
├── train.py          # Training loop, data loading, checkpoint saving
├── generate.py       # Text generation from a trained checkpoint
├── data/
│   └── prepare.py    # Download Tiny Shakespeare dataset
├── checkpoints/      # Saved model weights (gitignored)
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
python data/prepare.py
```

## Train

```bash
python train.py
```

Trains a 0.2M parameter character-level GPT on Tiny Shakespeare (5000 steps). Saves a checkpoint to `checkpoints/checkpoint.pt`.

| | |
|---|---|
| **Parameters** | 0.21M |
| **Layers** | 4 |
| **Heads** | 4 |
| **Embedding dim** | 64 |
| **Context length** | 32 |
| **Final train loss** | ~1.66 |
| **Final val loss** | ~1.82 |

## Generate

```bash
# Unconditional
python generate.py

# With a prompt
python generate.py --prompt "ROMEO:" --tokens 1000
```

## Architecture

```
Input tokens
    → Token Embedding + Positional Embedding
    → 4x Transformer Block
        → LayerNorm → Multi-Head Self-Attention (4 heads) → Residual
        → LayerNorm → Feed-Forward (ReLU, 4x expansion) → Residual
    → LayerNorm
    → Linear → Logits
```

## Roadmap

- [x] Baseline GPT training in Python/PyTorch
- [x] Checkpoint saving and generation script
- [ ] Export weights to a portable format (safetensors / raw binary)
- [ ] Rust sparse format converter (CSR/CSC, block-sparse, N:M sparsity)
- [ ] CUDA/Vulkan kernel for sparse matmul streaming
- [ ] Benchmark: dense vs sparse inference latency & memory
- [ ] Scale to Llama / DeepSeek / Qwen checkpoints
