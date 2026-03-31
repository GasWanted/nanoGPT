# nanoGPT

Exploratory repo for GPU-accelerated sparse neural network inference.

## Goal

Build a Rust tool that:
1. **Sparsifies** a trained network (pruning, quantization, structured sparsity)
2. **Streams** the sparse representation directly into the GPU for efficient inference

This small GPT (character-level, trained on Tiny Shakespeare) serves as the testbed. The techniques developed here will be applied to production-scale models: **Llama**, **DeepSeek**, **Qwen**.

## Current State

A minimal GPT implementation based on [Andrej Karpathy's "Zero to Hero"](https://karpathy.ai/zero-to-hero.html) lecture series.

- `model.py` — Transformer architecture (multi-head self-attention, feed-forward blocks, layer norm)
- `train.py` — Training loop with character-level tokenization on Tiny Shakespeare

### Quick Start

```bash
# Download the dataset
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# Train
python train.py
```

## Roadmap

- [x] Baseline GPT training in Python/PyTorch
- [ ] Export trained weights to a portable format
- [ ] Rust sparse format converter (CSR/CSC, block-sparse, N:M sparsity)
- [ ] CUDA/Vulkan kernel for sparse matmul streaming
- [ ] Benchmark: dense vs sparse inference latency & memory
- [ ] Scale to Llama / DeepSeek / Qwen checkpoints
