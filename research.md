# Research Notes

## Model Compression Pipeline

The full compression pipeline for taking a large model to edge-deployable:

```
Large Model (e.g. DeepSeek 671B)
  → Distillation  → Smaller architecture trained to mimic the original (~7-14B)
  → Sparsification → Zero out / remove unneeded weights (50-90% reduction)
  → Quantization   → Reduce precision: float32 → int4 (8x memory savings)
```

Each step compounds. A 671B fp32 model (~1.3 TB) can become a 7B sparse int4 model (~3.5 GB) that runs on consumer GPUs.

## Compression Techniques

### Distillation

Train a smaller "student" model to mimic a larger "teacher" model using soft targets (probability distributions) rather than hard labels. The student learns the teacher's "dark knowledge" — the relative likelihoods between outputs, not just the top prediction.

- **Pros:** Best quality at extreme compression ratios (100x). Student can restructure representations entirely.
- **Cons:** Expensive. Requires running the teacher for inference across billions of tokens. Needs GPU clusters.
- **For us:** Already done by Meta, DeepSeek, Qwen — we download their published distilled models.

### Magnitude Pruning

Zero out weights below a threshold. Simple, effective, no training required.

```
Original:  [0.8, -0.02, 0.5, 0.001, -0.7]
Pruned:    [0.8,  0.0,  0.5,  0.0,  -0.7]   (40% sparse)
```

- **Pros:** Dead simple. Pure math. Works on any model.
- **Cons:** Unstructured sparsity is hard to accelerate on GPUs without sparse kernels.

### SVD Decomposition

Decomposes a weight matrix into independent "signals" ranked by importance. Keeps the top-k signals, discards the rest. The result is NEW weight values — not a subset of the originals.

```
W [256 × 64]  →  U [256 × 16] × S [16] × V [16 × 64]
16,384 params  →  5,120 params (70% reduction)
```

- **Pros:** Pure math, instant, no training data needed. Genuinely transforms weights.
- **Cons:** Linear-only. Cannot capture nonlinear interactions (ReLU, softmax between layers).
- **Mitigation:** Activation-aware SVD uses a calibration pass to weight the decomposition by what the network actually does with those weights in practice.

### Why SVD Is Linear-Only

A transformer is: `Linear → ReLU → Linear → Softmax → Linear → ...`

SVD on a single weight matrix is blind to the nonlinear activations between layers. Two layers connected through ReLU work as a *pair* — SVD on either layer individually doesn't see this pairing. A row in W1 might look unimportant by magnitude but activate a critical ReLU pathway that W2 depends on.

**Approaches that capture nonlinear structure:**

| Method | Linear? | Nonlinear? | Cost |
|---|---|---|---|
| Raw SVD | Yes | No | Free (pure math) |
| Activation-aware SVD | Yes | Partially | ~1 min (calibration pass) |
| Block-wise reconstruction | Yes | Mostly | ~minutes (small optimization per block) |
| SVD + light fine-tuning | Yes | Yes | ~minutes (few training steps to fix errors) |
| Full distillation | Yes | Yes | Hours to days |

### Quantization

Reduce numerical precision of weights. No structural change, just fewer bits per value.

```
float32 (4 bytes) → float16 (2 bytes) → int8 (1 byte) → int4 (0.5 bytes)
```

Typically applied last since pruning/SVD work better in higher precision.

## Open Research Questions

### Mapping Distillation as a Transformation

Current distillation requires expensive training. But what if the mapping from teacher weights to student weights follows discoverable patterns?

**Observations from existing research:**
- CKA studies show student layers absorb multiple teacher layers (many-to-few mapping, not uniform)
- Student representations are more linearly separable than teacher's — distillation forces efficient reorganization
- "Born Again Networks" (2018) showed distilling into the SAME architecture improves quality — distillation regularizes, not just compresses
- Student early layers closely match teacher early layers; divergence increases in deeper layers

**Nobody has built:** An empirical study that takes many (teacher, student) pairs and maps the weight-level transformation patterns:
- How do singular value distributions change?
- Which teacher neurons map to which student neurons?
- Is there a consistent compression pattern per layer type (attention vs FFN)?
- Do attention heads merge in predictable ways?

If patterns exist, distillation could potentially be approximated as a learned meta-transformation — a "model that transforms models" trained on (teacher, student) pairs. This would replace expensive training with a direct weight mapping function.

**Relevant prior work:**
- Net2Net (2016) — function-preserving transforms, expansion only
- Knowledge Flow (2019) — tracks info movement during distillation, observational
- DepGraph (2023) — structural dependency mapping for pruning

## Hardware Constraints (RTX 4070 Ti, 12 GB VRAM)

The compression pipeline (pruning, SVD, quantization) is CPU-only math — no GPU required. GPU matters only for inference/benchmarking.

| Model | fp16 | int4 | Fits? |
|---|---|---|---|
| nanoGPT (0.2M) | 0.4 MB | 0.1 MB | Yes |
| Llama-3-8B | 16 GB | 4 GB | int4 yes |
| Qwen-2.5-7B | 14 GB | 3.5 GB | int4 yes |
| DeepSeek-R1-Distill-7B | 14 GB | 3.5 GB | int4 yes |
