# OPDforLLMs — On-Policy Distillation for LLMs

Implementing **Generalized Knowledge Distillation (GKD)** and **On-Policy Distillation** for language models, based on the paper:

> *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes*
> Agarwal et al., ICLR 2024

The notebook runs entirely on CPU in a few minutes using a small toy model, making every concept accessible without a GPU.

---

## Contents

| File | Description |
|------|-------------|
| `On_Policy_Distillation.ipynb` | Main tutorial notebook (13 sections) |
| `README.md` | Minimal project description |

---

## Tutorial Overview

### 1. Setup & Imports

Key dependencies: `torch`, `matplotlib`, `numpy`, plus standard library utilities (`copy`, `random`, `collections`).

```bash
pip install torch numpy matplotlib
```

### 2. Task & Vocabulary

The toy task is generating simple arithmetic expressions — e.g. given the prompt `3+2=`, the model outputs `5<EOS>`.

- Vocabulary: digits `0–9`, `+`, `=`, `<EOS>`, `<PAD>` (14 tokens total)
- Dataset: all single-digit addition problems (sums ≤ 18), ~100 examples
- Simple enough to train in seconds, rich enough to reveal distillation dynamics

### 3. Model Architecture

A small **LSTM-based autoregressive language model** (`TinyLM`):

| Model | Hidden Size | Role |
|-------|-------------|------|
| Teacher | 128 | Larger, better-performing model trained first |
| Student | 32 | Smaller model to be distilled |

Both share the same interface, making them interchangeable in any training loop.

### 4. Divergence Functions

GKD supports pluggable divergence functions between teacher and student distributions, applied token-by-token and averaged over sequence length:

- **Forward KL** — `D_KL(p_T ∥ p_S)`: mode-covering; forces the student to assign mass everywhere the teacher does
- **Reverse KL** — `D_KL(p_S ∥ p_T)`: mode-seeking; student focuses on the teacher's dominant modes, producing sharper but less diverse output
- **JSD(β)** — Generalized Jensen-Shannon Divergence: interpolates between the two
  - β → 0 behaves like forward KL
  - β → 1 behaves like reverse KL

### 5. Train the Teacher via SFT

The teacher is first trained with standard **cross-entropy** on ground-truth answer sequences (Supervised Fine-Tuning). This serves as the capable reference model for all subsequent distillation experiments.

### 6. Baseline: SFT Student

The student is trained on the same ground-truth data with standard cross-entropy. Because it is smaller, it plateaus below the teacher — this is the baseline all distillation methods are compared against.

### 7. Supervised KD (λ = 0)

Classic knowledge distillation: the student minimises **forward KL** against the teacher's distributions on **fixed ground-truth sequences**.

This is GKD with λ = 0 (no on-policy data). The train-inference distribution mismatch problem remains because sequences come from the fixed dataset, not from the student itself.

### 8. On-Policy GKD (λ = 1)

The core contribution of the paper. Instead of fixed sequences, training proceeds in three steps each iteration:

1. **Student generates** its own output sequences given input prompts
2. **Teacher is queried** for token-level distributions on those self-generated sequences
3. **Divergence is minimised** between teacher and student on those positions

The student now trains on exactly the distribution it encounters at inference. Mistakes made during generation become the training signal for correction.

> **Gradient note:** Backpropagation does NOT flow through the sampling step. The sampled sequence is treated as fixed data; only token-level logits receive gradients, keeping training stable and efficient.

### 9. Varying Divergences & λ

The best divergence is task-dependent. The notebook experiments with:

- **Reverse KL** (mode-seeking)
- **JSD(β = 0.9)** (near-reverse-KL)
- **Mixed λ = 0.5** (blend of fixed and on-policy data)

### 10. GKD + RL Fine-Tuning

Combines distillation with a REINFORCE-style RL objective:

$$L = (1 - \alpha) \cdot \mathbb{E}_{y \sim p_S}[-r(y)] + \alpha \cdot D(p_T \| p_S)(y \mid x)$$

**Shaped reward function:**

| Outcome | Reward |
|---------|--------|
| Exact correct answer | +1.0 |
| Correct number of digits | +0.3 |
| Otherwise | 0.0 |

A running-mean **baseline** is used to reduce REINFORCE variance.

### 11. Comparing All Methods

Side-by-side visualisation of all training curves — SFT student, supervised KD, on-policy GKD variants, and GKD+RL — against the teacher ceiling.

### 12. Qualitative Outputs

Sample generations from each trained model on held-out prompts for qualitative comparison.

### 13. Key Takeaways

| Concept | Implementation |
|---------|----------------|
| Train-inference mismatch | Supervised KD trains on fixed sequences; student-generated sequences differ |
| Forward KL is mode-covering | `forward_kl()` — forces student to cover all teacher mass |
| Reverse KL is mode-seeking | `reverse_kl()` — student concentrates on teacher's dominant modes |
| JSD interpolates | `jsd(beta=...)` — beta near 0 ≈ forward KL, beta near 1 ≈ reverse KL |
| GKD algorithm | `train_onpolicy_gkd()` — λ controls mix of fixed and on-policy data |
| No gradient through sampling | `student_generate_batch()` wraps generation in `torch.no_grad()` |
| GKD + RL | `train_gkd_rl()` — combined RL reward + distillation loss with α trade-off |

---

## Paper Reference

Agarwal, R., et al. (2024). *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes.* ICLR 2024.
