# DiT v2 Training Fixes

v1 (40k steps, 7M params, ε-prediction, no CFG) produced ε-MSE 0.012 but unconditional predictions — median TCP error ~90 cm on demo tasks, same pose regardless of input tokens. Root cause: the ε-loss at high t (t > 800) is minimized by predicting `ε_θ ≈ x_t`, which doesn't require using the task conditioning at all. The model found this shortcut and stopped using the cross-attention.

Apply these fixes in order before the next training run. Each is ranked by expected impact.

## ★★★★★ CFG dropout + learnable null token
**File:** `train_dit.py` (training loop) and `task_cond_dit.py` (add null token).
Drop the conditioning tokens with `p=0.1` during training:
```python
if torch.rand(1) < 0.1:
    token_mask = torch.zeros_like(token_mask)
    tokens = tokens * 0  # or swap for a learned null token
```
At inference use CFG with weight `w ≈ 3`:
```python
eps = (1 + w) * eps_cond - w * eps_uncond
```
Forces gradient to flow through cross-attention; breaks the "ignore condition" shortcut.

## ★★★★ v-prediction
**File:** `task_cond_dit.py` (sampler + loss target).
Switch the training target from ε to `v = α_t · ε − σ_t · x_0` (Salimans & Ho 2022). Two upsides:
- Numerically stable at both tails of t (no 1/√bar_α blowup)
- Empirically gives better sample quality on trajectory data
Conversion: at sample time, `ε_pred = α_t · v_pred + σ_t · x_t` (or skip ε and go straight to x_0 via `x_0 = α_t · x_t − σ_t · v_pred`).

## ★★★ Per-joint Q normalization
**File:** `train_dit.py` (dataset wrapper) + `infer_dit.py` (output denorm).
Compute per-joint `(mean, std)` over the training set, normalize q before noising:
```python
q_norm = (q - q_mean) / q_std
```
FR3 q4 (mean ≈ -1.6) and q6 (mean ≈ +2.5) are heavily off-center; without this the model wastes capacity on per-joint bias. Typical effect: 2× faster convergence.
Store `(q_mean, q_std)` in the checkpoint for inference.

## ★★ Use the ≥10 cm filtered dataset
Already produced by `fr3_dit/stitching/filter_min_subseg_length.py`. Use its output HDF5 as the new `--data` path. Short sub-segments (<10 cm) have noisy direction/Δθ tokens that hurt the conditioning signal.

## ★★ Longer training
v1 ran 40k steps × batch 128 on 1.2M tasks ≈ 4.3 epochs. Diffusion policies generally need 50–200 epochs. Bump `--num-steps` to **150k–200k**.

## ★ Bigger model (optional, only after the above)
- d_model 256 → 384
- n_layers 6 → 8
- Params 7M → ~15M
Only helps once training-length bottleneck is removed.

## Sampling-side at inference
- Keep DDIM, η=0, 100 steps
- Add CFG guidance `w` as a CLI flag (`--cfg 3.0`)
- After normalization + v-pred, the x0 clamp can go away; retain `clip=1.5` in normalized space as a safety net
