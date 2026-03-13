#!/usr/bin/env python3
"""
our_model_their_data.py
========================
Applies the dmk_reproduction_of_Jespersen25.py MLP pipeline to the
pre-processed dataset distributed with Jespersen et al. (2025).

The key difference from dmk_reproduction_of_Jespersen25.py is the INPUT data:

  OUR DATA  (dmk_reproduction_of_Jespersen25.py):
    - SDSS spectra preprocessed by us → spender run by us → our latents
    - ~238K galaxies

  THEIR DATA (this script):
    - Pre-computed latents from Jespersen et al.'s own pipeline, distributed
      in sdss_wise_cross_reg0.csv + sdss_wise_cross_reg1.csv
    - ~764K galaxies (reg0: 252K + reg1: 512K), close to the paper's 510K+
    - Their preprocessing may differ subtly in normalisation, sky-line masking,
      and quality cuts, which propagates into the latent space and therefore
      into the final WISE predictions.

Because the latents are already computed in the CSV files, the spender encoding
step is bypassed entirely.  Everything else — MLP architecture, loss function,
training schedule, evaluation, plots — is identical to dmk_reproduction_of_Jespersen25.py.

Purpose: isolate whether any remaining gap to the paper's chi^2 values comes
from our latent encoding quality vs. from the MLP or training procedure.

Usage:
    python our_model_their_data.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm as scipy_norm
import time, logging

import torch
import torch.nn as nn
import torch.optim as optim

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("OurModelTheirData")


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(4)
torch.set_num_interop_threads(2)

# Input data from Jespersen et al.'s distribution
DATA_DIR = Path("external/IR_optical_demo/data")
REG0_CSV = DATA_DIR / "sdss_wise_cross_reg0.csv.gz"
REG1_CSV = DATA_DIR / "sdss_wise_cross_reg1.csv.gz"

OUT_DIR = Path("results/their_data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}")

# MLP hyperparameters — identical to dmk_reproduction_of_Jespersen25.py
LEARNING_RATE = 4e-3
N_EPOCHS      = 500000
PATIENCE      = 1000
VAL_EVERY     = 10

# Sigma threshold above which a WISE band is considered undetected.
# The CSV encodes non-detections as sigma = 9999; a threshold of 5 cleanly
# separates real detections (sigma typically 0.02–0.5 mag) from upper limits.
NONDET_THRESHOLD = 5.0

# For the training loss, non-detected bands need a large sigma so the NLL
# gradient is ~0 for those data points.  The CSV already uses 9999; we keep it.
NON_DETECTION_SIGMA = 1e4

# Reference chi^2_N values from Jespersen et al. (2025) Table 3.
# NOTE on chi definition: following the demo notebook (IR_optical_demo.ipynb),
# chi is defined as  chi = (pred - obs) / sqrt(sigma_mag)  rather than the
# more common  chi = (pred - obs) / sigma_mag.  This is consistent with the
# GaussianNLLLoss using var = sqrt(sigma) as its variance argument.  The
# paper's reported chi^2_N values correspond to this definition.
PAPER_CHI2      = {"W1": 1.33, "W2": 1.17, "W3": 2.23, "W4": 1.41}
WISE_BAND_SHORT = ["W1", "W2", "W3", "W4"]
WISE_BAND_LABELS = {
    "W1": r"WISE 3.4 $\mu$m",  "W2": r"WISE 4.6 $\mu$m",
    "W3": r"WISE 12.1 $\mu$m", "W4": r"WISE 22.2 $\mu$m",
}

print("=" * 70)
print("OUR MODEL, THEIR DATA  —  Jespersen et al. (2025)")
print("=" * 70)
print(f"Device:       {DEVICE}")
print(f"Data:         {REG0_CSV.name} + {REG1_CSV.name}")
print(f"Architecture: 8 → 20 → 50 → 50 → 20 → 4  (4,884 parameters)")
print(f"Targets:      WISE Vega magnitudes (W1–W4)")
print(f"Chi def:      (pred - obs) / sqrt(sigma_mag)  [matches demo notebook]")
print(f"Max epochs:   {N_EPOCHS}  |  patience={PATIENCE}  |  val_every={VAL_EVERY}")
print()


# =============================================================================
# STEP 1: Load pre-computed latents and WISE photometry
# =============================================================================
#
# The CSV files distributed by Jespersen et al. contain, for each galaxy:
#
#   spender_0..5  — 6 latent dimensions from their spender encoding pipeline.
#                   These may differ from our encodings due to differences in
#                   SDSS preprocessing (sky-line masking, normalisation window,
#                   quality cuts), even though both use the same sdss_II model.
#   Z             — spectroscopic redshift from SDSS
#   reg           — V-band normalisation constant (same role as our norm_const)
#   w1..w4mpro    — WISE Vega magnitudes
#   w1..w4sigmpro — WISE magnitude errors; 9999 for undetected bands

log.info("=== STEP 1: Loading Jespersen et al. pre-computed dataset ===")

for p in [REG0_CSV, REG1_CSV]:
    if not p.exists():
        raise FileNotFoundError(f"Missing {p} — place it in {DATA_DIR}/")

df0 = pd.read_csv(REG0_CSV)
df1 = pd.read_csv(REG1_CSV)
log.info(f"  reg0: {len(df0):,} rows")
log.info(f"  reg1: {len(df1):,} rows")

df = pd.concat([df0, df1], ignore_index=True)
N  = len(df)
log.info(f"  Combined: {N:,} galaxies")

# ── Extract features ─────────────────────────────────────────────────────────
spender_cols = [f"spender_{i}" for i in range(6)]
feature_cols = spender_cols + ["Z", "reg"]

X_input = df[feature_cols].to_numpy(dtype=np.float32)   # (N, 8)
X_input = np.nan_to_num(X_input, nan=0.0, posinf=0.0, neginf=0.0)

# ── Extract targets ───────────────────────────────────────────────────────────
mag_cols = ["w1mpro",    "w2mpro",    "w3mpro",    "w4mpro"]
sig_cols = ["w1sigmpro", "w2sigmpro", "w3sigmpro", "w4sigmpro"]

Y_mag_raw   = df[mag_cols].to_numpy(dtype=np.float32)   # (N, 4) Vega magnitudes
Y_sigma_raw = df[sig_cols].to_numpy(dtype=np.float32)   # (N, 4) mag errors

# Detections: sigma < NONDET_THRESHOLD (separates real photometry from upper limits)
detected = Y_sigma_raw < NONDET_THRESHOLD     # (N, 4) bool

# For training: non-detected bands keep sigma = 9999 (already set in the CSV),
# capped at NON_DETECTION_SIGMA to guard against any larger sentinel values.
# Non-detected magnitude values are kept as-is from the CSV (real flux measurements
# at the source position, just with large sigma).  Do NOT replace with 0.0 — that
# would introduce a large systematic gradient toward unphysically bright predictions.
Y_sigma = np.where(detected, Y_sigma_raw, NON_DETECTION_SIGMA).astype(np.float32)
Y_sigma = np.where((Y_sigma > 0) & np.isfinite(Y_sigma), Y_sigma, NON_DETECTION_SIGMA)
Y_mag   = Y_mag_raw.copy().astype(np.float32)

log.info(f"Input shape:  {X_input.shape}")
log.info(f"Target shape: {Y_mag.shape}")
for j, b in enumerate(WISE_BAND_SHORT):
    n_det = detected[:, j].sum()
    m = Y_mag[detected[:, j], j]
    s = Y_sigma[detected[:, j], j]
    log.info(f"  {b}: {n_det:,}/{N:,} detected ({100*n_det/N:.1f}%)  "
             f"mag=[{m.min():.1f}, {m.max():.1f}]  sigma=[{s.min():.4f}, {s.max():.3f}]")

# ── Latent space summary ──────────────────────────────────────────────────────
print(f"\nLatent space statistics (from Jespersen et al. pipeline):")
for d in range(6):
    print(f"  L{d+1}: mean={X_input[:,d].mean():.3f}  std={X_input[:,d].std():.3f}")
print(f"  Z:   mean={X_input[:,6].mean():.3f}  std={X_input[:,6].std():.3f}")
print(f"  reg: mean={X_input[:,7].mean():.3f}  std={X_input[:,7].std():.3f}")


# =============================================================================
# STEP 2: Train / test split (80 / 20)
# =============================================================================
#
# Jespersen et al. distribute a test_ids.csv that indexes into reg1, intended
# for exact replication of their held-out set.  Here we use a random 80/20
# split on the combined dataset to keep our methodology self-consistent.
# The chi^2 comparison to the paper is informative either way, because with
# 764K galaxies the test set is large enough that split choice has negligible
# impact on aggregate statistics.

log.info("\n=== STEP 2: Train/test split (80/20) ===")

ids = np.arange(N)
np.random.shuffle(ids)
n_train   = int(N * 0.8)
n_test    = N - n_train
idx_train = ids[:n_train]
idx_test  = ids[n_train:]

X_tr = X_input[idx_train]
X_te = X_input[idx_test]

# Concatenate [mag, sigma] → (N, 8) as in demo's dat_y
Y_combined_tr = np.concatenate([Y_mag[idx_train],   Y_sigma[idx_train]],  axis=1).astype(np.float32)
Y_combined_te = np.concatenate([Y_mag[idx_test],    Y_sigma[idx_test]],   axis=1).astype(np.float32)

det_te        = detected[idx_test]    # (N_test, 4)

log.info(f"Train: {n_train:,}  |  Test: {n_test:,}")


# =============================================================================
# STEP 3: Build MLP
# =============================================================================

HIDDEN_LAYERS = [20, 50, 50, 20]

def build_mlp(n_in=8, hidden_layers=None, n_out=4):
    if hidden_layers is None:
        hidden_layers = [20, 50, 50, 20]
    layers = []
    prev = n_in
    for h in hidden_layers:
        layers += [nn.Linear(prev, h), nn.ReLU()]
        prev = h
    layers.append(nn.Linear(prev, n_out))
    return nn.Sequential(*layers)


model    = build_mlp(n_in=8, hidden_layers=HIDDEN_LAYERS, n_out=4).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nMLP: {n_params} parameters")
print(model)

loss_fn = nn.GaussianNLLLoss()

def compute_loss(predictions, y_combined):
    """
    GaussianNLLLoss with var = sqrt(sigma_mag), matching the demo exactly.
    The var argument to GaussianNLLLoss is the variance (σ²) of the Gaussian.
    Passing sqrt(sigma_mag) as 'var' is an unconventional choice that
    down-weights high-uncertainty bands less aggressively than using sigma^2.
    """
    mags  = y_combined[:, :4]
    sigma = y_combined[:, 4:]
    var   = torch.sqrt(sigma)    # demo's weighting: sqrt(sigma) as variance
    return loss_fn(predictions, mags, var)


# =============================================================================
# STEP 4: Training
# =============================================================================
#
# With 764K galaxies the full dataset is ~24 MB — still trivially fits in GPU
# VRAM, so we retain full-batch gradient steps for clean convergence.

log.info("\n=== STEP 3: Training ===")

X_tr_gpu = torch.tensor(X_tr, dtype=torch.float32).to(DEVICE)
Y_tr_gpu = torch.tensor(Y_combined_tr, dtype=torch.float32).to(DEVICE)
X_te_gpu = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
Y_te_gpu = torch.tensor(Y_combined_te, dtype=torch.float32).to(DEVICE)
n_tr = len(X_tr_gpu); n_te = len(X_te_gpu)
log.info(f"Data pre-loaded to {DEVICE}: train={n_tr:,}  test={n_te:,}")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=100, min_lr=1e-6,
    threshold=1e-6, threshold_mode="rel"
)

train_losses, test_losses = [], []
min_test_loss = float("inf")
n_no_improve  = 0
best_state    = None
t0            = time.time()

log.info(f"Training: max {N_EPOCHS} epochs, patience={PATIENCE}, "
         f"full-batch ({n_tr:,} samples), lr={LEARNING_RATE}")

for epoch in range(N_EPOCHS):
    if n_no_improve >= PATIENCE:
        log.info(f"Early stopping at epoch {epoch} "
                 f"(no improvement for {PATIENCE} consecutive val-checks)")
        break

    model.train()
    optimizer.zero_grad()
    out  = model(X_tr_gpu)
    loss = compute_loss(out, Y_tr_gpu)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    t_loss_sum = loss.item()

    if epoch % VAL_EVERY == 0:
        train_losses.append(t_loss_sum)

        model.eval()
        with torch.no_grad():
            v_loss_sum = compute_loss(model(X_te_gpu), Y_te_gpu).item()

        test_losses.append(v_loss_sum)
        scheduler.step(v_loss_sum)

        if v_loss_sum < min_test_loss * (1 - 1e-6):
            min_test_loss = v_loss_sum
            n_no_improve  = 0
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            n_no_improve += 1

        if (epoch // VAL_EVERY) % 50 == 0 or epoch < 20:
            elapsed = time.time() - t0
            cur_lr  = optimizer.param_groups[0]["lr"]
            log.info(f"Epoch {epoch:>6d}  train={t_loss_sum:.4f}  val={v_loss_sum:.4f}  "
                     f"patience={n_no_improve}/{PATIENCE}  lr={cur_lr:.2e}  ({elapsed:.0f}s)")

if best_state is not None:
    model.load_state_dict(best_state)
    log.info(f"Restored best model weights (min val loss = {min_test_loss:.6f})")

total_time = time.time() - t0
log.info(f"Training complete: {total_time:.1f}s, {epoch+1} epochs")

# ── Training curve ────────────────────────────────────────────────────────────
from scipy.signal import medfilt
fig, ax = plt.subplots(figsize=(9, 3))
x_epochs = np.arange(len(train_losses)) * VAL_EVERY
smooth   = min(21, len(train_losses) | 1)
ax.plot(x_epochs, medfilt(train_losses, smooth), alpha=0.8, label="Train loss")
ax.plot(x_epochs, medfilt(test_losses,  smooth), alpha=0.8, label="Validation loss")
ax.set(yscale="log", xscale="log", xlim=(max(1, x_epochs[1]), None),
       title=f"GaussianNLL Loss (smoothed)  —  {n_train:,} training galaxies,  {epoch+1} epochs",
       ylabel="Loss", xlabel="Epoch")
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "training_curve.png", dpi=130, bbox_inches="tight")
plt.close()
log.info("Saved training_curve.png")


# =============================================================================
# STEP 5: Evaluate on the test set
# =============================================================================
log.info("\n=== STEP 4: Evaluating on test set ===")

model.eval()
with torch.no_grad():
    Y_pred_mag = model(
        torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    ).cpu().numpy()    # (N_test, 4) predicted Vega magnitudes

Y_true_mag  = Y_mag[idx_test]
Y_sigma_mag = Y_sigma_raw[idx_test]   # raw sigma (not capped) for evaluation

# ── Chi-squared ───────────────────────────────────────────────────────────────
# Following the demo notebook exactly:
#   chi = (pred - obs) / sqrt(sigma_mag)
# This normalises residuals by the square root of the photometric uncertainty
# rather than the uncertainty itself, consistent with the loss function's use
# of sqrt(sigma) as the GaussianNLLLoss variance argument.
# Only detected bands (sigma < 5 mag) enter the statistic.
#
# For reference we also report the standard chi = (pred - obs) / sigma_mag.

print(f"\n{'Band':<6s} {'χ²_N (demo def)':>16s} {'χ²_N (std def)':>15s} {'Paper':>8s} "
      f"{'N_det':>8s} {'IQR_χ':>8s} {'<χ>':>7s}")
print("-" * 72)
chi_all_demo = {}    # chi = residual / sqrt(sigma)
chi_all_std  = {}    # chi = residual / sigma  (standard)

for j, b in enumerate(WISE_BAND_SHORT):
    det_mask = det_te[:, j] & (Y_sigma_mag[:, j] < NONDET_THRESHOLD)
    n_det    = det_mask.sum()
    if n_det < 5:
        print(f"{b:<6s} {'N/A':>16s} {'N/A':>15s} {PAPER_CHI2[b]:>8.2f} {n_det:>8d}")
        continue

    resid = Y_pred_mag[det_mask, j] - Y_true_mag[det_mask, j]
    sig   = Y_sigma_mag[det_mask, j]

    chi_demo = resid / np.sqrt(sig)          # matches demo notebook
    chi_std  = resid / sig                   # standard normalisation
    chi2n_demo = np.mean(chi_demo**2)
    chi2n_std  = np.mean(chi_std**2)
    iqr_chi    = (np.percentile(chi_demo, 84) - np.percentile(chi_demo, 16)) / 2.0
    chi_all_demo[b] = chi_demo
    chi_all_std[b]  = chi_std
    print(f"{b:<6s} {chi2n_demo:>16.2f} {chi2n_std:>15.2f} {PAPER_CHI2[b]:>8.2f} "
          f"{n_det:>8d} {iqr_chi:>8.3f} {np.mean(chi_demo):>7.3f}")


# =============================================================================
# STEP 6: Chi distribution plots
# =============================================================================

def IQR_sig(x):
    iqr = np.nanpercentile(x, [84, 16])
    return (iqr[0] - iqr[1]) / 2.0

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for idx, b in enumerate(WISE_BAND_SHORT):
    ax       = axes[idx // 2, idx % 2]
    det_mask = det_te[:, idx] & (Y_sigma_mag[:, idx] < NONDET_THRESHOLD)
    if det_mask.sum() < 5:
        ax.text(0.5, 0.5, f"Too few {b} detections", transform=ax.transAxes, ha="center")
        ax.set_title(WISE_BAND_LABELS[b]); continue

    resid    = Y_pred_mag[det_mask, idx] - Y_true_mag[det_mask, idx]
    sig      = Y_sigma_mag[det_mask, idx]
    chi      = resid / np.sqrt(sig)    # demo definition
    chi2n    = np.mean(chi**2)
    iqr_chi  = IQR_sig(chi)

    ax.hist(chi, bins=100, range=(-10, 10), histtype="step",
            density=True, color="steelblue", linewidth=1.5)
    x_ = np.linspace(-10, 10, 400)
    ax.plot(x_, scipy_norm.pdf(x_), "k--", lw=1.5, label=r"$\mathcal{N}(0,1)$")
    ax.set(xlim=(-10, 10),
           xlabel=r"$\chi = (\mathrm{pred}-\mathrm{obs})/\sqrt{\sigma}$",
           ylabel="Density", title=WISE_BAND_LABELS[b])
    ax.legend(fontsize=8)

    xt, yt, dy = 0.05, 0.82, 0.12
    font = {"size": 10}
    ax.text(xt, yt,      fr"$\chi^2_N$ = {chi2n:.2f}",       transform=ax.transAxes, fontdict=font)
    ax.text(xt, yt-dy,   fr"IQR: {iqr_chi:.2f}",             transform=ax.transAxes, fontdict=font)
    ax.text(xt, yt-2*dy, fr"$\langle\chi\rangle$ = {np.mean(chi):.2f}", transform=ax.transAxes, fontdict=font)

plt.suptitle(
    r"$\chi$-distribution  $[\chi = (\mathrm{pred}-\mathrm{obs})/\sqrt{\sigma}]$"
    "\nJespersen et al. latents + our MLP",
    fontsize=12, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUT_DIR / "chi_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved chi_distribution.png")


# =============================================================================
# STEP 7: PP-plot (paper Figure 3)
# =============================================================================

log.info("\n=== STEP 5: PP-plot ===")

paper_phat_mean = {"W1": 0.66, "W2": 0.70, "W3": 0.63, "W4": 0.70}
phat_emp, phat_data = {}, {}

print(f"\n{'Band':<6s} {'<p̂_emp>':>10s} {'<p̂_data>':>10s} {'paper':>8s} {'N':>8s}")
print("-" * 46)
for j, b in enumerate(WISE_BAND_SHORT):
    det_mask = det_te[:, j] & (Y_sigma_mag[:, j] < NONDET_THRESHOLD)
    if det_mask.sum() < 5:
        print(f"{b:<6s} {'—':>10s} {'—':>10s} {paper_phat_mean[b]:>8.2f} {det_mask.sum():>8d}")
        continue

    y_obs  = Y_true_mag[det_mask, j]
    y_pred = Y_pred_mag[det_mask, j]
    sigma  = np.sqrt(Y_sigma_mag[det_mask, j])

    phat_emp[b]  = scipy_norm.pdf(y_pred, loc=y_obs,  scale=sigma)
    phat_data[b] = scipy_norm.pdf(0,      loc=0,       scale=sigma)

    print(f"{b:<6s} {np.mean(phat_emp[b]):>10.4f} {np.mean(phat_data[b]):>10.4f} "
          f"{paper_phat_mean[b]:>8.2f} {det_mask.sum():>8d}")

fig, axes = plt.subplots(2, 2, figsize=(10, 10))
for idx, b in enumerate(WISE_BAND_SHORT):
    ax = axes[idx // 2, idx % 2]
    if b not in phat_emp:
        ax.text(0.5, 0.5, f"No {b} detections in test set",
                transform=ax.transAxes, ha="center", va="center")
        ax.set_title(WISE_BAND_LABELS[b]); continue

    pe  = np.sort(phat_emp[b])
    pd_ = np.sort(phat_data[b])
    thresholds = np.linspace(0, max(pe.max(), pd_.max()), 1000)
    cdf_e = np.searchsorted(pe,  thresholds) / len(pe)
    cdf_d = np.searchsorted(pd_, thresholds) / len(pd_)

    ax.plot([0, 1], [0, 1],  "r-",  lw=1.5,           label="Perfect calibration (diagonal)")
    ax.plot(cdf_d, cdf_d,    "k--", lw=1.5, alpha=0.7, label="Data baseline")
    ax.plot(cdf_d, cdf_e,    "g-",  lw=2.0,            label="This model")
    ax.annotate("Overconfident\nand biased",
                xy=(0.65, 0.35), xytext=(0.45, 0.15),
                fontsize=8, color="gray", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set(xlabel="CDF(p̂_data)", ylabel="CDF(p̂_empirical)",
           title=WISE_BAND_LABELS[b], xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.legend(loc="upper left", fontsize=8)

plt.suptitle(
    "PP-plot: probabilistic calibration of WISE magnitude predictions\n"
    "Jespersen et al. latents + our MLP  (cf. Figure 3 of Jespersen et al. 2025)",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "ppplot_figure3.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved ppplot_figure3.png")


# =============================================================================
# STEP 8: Predicted vs observed
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
names = ["WISE 3.4μm", "WISE 4.6μm", "WISE 12.1μm", "WISE 22.2μm"]

for j, b in enumerate(WISE_BAND_SHORT):
    ax       = axes[j // 2, j % 2]
    det_mask = det_te[:, j] & (Y_sigma_mag[:, j] < NONDET_THRESHOLD)
    if det_mask.sum() < 3:
        ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center")
        ax.set_title(b); continue

    true    = Y_true_mag[det_mask, j]
    pred    = Y_pred_mag[det_mask, j]
    scatter = (np.percentile(pred-true, [75, 25])[0] -
               np.percentile(pred-true, [75, 25])[1]) / 1.349
    bias    = np.median(true - pred)
    tot     = np.hstack([true, pred])
    l       = 0.01

    ax.hist2d(true, pred, bins=50,
              range=[np.percentile(tot, [l, 100-l])] * 2,
              norm=plt.matplotlib.colors.LogNorm(), cmap="viridis")
    lims = np.percentile(tot, [l, 100-l])
    ax.plot(lims, lims, "k--", lw=1, label="Perfect")
    ax.set(xlabel=f"Observed [{names[j]}]", ylabel=f"Predicted [{names[j]}]",
           title=names[j])
    font = {"size": 10}
    ax.text(0.03, 0.85, f"Bias: {bias:.2f} mag",  transform=ax.transAxes, fontdict=font)
    ax.text(0.03, 0.77, f"σ_IQR: {scatter:.2f}", transform=ax.transAxes, fontdict=font)
    ax.legend(loc="lower right", fontsize=8)

plt.suptitle("Predicted vs Observed WISE magnitudes (Vega)\nJespersen et al. latents + our MLP",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "mag_pred_vs_obs.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved mag_pred_vs_obs.png")


# =============================================================================
# Save model and results
# =============================================================================

torch.save(model.state_dict(), OUT_DIR / "mlp_weights.pt")
np.savez_compressed(
    OUT_DIR / "test_results.npz",
    y_pred_mag=Y_pred_mag, y_true_mag=Y_true_mag, y_sigma_mag=Y_sigma_mag,
    detected=det_te, idx_test=idx_test,
)

print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"  Dataset:      {N:,} galaxies  (train={n_train:,}, test={n_test:,})")
print(f"  Architecture: 8→20→50→50→20→4  ({n_params} params)")
print(f"  Epochs run:   {epoch+1}  |  Training time: {total_time:.0f}s")
print(f"\n  Chi-squared  [chi = (pred-obs)/sqrt(sigma)]  vs paper Table 3:")
print(f"  {'Band':<6s} {'χ²_N (demo)':>12s} {'χ²_N (std)':>12s} {'Paper':>8s}")
print("  " + "-" * 42)
for j, b in enumerate(WISE_BAND_SHORT):
    det_mask = det_te[:, j] & (Y_sigma_mag[:, j] < NONDET_THRESHOLD)
    if det_mask.sum() < 5 or b not in chi_all_demo:
        print(f"  {b:<6s} {'N/A':>12s} {'N/A':>12s} {PAPER_CHI2[b]:>8.2f}")
        continue
    chi2n_demo = np.mean(chi_all_demo[b]**2)
    chi2n_std  = np.mean(chi_all_std[b]**2)
    print(f"  {b:<6s} {chi2n_demo:>12.2f} {chi2n_std:>12.2f} {PAPER_CHI2[b]:>8.2f}")

print(f"\n  Outputs saved to: {OUT_DIR}/")
for f in sorted(OUT_DIR.iterdir()):
    if f.is_file():
        print(f"    {f.name:<45s} {f.stat().st_size/1024:>7.1f} kB")
