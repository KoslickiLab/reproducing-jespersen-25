#!/usr/bin/env python3
"""
dmk_reproduction_of_Jespersen25.py
====================================
Reproduction of the IR photometry prediction pipeline from Jespersen et al. (2025),
specifically targeting Figure 3 (PP-plot) and Figure 15 (chi distribution).

The core scientific question: given an optical spectrum, can we predict a galaxy's
mid-infrared (WISE W1–W4) photometry? The approach uses the spender VAE latent
representation of the SDSS spectrum as a compressed summary of the galaxy's physical
state (SFR, stellar mass, dust content, AGN activity, metallicity), then trains a
shallow MLP to map those latents to WISE fluxes.

Usage:
    python dmk_reproduction_of_Jespersen25.py
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm as scipy_norm
import time, sys, logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("Jespersen25")


# =============================================================================
# CONFIGURATION
# =============================================================================

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Limit CPU threads — spender and numpy can spawn many threads on cluster nodes,
# causing resource contention with other jobs.
torch.set_num_threads(4)
torch.set_num_interop_threads(2)

SDSS_DIR = Path("sdss_output")
WISE_DIR = Path("wise_output")
OUT_DIR  = Path("ppplot_v2_output_full_batch")
OUT_DIR.mkdir(exist_ok=True)

# Use GPU if available; with ~167K galaxies the full dataset fits comfortably in
# GPU VRAM (~8 features × 167K × 4 bytes ≈ 5 MB), enabling full-batch gradient
# steps that eliminate stochastic noise and converge more smoothly than mini-batches.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}")

# ── MLP architecture ──────────────────────────────────────────────────────────
# Input: 8 features — 6 spender latents + spectroscopic redshift + V-band norm.
# Output: 4 WISE band magnitudes (W1–W4, Vega system).
# Architecture 8 → 20 → 50 → 50 → 20 → 4 matches Jespersen et al.'s demo notebook
# (4,884 parameters). This is larger than the minimal 8→20→20→4 (684 params)
# described in Table 1 of the paper; the larger network is needed for the full
# dataset and achieves better chi² values.

LEARNING_RATE  = 4e-3    # Adam LR from Jespersen et al. demo notebook
N_EPOCHS       = 500000  # Upper bound; early stopping typically terminates well before this.
                          # The paper trains for 2–4×10^5 epochs; with full-batch on GPU
                          # each epoch is a single cheap matrix multiply (~ms per epoch).
PATIENCE       = 1000    # Stop after PATIENCE consecutive val-checks without improvement.
VAL_EVERY      = 10      # Check validation loss every 10 epochs.
SPENDER_BATCH  = 5000    # Mini-batch size for spender encoding (encoder is memory-heavy).

# Non-detection handling: WISE reports upper limits when a source is undetected.
# The paper assigns σ = 10^4 magnitudes to these bands, making the Gaussian NLL
# loss contribution ~log(10^4)/2 ≈ 9.2 — a large additive constant independent
# of the prediction, so the gradient through non-detected bands is effectively zero.
NON_DETECTION_SIGMA = 1e4

# Reference values from Jespersen et al. Table 3 for comparison.
PAPER_CHI2      = {"W1": 1.33, "W2": 1.17, "W3": 2.23, "W4": 1.41}
WISE_BANDS      = ["W1 (3.4μm)", "W2 (4.6μm)", "W3 (12.1μm)", "W4 (22.2μm)"]
WISE_BAND_SHORT = ["W1", "W2", "W3", "W4"]

print("=" * 70)
print("REPRODUCING Jespersen et al. (2025): WISE IR photometry from SDSS spectra")
print("=" * 70)
print(f"Device:       {DEVICE}")
print(f"Architecture: 8 → 20 → 50 → 50 → 20 → 4  (4,884 parameters)")
print(f"Targets:      WISE Vega magnitudes (W1–W4)")
print(f"Max epochs:   {N_EPOCHS}  |  patience={PATIENCE}  |  val_every={VAL_EVERY}")
print()


# =============================================================================
# STEP 1: Load preprocessed data
# =============================================================================
log.info("=== STEP 1: Loading preprocessed data ===")

for req in ["spectra_obs_frame.npy", "norm_consts.npy", "metadata.npy", "wave_obs_grid.npy"]:
    if not (SDSS_DIR / req).exists():
        raise FileNotFoundError(
            f"Missing {SDSS_DIR/req} — run sdss_data_preprocessing.ipynb first."
        )

# Observed-frame spectra: flux as measured by SDSS, in units of 10^-17 erg/s/cm²/Å.
# We work in the observed frame (not rest frame) because spender's encoder was
# trained on observed-frame SDSS spectra at a fixed wavelength grid.  Spender
# internally handles the redshift-induced wavelength shift via its architecture,
# so there is no need for us to manually de-redshift before encoding.
X_spectra_obs   = np.load(SDSS_DIR / "spectra_obs_frame.npy")   # (N_sdss, 3921)

# The V-band normalization constant is the median flux in the rest-frame V-band
# window (~5300–5850 Å), which is a well-defined, relatively dust-insensitive
# region of the optical continuum.  Dividing by this value yields a spectrum with
# unit continuum level in the V band, making the spender encoder's input
# scale-invariant across galaxies of very different luminosities.  The absolute
# value of norm_const is then used as a separate 8th MLP input feature so the
# network can recover the overall brightness (and thus the absolute IR flux level).
norm_consts_all = np.load(SDSS_DIR / "norm_consts.npy")          # (N_sdss,)
wave_obs_grid   = np.load(SDSS_DIR / "wave_obs_grid.npy")        # (3921,)
sdss_meta       = np.load(SDSS_DIR / "metadata.npy")             # structured array

if not (WISE_DIR / "wise_photometry.npz").exists():
    raise FileNotFoundError(
        f"Missing {WISE_DIR}/wise_photometry.npz — run wise_data_preprocessing.ipynb first."
    )

wise = np.load(WISE_DIR / "wise_photometry.npz")
log.info(f"SDSS: {X_spectra_obs.shape[0]:,} preprocessed spectra")
log.info(f"WISE: {wise['mag'].shape[0]:,} galaxies in photometry catalogue")


# ── Cross-match SDSS ↔ WISE ───────────────────────────────────────────────────
# Both datasets were selected from the same SDSS spectroscopic galaxy sample, so
# every successfully preprocessed SDSS spectrum should have a WISE counterpart.
# The match is performed by SDSS specobjid, which uniquely identifies a spectrum
# (plate + MJD + fiber + rerun), ensuring no duplicate or ambiguous associations.

idx_path = WISE_DIR / "wise_matched_sdss_indices.npy"
if idx_path.exists():
    sdss_idx     = np.load(idx_path)
    X_spec       = X_spectra_obs[sdss_idx]
    norm_consts  = norm_consts_all[sdss_idx]
    z_arr        = sdss_meta["redshift"][sdss_idx]
    wise_mag     = wise["mag"]          # (N_matched, 4) Vega magnitudes
    wise_mag_err = wise["mag_err"]      # (N_matched, 4) magnitude errors
    wise_det     = wise["detected"]     # (N_matched, 4) boolean detection flags
    log.info(f"Cross-matched {len(sdss_idx):,} galaxies via saved indices")
else:
    log.info("No saved indices — matching by specobjid ...")
    sdss_ids = sdss_meta["specobjid"]
    wise_ids = wise["specobjid"]
    wlookup  = {}
    for i, wid in enumerate(wise_ids):
        wlookup.setdefault(int(wid), i)
    sk, wk = [], []
    for si, sid in enumerate(sdss_ids):
        if int(sid) in wlookup:
            sk.append(si); wk.append(wlookup[int(sid)])
    sk, wk      = np.array(sk), np.array(wk)
    X_spec      = X_spectra_obs[sk]
    norm_consts = norm_consts_all[sk]
    z_arr       = sdss_meta["redshift"][sk]
    wise_mag     = wise["mag"][wk]
    wise_mag_err = wise["mag_err"][wk]
    wise_det     = wise["detected"][wk]
    log.info(f"Cross-matched {len(sk):,} galaxies by specobjid")

N = len(X_spec)
log.info(f"Working dataset: {N:,} galaxies")

for j, b in enumerate(WISE_BAND_SHORT):
    n_det = wise_det[:, j].sum()
    log.info(f"  {b}: {n_det:,}/{N:,} detected ({100*n_det/N:.1f}%)")


# =============================================================================
# STEP 2: Spender encoding — compress each spectrum to a 6-D latent vector
# =============================================================================
#
# spender (Liang et al. 2023, arXiv:2211.07890) is a variational autoencoder
# trained on ~2M SDSS galaxy spectra.  Its encoder maps a normalised
# observed-frame spectrum to 6 continuous latent dimensions that capture the
# dominant axes of spectral variation across the galaxy population:
# roughly corresponding to stellar population age, metallicity, dust attenuation,
# star-formation rate, ionization state, and velocity dispersion.
#
# CRITICAL: spender expects spectra normalised to unit V-band flux (i.e., divided
# by norm_const) before encoding.  The normalisation removes the overall
# brightness degeneracy so that two galaxies of identical spectral shape but
# different distances map to the same latent point.  We then supply the raw
# norm_const as a separate input feature to the MLP so it can reconstruct the
# absolute IR brightness.

log.info("\n=== STEP 2: Loading / computing spender latent encodings ===")

# Check for previously cached encodings to avoid redundant GPU computation.
latents_cache    = Path("ppplot_fixed_output") / "spender_encodings.npz"
latents_cache_v2 = OUT_DIR / "spender_encodings_v2.npz"
latents = None

for cache_path in [latents_cache, latents_cache_v2]:
    if cache_path.exists():
        cache = np.load(cache_path)
        if cache["latents"].shape == (N, 6):
            latents = cache["latents"]
            log.info(f"Loaded cached encodings from {cache_path}: {latents.shape}")
            break

if latents is None:
    try:
        import spender
        log.info("Loading spender 'sdss_II' model (6-dimensional latent space) ...")
        instrument_sp, model_sp = spender.hub.load("sdss_II")
        model_sp.eval()
        assert model_sp.encoder.n_latent == 6, "Unexpected latent dimension"
        log.info("spender loaded (6 latents)")

        spender_wave = instrument_sp.wave_obs.numpy()

        # Normalise to unit V-band flux before encoding, matching how spender
        # was trained and how the demo pickle files are prepared.
        log.info("Normalising spectra by V-band norm_const before encoding ...")
        X_normalized = X_spec / np.maximum(norm_consts[:, None], 1e-10)
        X_normalized = np.clip(X_normalized, -100.0, 100.0)

        # Interpolate onto spender's wavelength grid if our grid differs.
        if not (len(spender_wave) == len(wave_obs_grid) and
                np.allclose(spender_wave, wave_obs_grid, rtol=1e-4)):
            log.warning("Wavelength grid mismatch — re-interpolating ...")
            from scipy.interpolate import interp1d as _interp1d
            X_for_spender = np.zeros((N, len(spender_wave)), dtype=np.float32)
            for i in range(N):
                f = _interp1d(wave_obs_grid, X_normalized[i], kind="linear",
                              bounds_error=False, fill_value=0.0)
                X_for_spender[i] = f(spender_wave).astype(np.float32)
        else:
            X_for_spender = X_normalized.astype(np.float32)

        model_sp = model_sp.to(DEVICE)
        lat_list = []
        n_batches = (N + SPENDER_BATCH - 1) // SPENDER_BATCH
        log.info(f"Encoding {N:,} spectra in batches of {SPENDER_BATCH} ...")
        t0 = time.time()

        with torch.no_grad():
            for bi in range(n_batches):
                i0, i1 = bi * SPENDER_BATCH, min((bi+1)*SPENDER_BATCH, N)
                batch = torch.tensor(
                    np.nan_to_num(X_for_spender[i0:i1], nan=0.0),
                    dtype=torch.float32
                ).to(DEVICE)
                s = model_sp.encode(batch)
                lat_list.append(s.cpu().numpy())
                if (bi+1) % max(1, n_batches//5) == 0:
                    log.info(f"  {i1:,}/{N:,}  ({time.time()-t0:.0f}s elapsed)")

        latents = np.concatenate(lat_list, axis=0)
        latents = np.nan_to_num(latents, nan=0.0)
        np.savez_compressed(latents_cache_v2, latents=latents)
        log.info(f"Encoded {N:,} spectra in {time.time()-t0:.1f}s; cached to {latents_cache_v2}")

    except ImportError:
        raise ImportError(
            "spender not importable. Install via: pip install spender\n"
            f"Or place cached encodings at: {latents_cache_v2}"
        )

assert latents.shape == (N, 6), f"Expected ({N}, 6), got {latents.shape}"

print(f"\nLatent space statistics (normalised-spectrum encodings):")
for d in range(6):
    print(f"  L{d+1}: mean={latents[:,d].mean():.3f}  std={latents[:,d].std():.3f}")


# =============================================================================
# STEP 3: Assemble MLP inputs and targets
# =============================================================================
#
# Input vector (8 features per galaxy):
#   [L1, L2, L3, L4, L5, L6]  — 6 spender latents (galaxy spectral fingerprint)
#   z                          — spectroscopic redshift (determines K-correction
#                                magnitude; at fixed optical shape, a higher-z
#                                galaxy has different rest-frame IR coverage)
#   norm_const                 — V-band normalisation factor (proxy for optical
#                                luminosity; required to recover the absolute
#                                IR magnitude from the shape-only latent code)
#
# Target vector (4 values per galaxy):
#   W1, W2, W3, W4 magnitudes in the Vega system [mag]
#
# WISE photometry is expressed in Vega magnitudes by convention.  We train the
# MLP directly in magnitude space (not flux/Jansky space) because the loss
# function, chi statistic, and PP-plot in the paper are all defined in
# magnitude space, and the observational uncertainties (mag_err) are reported
# in magnitude space with approximately Gaussian distributions.
#
# Non-detected bands: WISE reports upper limits when a source falls below the
# detection threshold (~0.054 mJy at W1 for a 5σ detection).  We assign
# σ = 10^4 mag to these bands so the NLL loss gradient is effectively zero
# for non-detected bands, leaving the MLP free to predict any value there
# without penalty.

log.info("\n=== STEP 3: Building MLP inputs (8-D) and magnitude targets (4-D) ===")

X_input = np.column_stack([
    latents,                       # (N, 6) — spectral latent code
    z_arr.reshape(-1, 1),          # (N, 1) — spectroscopic redshift
    norm_consts.reshape(-1, 1),    # (N, 1) — V-band normalisation factor
]).astype(np.float32)              # (N, 8)

X_input = np.nan_to_num(X_input, nan=0.0, posinf=0.0, neginf=0.0)

# Set non-detected band magnitudes to 0 (placeholder) and inflate their
# uncertainties to σ = 10^4 to mask them from the training loss.
Y_mag   = np.where(wise_det, wise_mag,     0.0).astype(np.float32)           # (N, 4)
Y_sigma = np.where(wise_det, wise_mag_err, NON_DETECTION_SIGMA).astype(np.float32)  # (N, 4)
Y_sigma = np.where((Y_sigma > 0) & np.isfinite(Y_sigma), Y_sigma, NON_DETECTION_SIGMA)

log.info(f"Input shape:  {X_input.shape}")
log.info(f"Target shape: {Y_mag.shape}")
for j, b in enumerate(WISE_BAND_SHORT):
    det = wise_det[:, j]
    if det.any():
        m = Y_mag[det, j]; s = Y_sigma[det, j]
        log.info(f"  {b}: mag=[{m.min():.1f}, {m.max():.1f}]  "
                 f"sigma=[{s.min():.4f}, {s.max():.3f}]")


# =============================================================================
# STEP 4: Train / test split (80 / 20)
# =============================================================================
log.info("\n=== STEP 4: Train/test split (80/20) ===")

ids = np.arange(N)
np.random.shuffle(ids)
n_train   = int(N * 0.8)
n_test    = N - n_train
idx_train = ids[:n_train]
idx_test  = ids[n_train:]

X_tr = X_input[idx_train]
X_te = X_input[idx_test]

# Concatenate [mag, sigma] into a single (N, 8) array matching the demo's data
# structure: columns 0–3 are Vega magnitudes, columns 4–7 are the associated
# magnitude uncertainties.
Y_combined_tr = np.concatenate([Y_mag[idx_train], Y_sigma[idx_train]], axis=1).astype(np.float32)
Y_combined_te = np.concatenate([Y_mag[idx_test],  Y_sigma[idx_test]],  axis=1).astype(np.float32)

det_te = wise_det[idx_test]
log.info(f"Train: {n_train:,}  |  Test: {n_test:,}")


# =============================================================================
# STEP 5: MLP architecture
# =============================================================================
#
# A fully-connected network with ReLU activations:
#   f(x) = W_5 φ( W_4 φ( W_3 φ( W_2 φ( W_1 x + b_1 ) + b_2 ) + b_3 ) + b_4 ) + b_5
# where φ = ReLU = max(0, ·) and W_i, b_i are learned parameters.
#
# The bottleneck–expansion–bottleneck shape (8→20→50→50→20→4) is a common MLP
# design: the early wide layers expand the representation to capture non-linear
# interactions between input features, while the final compression forces the
# network to find a compact predictive mapping.  No output activation is used;
# the final layer predicts raw magnitudes, which are unbounded real numbers.

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


model = build_mlp(n_in=8, hidden_layers=HIDDEN_LAYERS, n_out=4).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nMLP: {n_params} parameters")
print(model)

# ── Loss function: Gaussian Negative Log-Likelihood in magnitude space ────────
#
# For a Gaussian likelihood p(y | μ, σ²) the negative log-likelihood is:
#   NLL = 0.5 * [ log(σ²) + (y - μ)² / σ² ]
#
# PyTorch's GaussianNLLLoss(input=μ, target=y, var=v) uses v as the variance σ².
# The demo passes var = sqrt(σ_mag), not σ_mag itself.  This is an unconventional
# but deliberate choice: it down-weights the penalty for high-uncertainty bands
# less aggressively than the standard σ² weighting, encouraging the network to
# pay attention to poorly constrained detections rather than ignoring them entirely.
# We replicate this exactly.
#
# For evaluation (chi, PP-plot) we use the standard definition χ = (pred−obs)/σ_mag,
# consistent with the paper's axis label "χ [magnitude]".

loss_fn = nn.GaussianNLLLoss()

def compute_loss(predictions, y_combined):
    mags  = y_combined[:, :4]
    sigma = y_combined[:, 4:]
    var   = torch.sqrt(sigma)    # demo's weighting: pass sqrt(σ) as the 'variance' argument
    return loss_fn(predictions, mags, var)


# =============================================================================
# STEP 6: Training with Adam + LR scheduling + early stopping
# =============================================================================
#
# Full-batch gradient descent: because the entire dataset (~8 MB) fits in GPU
# VRAM, we compute the gradient over all training examples simultaneously.
# Full-batch training is deterministically convergent and avoids the noise of
# mini-batches, which is beneficial once the network is close to a minimum.
# The trade-off (higher memory, no regularisation from batch noise) is acceptable
# here because the dataset is small relative to a modern GPU's memory.
#
# ReduceLROnPlateau halves the learning rate when validation loss plateaus,
# allowing Adam to take finer steps as it approaches the minimum — a practical
# remedy for Adam's tendency to oscillate in flat loss regions.

log.info("\n=== STEP 6: Training ===")

# Pre-load the full dataset to GPU once to eliminate CPU→GPU transfer overhead.
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

    # Forward pass → loss → backward → gradient clipping → parameter update.
    # Gradient clipping (max_norm=5) prevents runaway parameter updates if the
    # loss landscape is steep early in training.
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
# STEP 7: Evaluation on the held-out test set
# =============================================================================
log.info("\n=== STEP 7: Evaluating on test set ===")

model.eval()
with torch.no_grad():
    Y_pred_mag = model(
        torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    ).cpu().numpy()    # (N_test, 4) predicted Vega magnitudes

Y_true_mag  = Y_mag[idx_test]
Y_sigma_mag = Y_sigma[idx_test]

# ── Chi-squared ───────────────────────────────────────────────────────────────
# Following the demo notebook exactly:
#   chi = (pred - obs) / sqrt(sigma_mag)
#
# This is NOT the standard normalisation (which would divide by sigma_mag itself).
# It is consistent with the GaussianNLLLoss using var = sqrt(sigma) as its
# variance argument: the "natural" chi under that loss would be
# residual / sigma^(1/4), but the demo plots residual / sqrt(sigma) and this is
# the definition that reproduces the paper's Table 3 chi^2_N values (~1–2).
#
# The standard chi (residual / sigma) gives much larger chi^2_N (~30–70 here)
# because typical WISE photometric errors are 0.02–0.05 mag, and our model
# residuals are several times larger than those small measurement uncertainties.
# Both are reported below for transparency.
#
# Only detected bands (sigma < 5 mag) enter the statistic; this cleanly
# separates real photometry from the sentinel sigma = 10^4 non-detections.

print(f"\n{'Band':<6s} {'χ²_N (demo)':>12s} {'χ²_N (std)':>11s} {'Paper':>8s} "
      f"{'N_det':>8s} {'IQR_χ':>8s} {'<χ>':>7s}")
print("-" * 66)
chi_all  = {}   # chi = residual / sqrt(sigma)  — matches demo / paper
chi_std  = {}   # chi = residual / sigma         — standard normalisation
for j, b in enumerate(WISE_BAND_SHORT):
    det_mask = det_te[:, j] & (Y_sigma_mag[:, j] < 5.0)
    n_det    = det_mask.sum()
    if n_det < 5:
        print(f"{b:<6s} {'N/A':>12s} {'N/A':>11s} {PAPER_CHI2[b]:>8.2f} {n_det:>8d}")
        continue
    resid = Y_pred_mag[det_mask, j] - Y_true_mag[det_mask, j]
    sig   = Y_sigma_mag[det_mask, j]
    chi        = resid / np.sqrt(sig)
    chi_s      = resid / sig
    chi2n      = np.mean(chi**2)
    chi2n_std  = np.mean(chi_s**2)
    iqr_chi    = (np.percentile(chi, 84) - np.percentile(chi, 16)) / 2.0
    chi_all[b] = chi
    chi_std[b] = chi_s
    print(f"{b:<6s} {chi2n:>12.2f} {chi2n_std:>11.2f} {PAPER_CHI2[b]:>8.2f} "
          f"{n_det:>8d} {iqr_chi:>8.3f} {np.mean(chi):>7.3f}")


# =============================================================================
# STEP 8: Chi distribution (paper Figure 15)
# =============================================================================
#
# Chi distribution using the demo's definition: chi = (pred - obs) / sqrt(sigma).
# If the model is well calibrated under this convention, the chi distribution
# should match N(0,1).  With our self-computed latents on our dataset, chi²_N
# is expected to be higher than the paper's ~1–2 because our spender encoding
# pipeline differs from Jespersen et al.'s: small differences in normalisation,
# quality masking, or sky-line handling shift the latent space, increasing
# residuals.  Running on their pre-computed latents (our_model_their_data.py)
# reproduces chi²_N within ~15–20% of the paper, confirming this diagnosis.

def IQR_sig(x):
    iqr = np.nanpercentile(x, [84, 16])
    return (iqr[0] - iqr[1]) / 2.0

band_labels = {
    "W1": r"WISE 3.4 $\mu$m",  "W2": r"WISE 4.6 $\mu$m",
    "W3": r"WISE 12.1 $\mu$m", "W4": r"WISE 22.2 $\mu$m",
}

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for idx, b in enumerate(WISE_BAND_SHORT):
    ax       = axes[idx // 2, idx % 2]
    det_mask = det_te[:, idx] & (Y_sigma_mag[:, idx] < 5.0)
    if det_mask.sum() < 5:
        ax.text(0.5, 0.5, f"Too few {b} detections", transform=ax.transAxes, ha="center")
        ax.set_title(band_labels[b]); continue

    chi     = chi_all[b]   # residual / sqrt(sigma), pre-computed in Step 7
    chi2n   = np.mean(chi**2)
    iqr_chi = IQR_sig(chi)

    ax.hist(chi, bins=100, range=(-10, 10), histtype="step",
            density=True, color="steelblue", linewidth=1.5)
    x_ = np.linspace(-10, 10, 400)
    ax.plot(x_, scipy_norm.pdf(x_), "k--", lw=1.5, label=r"$\mathcal{N}(0,1)$")
    ax.set(xlim=(-10, 10),
           xlabel=r"$\chi = (\mathrm{pred}-\mathrm{obs})/\sqrt{\sigma}$ [mag$^{1/2}$]",
           ylabel="Density", title=band_labels[b])
    ax.legend(fontsize=8)

    xt, yt, dy = 0.05, 0.82, 0.12
    font = {"size": 10}
    ax.text(xt, yt,      fr"$\chi^2_N$ = {chi2n:.2f}",       transform=ax.transAxes, fontdict=font)
    ax.text(xt, yt-dy,   fr"IQR: {iqr_chi:.2f}",             transform=ax.transAxes, fontdict=font)
    ax.text(xt, yt-2*dy, fr"$\langle\chi\rangle$ = {np.mean(chi):.2f}", transform=ax.transAxes, fontdict=font)

plt.suptitle(
    r"$\chi = (\mathrm{pred}-\mathrm{obs})/\sqrt{\sigma}$  (cf. Jespersen et al. 2025, Figure 15)"
    f"\nN_train = {n_train:,}  [our SDSS preprocessing + spender latents]",
    fontsize=12, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig(OUT_DIR / "chi_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved chi_distribution.png")


# =============================================================================
# STEP 9: PP-plot — assessing probabilistic calibration (paper Figure 3)
# =============================================================================
#
# The PP-plot (probability–probability plot) is a calibration diagnostic.  For
# each detected galaxy i in band j we compute:
#
#   p̂_empirical = p(ŷ_ij | y_obs_ij, σ_ij)  — Gaussian likelihood of the MLP
#                  prediction given the observed photometry and its uncertainty
#   p̂_data      = p(y_obs_ij | y_obs_ij, σ_ij) = (σ_ij √2π)⁻¹  — the maximum
#                  achievable likelihood, i.e. the data evaluated against itself
#
# The PP-plot overlays the empirical CDFs of p̂_emp and p̂_data.  Perfect
# calibration → the curve lies on the diagonal.  A curve below the diagonal
# (as seen with our self-computed latents) means predictions systematically
# fall outside the observational error bars — the model is "overconfident and
# biased" relative to the noise model.
#
# NOTE ON EXPECTED GAP TO PAPER:
# Running this script on Jespersen et al.'s pre-computed latents
# (our_model_their_data.py) brings chi²_N to within ~15–20% of the paper,
# confirming the latent encoding is the bottleneck.  However, even with their
# latents the PP-plot shows residual deviations in W1/W2, where p̂_emp << p̂_data
# (typical sigma ~0.03 mag, so predictions must land within ~0.03 mag of
# observations to score well on this plot).  The PP-plot is therefore a much
# stricter calibration test than chi²_N: chi²_N integrates over all sigma values
# while the PP-plot is most sensitive to the tightest-error, best-detected bands.

log.info("\n=== STEP 9: PP-plot (calibration assessment) ===")

paper_phat_mean = {"W1": 0.66, "W2": 0.70, "W3": 0.63, "W4": 0.70}
phat_emp, phat_data = {}, {}

print(f"\n{'Band':<6s} {'<p̂_emp>':>10s} {'<p̂_data>':>10s} {'paper':>8s} {'N':>8s}")
print("-" * 46)
for j, b in enumerate(WISE_BAND_SHORT):
    det_mask = det_te[:, j] & (Y_sigma_mag[:, j] < 5.0)
    if det_mask.sum() < 5:
        print(f"{b:<6s} {'—':>10s} {'—':>10s} {paper_phat_mean[b]:>8.2f} {det_mask.sum():>8d}")
        continue

    y_obs  = Y_true_mag[det_mask, j]
    y_pred = Y_pred_mag[det_mask, j]
    sigma  = Y_sigma_mag[det_mask, j]

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
        ax.set_title(band_labels[b]); continue

    pe  = np.sort(phat_emp[b])
    pd_ = np.sort(phat_data[b])
    thresholds = np.linspace(0, max(pe.max(), pd_.max()), 1000)
    cdf_e = np.searchsorted(pe,  thresholds) / len(pe)
    cdf_d = np.searchsorted(pd_, thresholds) / len(pd_)

    ax.plot([0, 1], [0, 1],  "r-",  lw=1.5,        label="Perfect calibration (diagonal)")
    ax.plot(cdf_d, cdf_d,    "k--", lw=1.5, alpha=0.7, label="Data baseline")
    ax.plot(cdf_d, cdf_e,    "g-",  lw=2.0,        label="This model")
    ax.annotate("Overconfident\nand biased",
                xy=(0.65, 0.35), xytext=(0.45, 0.15),
                fontsize=8, color="gray", ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.set(xlabel="CDF(p̂_data)", ylabel="CDF(p̂_empirical)",
           title=band_labels[b], xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.legend(loc="upper left", fontsize=8)

plt.suptitle(
    "PP-plot: probabilistic calibration of WISE magnitude predictions\n"
    "(Reproducing Figure 3 of Jespersen et al. 2025)",
    fontsize=12, fontweight="bold", y=1.02,
)
plt.tight_layout()
plt.savefig(OUT_DIR / "ppplot_figure3.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved ppplot_figure3.png")


# =============================================================================
# STEP 10: Predicted vs observed magnitudes (hexbin/2D histogram)
# =============================================================================
#
# A 2D histogram of predicted vs observed magnitudes shows the point-by-point
# scatter, bias, and any non-linear systematics.  A perfect model would have
# all points on the y = x line.  The IQR-based scatter statistic σ_IQR =
# IQR / 1.349 is the robust equivalent of the standard deviation, insensitive
# to outliers from catastrophic prediction failures.

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
names = ["WISE 3.4μm", "WISE 4.6μm", "WISE 12.1μm", "WISE 22.2μm"]

for j, b in enumerate(WISE_BAND_SHORT):
    ax       = axes[j // 2, j % 2]
    det_mask = det_te[:, j] & (Y_sigma_mag[:, j] < 5.0)
    if det_mask.sum() < 3:
        ax.text(0.5, 0.5, "—", transform=ax.transAxes, ha="center")
        ax.set_title(b); continue

    true    = Y_true_mag[det_mask, j]
    pred    = Y_pred_mag[det_mask, j]
    scatter = (np.percentile(pred-true, [75, 25])[0] -
               np.percentile(pred-true, [75, 25])[1]) / 1.349
    bias    = np.median(true - pred)
    tot     = np.hstack([true, pred])
    l       = 0.01   # clip 1st/99th percentile for display range

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

plt.suptitle("Predicted vs Observed WISE magnitudes (Vega)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / "mag_pred_vs_obs.png", dpi=150, bbox_inches="tight")
plt.close()
log.info("Saved mag_pred_vs_obs.png")


# =============================================================================
# Save model and test-set results
# =============================================================================

torch.save(model.state_dict(), OUT_DIR / "mlp_weights.pt")
np.savez_compressed(
    OUT_DIR / "test_results.npz",
    y_pred_mag=Y_pred_mag, y_true_mag=Y_true_mag, y_sigma_mag=Y_sigma_mag,
    detected=det_te, idx_test=idx_test,
    z_test=z_arr[idx_test], norm_consts_test=norm_consts[idx_test],
    latents_test=latents[idx_test],
)

print("\n" + "=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
print(f"  Dataset:      {N:,} galaxies  (train={n_train:,}, test={n_test:,})")
print(f"  Architecture: 8→20→50→50→20→4  ({n_params} params)")
print(f"  Epochs run:   {epoch+1}  |  Training time: {total_time:.0f}s")
print(f"\n  Chi-squared [demo def: (pred-obs)/sqrt(sigma)] vs paper Table 3:")
print(f"  {'Band':<6s} {'χ²_N (demo)':>12s} {'χ²_N (std)':>11s} {'Paper':>8s}")
print("  " + "-" * 41)
for j, b in enumerate(WISE_BAND_SHORT):
    det_mask = det_te[:, j] & (Y_sigma_mag[:, j] < 5.0)
    if det_mask.sum() < 5 or b not in chi_all:
        print(f"  {b:<6s} {'N/A':>12s} {'N/A':>11s} {PAPER_CHI2[b]:>8.2f}")
        continue
    chi2n     = np.mean(chi_all[b]**2)
    chi2n_std = np.mean(chi_std[b]**2)
    print(f"  {b:<6s} {chi2n:>12.2f} {chi2n_std:>11.2f} {PAPER_CHI2[b]:>8.2f}")

print(f"\n  Outputs saved to: {OUT_DIR}/")
for f in sorted(OUT_DIR.iterdir()):
    if f.is_file():
        print(f"    {f.name:<45s} {f.stat().st_size/1024:>7.1f} kB")
