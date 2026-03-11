# Predicting WISE IR Photometry from SDSS Optical Spectra

Reproduction and extension of Jespersen et al. (2025), which demonstrates that
mid-infrared (WISE W1-W4) photometry of galaxies can be predicted directly from
their optical spectra via a shallow MLP trained on compressed spectral
representations.

**Reference paper**: Jespersen et al. (2025), "Predicting infrared photometry
from optical spectra using machine learning"


## Scientific background

The central question is whether the mid-infrared emission of a galaxy (which
encodes star formation rate, dust content, and AGN activity) is sufficiently
constrained by its optical spectrum that a small neural network can bridge the
two. The approach:

1. Each SDSS galaxy spectrum is compressed to a 6-dimensional latent vector
   using **spender** (Liang et al. 2023, arXiv:2211.07890), a variational
   autoencoder pre-trained on ~2M SDSS galaxy spectra. The latent dimensions
   capture the dominant axes of spectral variation (stellar age, metallicity,
   dust attenuation, star-formation rate, ionization state, velocity
   dispersion).

2. A shallow **MLP** (8 → 20 → 50 → 50 → 20 → 4, 4,884 parameters) maps the
   6 spender latents plus spectroscopic redshift and V-band normalization
   constant to predicted WISE Vega magnitudes in all four bands (W1 3.4 μm,
   W2 4.6 μm, W3 12.1 μm, W4 22.2 μm).

3. Training uses Gaussian NLL loss in magnitude space, full-batch gradient
   descent (Adam + ReduceLROnPlateau), and early stopping on a held-out 20%
   test set. Non-detected WISE bands are masked by inflating their uncertainty
   to sigma = 10^4.

4. Calibration is assessed via chi-squared and PP-plots following the paper's
   Figure 3 and Figure 15 conventions.


## Repository layout

```
.
├── train_our_data.py          # Main training script using our SDSS/WISE preprocessing
├── train_their_data.py        # Same pipeline on Jespersen et al.'s pre-computed data
├── install.sh                 # Conda environment setup
├── run_jupyter_notebook.sh    # Convenience script for launching Jupyter on the cluster
│
├── notebooks/
│   ├── data_download.ipynb            # Download SDSS spectra and WISE photometry
│   ├── sdss_data_preprocessing.ipynb  # Preprocess SDSS spectra → data/sdss_output/
│   └── wise_data_preprocessing.ipynb  # Crossmatch and preprocess WISE → data/wise_output/
│
├── data/                      # Gitignored. Populated by the notebooks.
│   ├── sdss_raw/              # Raw SDSS spectral files
│   ├── sdss_raw2/             # Additional raw SDSS downloads
│   ├── wise_raw/              # Raw WISE crossmatch files
│   ├── sdss_output/           # Preprocessed SDSS arrays (spectra, metadata, etc.)
│   └── wise_output/           # Preprocessed WISE photometry (.npz)
│
├── external/
│   └── IR_optical_demo/       # Jespersen et al.'s demo notebook and CSV data files
│                              # (sdss_wise_cross_reg0.csv, sdss_wise_cross_reg1.csv)
│
├── results/                   # Gitignored. Populated by training scripts.
│   ├── our_data/              # Outputs from train_our_data.py
│   └── their_data/            # Outputs from train_their_data.py
│
├── logs/                      # Gitignored. Run logs.
│   ├── our_data_run.log       # Log from the baseline run on our preprocessed data
│   └── their_data_run.log     # Log from the baseline run on Jespersen et al.'s data
│
└── deprecated/                # Old development scripts and intermediate results.
    ├── reproduce_figure3_v1.py
    ├── reproduce_figure3_v2.py
    ├── reproduce_figure3_v2_huge_batch.py
    ├── reproduce_figure3_ppplot.ipynb
    └── old_results/
```


## Workflow

Run these steps in order on a fresh checkout (see `install.sh` for the
conda environment).

### Step 1: Data acquisition

Open `notebooks/data_download.ipynb` and run all cells. This downloads:
- SDSS DR16 galaxy spectra from the SDSS SAS via `astroquery`
- WISE photometry from the AllWISE catalog via CDS/VizieR crossmatch

### Step 2: SDSS preprocessing

Open `notebooks/sdss_data_preprocessing.ipynb` and run all cells. Outputs
written to `data/sdss_output/`:
- `spectra_obs_frame.npy` — observed-frame flux arrays (N_gal, 3921)
- `norm_consts.npy` — per-spectrum V-band normalization constants
- `metadata.npy` — structured array with specobjid, redshift, etc.
- `wave_obs_grid.npy` — the shared observed-frame wavelength grid

### Step 3: WISE preprocessing

Open `notebooks/wise_data_preprocessing.ipynb` and run all cells. Outputs
written to `data/wise_output/`:
- `wise_photometry.npz` — Vega magnitudes, errors, and detection flags
- `wise_matched_sdss_indices.npy` — indices into the SDSS array for
  cross-matched galaxies

### Step 4: Train on our preprocessed data

```bash
python train_our_data.py
```

This runs the full pipeline: spender encoding (cached to
`results/our_data/spender_encodings.npz` after the first run), MLP training,
and evaluation. Outputs written to `results/our_data/`:
- `training_curve.png`
- `chi_distribution.png`
- `ppplot_figure3.png`
- `mag_pred_vs_obs.png`
- `mlp_weights.pt`
- `test_results.npz`

**Expected results** (from `logs/our_data_run.log`, 238,839 galaxies):

| Band | chi²_N (ours) | chi²_N (paper) |
|------|--------------|---------------|
| W1   | 1.93         | 1.33          |
| W2   | 1.75         | 1.17          |
| W3   | 2.97         | 2.23          |
| W4   | 2.10         | 1.41          |

The ~40-50% excess in chi²_N relative to the paper arises from differences in
spender encoding: small variations in SDSS preprocessing (sky-line masking,
normalization windowing, quality cuts) shift the latent space and inflate
prediction residuals. This was confirmed by Step 5.

### Step 5: Train on Jespersen et al.'s pre-computed data

```bash
python train_their_data.py
```

Uses the pre-computed latents distributed in
`external/IR_optical_demo/data/sdss_wise_cross_reg0.csv` and
`sdss_wise_cross_reg1.csv` (combined ~764K galaxies), bypassing the spender
encoding step entirely. Everything else — MLP, loss, training schedule,
evaluation — is identical to `train_our_data.py`. Outputs go to
`results/their_data/`.

**Expected results** (from `logs/their_data_run.log`, 764,151 galaxies):

| Band | chi²_N (ours) | chi²_N (paper) |
|------|--------------|---------------|
| W1   | 1.53         | 1.33          |
| W2   | 1.35         | 1.17          |
| W3   | 2.54         | 2.23          |
| W4   | 1.79         | 1.41          |

This brings chi²_N to within ~15-20% of the paper, confirming that the latent
encoding quality (not the MLP or training procedure) is the main gap when using
our own preprocessing.


## Key implementation notes

**Chi definition**: Following the Jespersen et al. demo notebook, chi is
defined as `(pred - obs) / sqrt(sigma_mag)`, not the standard
`(pred - obs) / sigma_mag`. This is consistent with the GaussianNLLLoss using
`var = sqrt(sigma)` as its variance argument. The paper's Table 3 chi²_N
values correspond to this convention.

**Non-detections**: WISE bands below the detection threshold are assigned
sigma = 10^4 mag during training, making the NLL loss gradient effectively
zero for those bands.

**Full-batch training**: With datasets of 200K-800K galaxies (< 30 MB), the
entire dataset fits in GPU VRAM. Full-batch gradient steps eliminate
mini-batch noise and converge cleanly.

**spender cache**: The first run of `train_our_data.py` computes spender
encodings for all galaxies (a few minutes on GPU) and caches the result to
`results/our_data/spender_encodings.npz`. Subsequent runs load from cache.


## Environment setup

```bash
bash install.sh
conda activate astronomy
```

Requires: numpy, scipy, matplotlib, astropy, astroquery, PyTorch (CUDA),
spender (`pip install spender`), pandas.


## Planned extensions

The following investigations are planned as natural next steps from this
baseline:

- **Ablations on the input representation**: vary the number of latent
  dimensions, exclude redshift or norm_const from the input, test PCA-compressed
  spectra instead of spender latents.
- **Raw spectra as input**: replace the spender encoder with a 1D CNN or
  transformer operating directly on the observed-frame flux array, eliminating
  the spender bottleneck and allowing end-to-end training.
- **Architecture ablations**: vary MLP depth and width, add batch normalization,
  test residual connections.
- **Uncertainty calibration**: replace the fixed NLL loss with a
  heteroscedastic output head that predicts both mean and variance per band.
- **Multi-survey generalization**: test on galaxies with DESI spectra or
  photometric redshifts in place of SDSS spectra.
