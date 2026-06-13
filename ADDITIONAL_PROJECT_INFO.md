# PROJECT INFO

This file contains additional information about the project, but does not always need to be in the context window.

## Commands

### Build the book
```bash
cd docs
myst build --html
```

### Preview locally (live reload)
```bash
cd docs
myst start
```

### Run a notebook manually
```bash
jupyter lab docs/TCN.ipynb
```

### Preprocess dataset (one-time setup)
The raw Gu et al. 2024 SSVEP dataset must be downloaded separately and placed in `dataset/Data/`. To downsample from 1000 Hz to 250 Hz:
```bash
python utils/downsample_Gu_et_al.py
```

### Data format
All notebooks load `dataset/Data/data_s1_64_down.npy` (subject 1, 64-channel, downsampled to 250 Hz). Shape is `(blocks, stim_frequencies, time, channels, conditions)` — 5 blocks, 60 stim frequencies (1–60 Hz), 1285 time points (~5.14 s at 250 Hz), 64 EEG channels, 2 conditions (luminance modulation depths).

SSVEP channels of interest are the posterior/occipital electrodes: `['Pz', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'O1', 'Oz', 'O2']` at indices `[48,54,55,56,57,58,61,62,63]`.

### Key utility: `utils/utils.py`
- `OptimalSVHT` — Gavish & Donoho (2014) optimal SVD hard thresholding, used in DMD
- `multivariate_stationarity_check` — ADF/KPSS tests across all channels
- `nth_order_difference` / `invert_nth_order_difference` — differencing for VARIMA stationarity
- `customDataSet` — PyTorch `Dataset` wrapper for EEG trial arrays
- `sliding_window_extraction` — extracts overlapping windows from `(N, T, C)` trial arrays

### Python stack
- **Statistical models:** `statsmodels` (VARIMA via `VARMAX`, `VAR`)
- **Deep learning:** `PyTorch` — TCN, GRU, Transformer, S6 are all implemented from scratch as `nn.Module` subclasses; training targets `cuda`
- **Numerics/signal:** `numpy`, `scipy.linalg` (SVD, eigendecomposition for DMD), `scikit-learn` (PCA, StandardScaler)
- **Book:** `mystmd` — book config is `docs/myst.yml`; chapters are registered in the `toc` section there

### Notebook conventions
- Data is always normalized using training-set mean/std before model input, then denormalized for evaluation/plotting
- Forecasting is framed as: given the first `T - HORIZON` time points, predict the last `HORIZON` time points
- Deep learning models use noise augmentation during training (Gaussian noise added to input)
- The DMD chapter imports `from utils import OptimalSVHT` — this assumes the working directory contains the `utils/` folder (i.e., run notebooks from the repo root)