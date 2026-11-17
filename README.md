# Storage time prediction of shrimp using 2T2D correlation spectroscopy and LTC-Attn

Code and example data for the manuscript *"Storage time prediction of shrimp using
two-trace two-dimensional correlation spectroscopy combined with an attention-integrated
liquid time-constant network"* (Journal of Food Measurement and Characterization).

## Contents

- `preprocessing/` — spectral preprocessing (smoothing, baseline, normalization)
- `2T2D/` — 2T2D correlation spectra and horizontal slice extraction
- `models/` — networks
- `data/example/` — representative subset of the measured spectra

## Requirements

Python 3.10, PyTorch 2.x

```bash
pip install -r requirements.txt

## Quick start
git clone https://github.com/<username>/<repository>.git
cd <repository>
pip install -r requirements.txt
python examples/run_demo.py

## data
data/example/spectra_example.csv — the first column is raman_shift; each remaining column is one measured spectrum named <shrimp_id>_<position>. The complete dataset and the Android application are available from the corresponding author upon reasonable request.
