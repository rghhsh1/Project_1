# Storage time prediction of shrimp using two-trace two-dimensional correlation spectroscopy and an attention-integrated liquid time-constant network

This repository contains the source code and a representative subset of the spectral data used in the manuscript:

> **Storage time prediction of shrimp using two-trace two-dimensional correlation spectroscopy combined with an attention-integrated liquid time-constant network**
> *Journal of Food Measurement and Characterization*

---

## Overview

The pipeline implemented here follows the workflow described in the manuscript:

1. **Acquisition** — Raman spectra of intact shrimp, 〈306〉 spectral points per measurement, three measurement positions per shrimp.
2. **Preprocessing** — 〈Savitzky–Golay smoothing, baseline correction, min–max normalization〉.
3. **2T2D correlation analysis** — construction of the synchronous and asynchronous two-trace two-dimensional correlation spectra from each spectrum and a pre-registered reference spectrum.
4. **Feature sequence construction** — extraction of horizontal slice spectra along the vertical band coordinate, forming a sequential feature representation.
5. **Modelling** — an attention-integrated liquid time-constant network (LTC-Attn) for storage time regression.
6. **Interpretation** — SHAP analysis of feature contributions.

## Repository structure

```
.
├── data/
│   └── example/          # representative spectral subset (see Data)
├── preprocessing/        # smoothing, baseline correction, normalization
├── twotwod/              # 2T2D correlation spectra and slice extraction
├── model/                # LTC-Attn definition and training loop
├── examples/
│   └── run_demo.py       # end-to-end demo on the example data
├── requirements.txt
└── README.md
```

## Environment

- Python 〈3.10〉
- PyTorch 〈2.x〉, NumPy, SciPy, pandas, scikit-learn, openpyxl

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Data

`data/example/` contains a representative subset of the measured spectra — 〈e.g. two storage days, ten shrimp per day〉 — sufficient to run the full pipeline end to end. The complete dataset is not included in this repository (see *Data availability*).

**File format** (`spectra_example.csv`):

| Column(s) | Content |
|---|---|
| `raman_shift` | Raman shift (cm⁻¹), common to all measurements |
| `shrimp_<id>_<pos>` | Intensity of one measurement; `<id>` = shrimp identifier, `<pos>` = measurement position (〈head / middle / tail〉) |

〈Adjust the table above to match the actual columns of your example file.〉

## Quick start

```bash
git clone https://github.com/〈username〉/〈repository〉.git
cd 〈repository〉
pip install -r requirements.txt
python examples/run_demo.py
```

The demo loads the example data, builds the 2T2D feature sequences, trains LTC-Attn for 〈a small number of〉 epochs and reports 〈MAE, RMSE and R²〉 on the held-out split.

## Reproducing the results in the manuscript

〈Describe the exact settings used for the reported results: training/validation/test split, number of seeds, learning rate, batch size, number of epochs, and the hardware used. State explicitly whether the demo reproduces the reported numbers exactly or only illustrates the pipeline.〉

## Data availability

The complete spectral dataset and the Android application are available from the corresponding author upon reasonable request. Public release of the full dataset is restricted by institutional intellectual property constraints associated with the application and the device integration.

## License

- Code: 〈MIT License〉
- Data in `data/example/`: 〈CC BY-NC 4.0〉

## Citation

〈BibTeX entry once the paper is published.〉

## Contact

〈Corresponding author name and e-mail〉
