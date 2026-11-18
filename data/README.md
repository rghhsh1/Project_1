# Data
 
Representative subset of the Raman spectra used in:

> **Storage time prediction of shrimp using two-trace two-dimensional correlation spectroscopy combined with an attention-integrated liquid time-constant network**

## Layout

```
data/
├── train/  day1.xlsx … day7.xlsx   batch 1, shrimp 1–80
├── val/    day1.xlsx … day7.xlsx   batch 1, shrimp 81–100
└── test/   day1.xlsx … day7.xlsx   batch 2, independent batch, shrimp 1–30
```

Each file holds one storage day. The shrimp ranges above describe the complete
dataset; this repository ships a reduced subset with exactly the same layout
(〈N〉 shrimp per day in `train`, 〈M〉 in `val` and `test`).

The file names must keep the `dayN.xlsx` pattern — the scripts discover the
available storage days from them.

## File format

- Sheet name: `Sheet1`
- First row: header
- Column A: `raman_shift` — the Raman shifts of the full range exported by the
  instrument (83–2182 cm⁻¹ for the example files)
- Columns B onwards: one spectrum per shrimp, named `s001`, `s002`, …

| Property | Value |
|---|---|
| Rows | header + one row per spectral point of the export |
| Columns | 1 + number of shrimp |
| Content | raw intensity, no preprocessing applied |
| File format | `.xlsx` |

## Band selection

The files store the complete spectrum as exported by the instrument, without
any truncation. The modelling band is 600–1800 cm⁻¹: `2T2D.py` selects it
through `BAND_RANGE`, which leaves the 306 spectral points used by the model.
Savitzky–Golay smoothing, MSC and global scaling are then applied to that band,
so the source files can be reprocessed with a different range or different
parameters without being edited.

## Notes

- Each spectrum is the average of three measurements on the same shrimp (head,
  middle and tail), so one column corresponds to one shrimp on one day.
- `train` and `val` are drawn from the first batch; `test` is an independent
  second batch acquired separately and used to assess generalisation across
  batches.
