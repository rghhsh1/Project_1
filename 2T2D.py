"""Generate two-trace two-dimensional (2T2D) correlation matrices for the
shrimp storage-time dataset.

Dataset layout
--------------
::

    data/
    ├── train/  day1.xlsx ... day7.xlsx    80 spectra per day  (batch 1, shrimp 1-80)
    ├── val/    day1.xlsx ... day7.xlsx    20 spectra per day  (batch 1, shrimp 81-100)
    └── test/   day1.xlsx ... day7.xlsx    30 spectra per day  (batch 2, shrimp 1-30)

Every Excel file holds one storage day: the first column is the Raman shift and
each remaining column is the spectrum of one shrimp, already averaged over the
three measurement positions. Rows of one file must share the same Raman shift.

The number of days is discovered from the files, so the same code runs on a
reduced example subset shipped with the repository.

References
----------
Both the MSC reference spectrum and the 2T2D reference spectrum come from the
first storage day of the training set, and the global scaling factor is
estimated on the training set only. Validation and test spectra are therefore
transformed with statistics that never saw them.

Usage
-----
::

    python 2T2D.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from preprocessing import global_max_scale, msc, savitzky_golay_smoothing

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"
OUTPUT_DIR = ROOT / "outputs" / "2t2d"

SHEET_NAME = "Sheet1"
BAND_RANGE = (600.0, 1800.0)

#: Expected number of shrimp per day in the full dataset. Only reported, never
#: enforced, so that the example subset can be processed as well.
EXPECTED_COLUMNS = {"train": 80, "val": 20, "test": 30}

DAY_PATTERN = re.compile(r"^day(\d+)\.xlsx$", re.IGNORECASE)

DayBlock = Tuple[int, np.ndarray]  # (day label, spectra of shape (n_points, n_shrimp))


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def read_day_file(
    path: Path,
    band_range: Tuple[float, float] = BAND_RANGE,
    sheet_name: str = SHEET_NAME,
) -> Tuple[np.ndarray, np.ndarray]:
    """Read one storage-day Excel file.

    Returns
    -------
    shift
        Raman shift of the retained band, shape ``(n_points,)``.
    spectra
        Intensities, shape ``(n_points, n_spectra)``.
    """
    if not path.exists():
        raise FileNotFoundError(f"missing input file: {path}")

    frame = pd.read_excel(path, sheet_name=sheet_name, header=0)
    values = frame.to_numpy(dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError(f"{path}: expected at least two columns, got {values.shape}")

    shift = values[:, 0]
    spectra = values[:, 1:]

    low, high = band_range
    mask = (shift >= low) & (shift <= high)
    if not mask.any():
        raise ValueError(
            f"{path}: no Raman shift within {band_range}; "
            f"file spans {shift.min():.1f}-{shift.max():.1f}"
        )
    return shift[mask], spectra[mask, :]


def discover_days(folder: Path) -> List[Tuple[int, Path]]:
    """Return ``(day label, path)`` for every ``dayN.xlsx`` in ``folder``."""
    if not folder.exists():
        raise FileNotFoundError(
            f"missing data folder: {folder}\n"
            "expected data/train, data/val and data/test with dayN.xlsx files"
        )
    found = []
    for path in folder.iterdir():
        match = DAY_PATTERN.match(path.name)
        if match:
            found.append((int(match.group(1)), path))
    if not found:
        raise FileNotFoundError(f"no dayN.xlsx files found in {folder}")
    return sorted(found, key=lambda item: item[0])


def load_split(folder: Path) -> Tuple[np.ndarray, List[DayBlock]]:
    """Load every storage day of one split.

    Returns the shared Raman shift and a list of ``(day label, spectra)`` pairs
    ordered by day.
    """
    shift = None
    blocks: List[DayBlock] = []
    for day, path in discover_days(folder):
        day_shift, spectra = read_day_file(path)
        if shift is None:
            shift = day_shift
        elif day_shift.shape != shift.shape or not np.allclose(day_shift, shift):
            raise ValueError(f"{path}: Raman shift differs from the other files")
        blocks.append((day, spectra))
    assert shift is not None
    return shift, blocks


# --------------------------------------------------------------------------- #
# 2T2D
# --------------------------------------------------------------------------- #

def compute_2t2d(
    reference: np.ndarray, sample: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Synchronous and asynchronous 2T2D correlation matrices of two spectra."""
    synchronous = 0.5 * (np.outer(reference, reference) + np.outer(sample, sample))
    asynchronous = 0.5 * (np.outer(sample, reference) - np.outer(reference, sample))
    return synchronous, asynchronous


def make_transform(reference: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
    """Return a function applying Savitzky-Golay smoothing and MSC."""

    def transform(spectra: np.ndarray) -> np.ndarray:
        return msc(savitzky_golay_smoothing(spectra), reference=reference)

    return transform


def save_2t2d(
    blocks: Sequence[DayBlock],
    split_name: str,
    reference: np.ndarray,
    output_dir: Path = OUTPUT_DIR,
) -> List[dict]:
    """Write the 2T2D matrices of one split to disk and return the manifest."""
    entries: List[dict] = []
    for day, spectra in blocks:
        day_dir = output_dir / split_name / f"day{day}"
        day_dir.mkdir(parents=True, exist_ok=True)

        for sample_index in range(spectra.shape[1]):
            sample_id = f"sample_{sample_index + 1:03d}"
            synchronous, asynchronous = compute_2t2d(reference, spectra[:, sample_index])

            sync_path = day_dir / f"{sample_id}_sync.npy"
            async_path = day_dir / f"{sample_id}_async.npy"
            np.save(sync_path, synchronous)
            np.save(async_path, asynchronous)

            entries.append(
                {
                    "split": split_name,
                    "day": day,
                    "sample": sample_index + 1,
                    "sync": str(sync_path.relative_to(output_dir)),
                    "async": str(async_path.relative_to(output_dir)),
                }
            )
        print(f"  day {day}: {spectra.shape[1]} samples -> {day_dir}")
    return entries


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 2T2D matrices.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"where to write the matrices (default: {OUTPUT_DIR})",
    )
    args = parser.parse_args()

    train_shift, train_blocks = load_split(TRAIN_DIR)
    _, val_blocks = load_split(VAL_DIR)
    test_shift, test_blocks = load_split(TEST_DIR)

    if not np.allclose(train_shift, test_shift):
        raise ValueError("train and test sets do not share the same Raman shift grid")

    train_days = [day for day, _ in train_blocks]
    if [day for day, _ in val_blocks] != train_days:
        raise ValueError("train and val sets cover different storage days")
    if [day for day, _ in test_blocks] != train_days:
        raise ValueError("train and test sets cover different storage days")

    for name, blocks in (("train", train_blocks), ("val", val_blocks), ("test", test_blocks)):
        counts = {block.shape[1] for _, block in blocks}
        expected = EXPECTED_COLUMNS[name]
        note = "" if counts == {expected} else f" (full dataset expects {expected})"
        print(f"{name}: {len(blocks)} days, {sorted(counts)} spectra per day{note}")

    # --- preprocessing statistics: training set only -----------------------
    reference = savitzky_golay_smoothing(train_blocks[0][1]).mean(axis=1)
    transform = make_transform(reference)

    train_msc = [transform(spectra) for _, spectra in train_blocks]
    scale = max(float(block.max()) for block in train_msc)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"invalid scaling factor estimated on the training set: {scale!r}")

    train_pp = [global_max_scale(block, scale) for block in train_msc]
    val_pp = [global_max_scale(transform(spectra), scale) for _, spectra in val_blocks]
    test_pp = [global_max_scale(transform(spectra), scale) for _, spectra in test_blocks]

    # --- 2T2D reference: mean preprocessed spectrum of training day 1 ------
    reference_2t2d = train_pp[0].mean(axis=1)
    print(
        f"2T2D reference: mean of {train_pp[0].shape[1]} preprocessed "
        f"day-{train_blocks[0][0]} training spectra"
    )

    manifest: List[dict] = []
    for split_name, blocks, processed in (
        ("train", train_blocks, train_pp),
        ("val", val_blocks, val_pp),
        ("test", test_blocks, test_pp),
    ):
        print(f"generating 2T2D for the {split_name} set ...")
        day_blocks = [(day, pp) for (day, _), pp in zip(blocks, processed)]
        manifest.extend(save_2t2d(day_blocks, split_name, reference_2t2d, args.output_dir))

    metadata = {
        "sheet_name": SHEET_NAME,
        "band_range": list(BAND_RANGE),
        "days": train_days,
        "n_points": int(train_shift.shape[0]),
        "global_scale": scale,
        "msc_reference": reference.tolist(),
        "t2d_reference": reference_2t2d.tolist(),
        "reference_source": f"training set, day {train_blocks[0][0]}, mean spectrum",
        "manifest": manifest,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"metadata written to {args.output_dir / 'metadata.json'}")
    print("done.")


if __name__ == "__main__":
    main()
