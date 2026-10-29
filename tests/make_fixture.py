"""
A tiny synthetic input that follows docs/conventions.md of workflow-epilepsy-ieeg:

  <out>/linked/ieeg/                     the `ieeg` port of ieeg-prep (a BIDS-iEEG derivative)
    dataset_description.json
    recordings.csv
    sub-01/ieeg/sub-01_task-ictal_run-01_desc-preproc_ieeg.edf      6 bipolar SEEG channels, 256 Hz, 150 s
    sub-01/ieeg/sub-01_task-ictal_run-01_desc-preproc_channels.tsv  onset/spread labels
    sub-01/ieeg/sub-01_task-ictal_run-01_desc-preproc_events.tsv    seizure + baseline
    sub-01/ieeg/sub-01_task-ictal_run-02_desc-preproc_*             a second seizure, same layout
  <out>/plain/                           a plain referential BIDS-iEEG recording (no recordings.csv),
                                         onset from an events `ictal` row, plus a broken EDF

The seizure: low-voltage fast activity (20-40 Hz, growing) on A01-A02 and A02-A03 from the onset,
recruiting A03-A04 8 s later, then rhythmic spike-waves; the other lead stays background.

Run: python -m tests.make_fixture <out-dir>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

FS = 256
DURATION = 150.0
ONSET = 90.0
BASELINE = (0.0, 60.0)
CONTACTS = ["A01", "A02", "A03", "A04", "B01", "B02", "B03", "B04"]


def write_edf(path: Path, names: list[str], data_uv: np.ndarray, fs: int, units: str = "uV") -> None:
    """Minimal EDF writer: 1 s records, 16-bit, per-channel physical range from the data."""
    n_ch, n = data_uv.shape
    n_rec = int(np.ceil(n / fs))
    pad = np.zeros((n_ch, n_rec * fs))
    pad[:, :n] = data_uv
    pmax = np.maximum(np.abs(pad).max(axis=1) * 1.1, 1.0)
    pmin = -pmax
    dmin, dmax = -32768, 32767
    digital = np.round((pad - pmin[:, None]) / (pmax - pmin)[:, None] * (dmax - dmin) + dmin).astype("<i2")

    def f(s, w):
        s = str(s)[:w]
        return s + " " * (w - len(s))

    hdr = f("0", 8) + f("X X X X", 80) + f("Startdate X X X X", 80) + "01.01.01" + "00.00.00"
    hdr += f(256 + 256 * n_ch, 8) + f("", 44) + f(n_rec, 8) + f(1, 8) + f(n_ch, 4)
    hdr += "".join(f(nm, 16) for nm in names)
    hdr += "".join(f("", 80) for _ in names)
    hdr += "".join(f(units, 8) for _ in names)
    hdr += "".join(f(f"{v:.3f}", 8) for v in pmin)
    hdr += "".join(f(f"{v:.3f}", 8) for v in pmax)
    hdr += "".join(f(dmin, 8) for _ in names) + "".join(f(dmax, 8) for _ in names)
    hdr += "".join(f("", 80) for _ in names) + "".join(f(fs, 8) for _ in names) + "".join(f("", 32) for _ in names)
    assert len(hdr) == 256 + 256 * n_ch
    with open(path, "wb") as fh:
        fh.write(hdr.encode("ascii"))
        for r in range(n_rec):
            fh.write(digital[:, r * fs:(r + 1) * fs].tobytes())


def background(rng, n: int, n_ch: int) -> np.ndarray:
    """1/f-like background, ~30 µV: integrated white noise, high-passed, plus an alpha-ish rhythm."""
    from scipy import signal

    w = rng.standard_normal((n_ch, n))
    b, a = signal.butter(2, [1.0, 60.0], "bandpass", fs=FS)
    x = signal.filtfilt(b, a, np.cumsum(w, axis=1) * 0.2 + w, axis=1)
    x /= x.std(axis=1, keepdims=True)
    t = np.arange(n) / FS
    return 30.0 * x + 8.0 * np.sin(2 * np.pi * 9.0 * t + rng.uniform(0, 6.28, (n_ch, 1)))


def seizure(t: np.ndarray, start: float, rng) -> np.ndarray:
    """LVFA growing over 10 s, then 3 Hz spike-and-wave, until the end."""
    x = np.zeros_like(t)
    on = t >= start
    tt = t[on] - start
    f = 22.0 + 1.5 * np.minimum(tt, 10.0)
    lvfa = np.minimum(tt / 3.0, 1.0) * 150.0 * np.sin(2 * np.pi * np.cumsum(f) / FS) * (tt < 12)
    sw = (tt >= 10) * 500.0 * (np.abs(np.sin(np.pi * 3.0 * tt)) ** 12) * np.sign(np.sin(2 * np.pi * 1.5 * tt))
    x[on] = lvfa + sw + 60.0 * np.minimum(tt / 3.0, 1.0) * rng.standard_normal(on.sum())
    return x


def referential(seed: int, onset: float = ONSET) -> tuple[list[str], np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(DURATION * FS)
    t = np.arange(n) / FS
    x = background(rng, n, len(CONTACTS))
    # Seizure on contacts A01, A02 (so A01-A02 and A02-A03 carry it) and A03 from +8 s.
    x[0] += seizure(t, onset, rng)
    x[1] += 0.5 * seizure(t, onset, rng)
    x[2] += 0.8 * seizure(t, onset + 8.0, rng)
    return CONTACTS, x


def bipolar(names: list[str], x: np.ndarray) -> tuple[list[str], list[str], np.ndarray]:
    out, contacts, rows = [], [], []
    for i in range(len(names) - 1):
        a, b = names[i], names[i + 1]
        if a[0] != b[0]:
            continue
        out.append(f"{a}-{b}")
        contacts.append(f"{a}|{b}")
        rows.append(x[i] - x[i + 1])
    return out, contacts, np.array(rows)


def write_tsv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def make(out: Path) -> Path:
    out = Path(out)
    d = out / "linked" / "ieeg"
    ieeg = d / "sub-01" / "ieeg"
    ieeg.mkdir(parents=True, exist_ok=True)
    (d / "dataset_description.json").write_text(json.dumps({
        "Name": "synthetic fixture", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
        "GeneratedBy": [{"Name": "ieeg-prep"}]}, indent=2))
    index = []
    onset_ch = {"A01-A02", "A02-A03"}
    spread_ch = {"A01-A02", "A02-A03", "A03-A04"}
    for run, seed in (("01", 1), ("02", 2)):
        stem = f"sub-01_task-ictal_run-{run}"
        names, x = referential(seed)
        bp, contacts, xb = bipolar(names, x)
        write_edf(ieeg / f"{stem}_desc-preproc_ieeg.edf", bp, xb, FS)
        (ieeg / f"{stem}_desc-preproc_ieeg.json").write_text(json.dumps({
            "SamplingFrequency": FS, "PowerLineFrequency": 60, "iEEGReference": "bipolar"}, indent=2))
        write_tsv(ieeg / f"{stem}_desc-preproc_channels.tsv", [
            {"name": n, "type": "SEEG", "units": "µV", "low_cutoff": "n/a", "high_cutoff": "n/a",
             "sampling_frequency": FS, "status": "good", "status_description": "n/a", "contacts": c,
             "soz": "true" if n in onset_ch else "false", "resected": "", "onset": "true" if n in onset_ch else "false",
             "spread": "true" if n in spread_ch else "false"} for n, c in zip(bp, contacts)])
        write_tsv(ieeg / f"{stem}_desc-preproc_events.tsv", [
            {"onset": BASELINE[0], "duration": BASELINE[1] - BASELINE[0], "trial_type": "baseline", "source": "fixture"},
            {"onset": ONSET, "duration": DURATION - ONSET, "trial_type": "seizure", "source": "fixture"}])
        index.append({"recording_id": stem, "subject": "01", "session": "", "task": "ictal", "run": run,
                      "path": f"sub-01/ieeg/{stem}_desc-preproc_ieeg.edf", "sfreq": FS, "n_channels": len(bp),
                      "n_good": len(bp), "duration_s": DURATION, "montage": "bipolar", "kind": "ictal",
                      "onset_s": ONSET, "baseline_start_s": BASELINE[0], "baseline_end_s": BASELINE[1],
                      "source_format": "edf"})
    with open(d / "recordings.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0]))
        w.writeheader()
        w.writerows(index)
    # Plain referential BIDS recording (no labels, no recordings.csv) and a broken file.
    p = out / "plain" / "sub-02" / "ieeg"
    p.mkdir(parents=True, exist_ok=True)
    names, x = referential(3)
    write_edf(p / "sub-02_task-ictal_ieeg.edf", names, x, FS)
    write_tsv(p / "sub-02_task-ictal_channels.tsv", [
        {"name": n, "type": "SEEG", "units": "uV", "sampling_frequency": FS, "status": "good",
         "status_description": "n/a"} for n in names])
    write_tsv(p / "sub-02_task-ictal_events.tsv", [
        {"onset": ONSET, "duration": DURATION - ONSET, "trial_type": "ictal", "value": 1, "sample": int(ONSET * FS)}])
    (p / "sub-02_task-broken_ieeg.edf").write_bytes(b"0       not an EDF file")
    return out


if __name__ == "__main__":
    print(make(Path(sys.argv[1] if len(sys.argv) > 1 else "data/fixture")))
