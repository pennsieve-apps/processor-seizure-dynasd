"""
Reading the iEEG input of the seizure-onset apps.

The normal input is the BIDS-iEEG derivative written by ieeg-prep (docs/conventions.md of
workflow-epilepsy-ieeg): a recordings.csv index plus per recording an EDF, _channels.tsv,
_events.tsv and _ieeg.json. A plain BIDS-iEEG dataset, or bare EDF / BrainVision / FIF files,
are read too, so the step can run on its own:

- recordings are found by name anywhere under INPUT_DIR (symlinks followed);
- labels (onset, spread, soz, resected) come from the _channels.tsv columns when present;
- the seizure onset comes from recordings.csv `onset_s`, else from an events row whose
  trial_type is `seizure` (or `ictal`, as in the Multi-Expert Seizure Annotation dataset);
- the baseline from recordings.csv `baseline_start_s`/`baseline_end_s`, else a `baseline` event;
- a recording that is not bipolar already (ieeg-prep montage `bipolar`, or channel names
  `A-B`) is re-referenced to neighbouring contacts on the same lead (LA01-LA02, ...), which is
  what both methods were designed for.

Signals are read lazily (MNE, preload=False) one block of channels at a time, so memory scales
with one block, not the whole recording.

This module is shared, verbatim, by processor-seizure-dynasd and processor-seizure-ei.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np

log = logging.getLogger(__name__)

SIGNAL_EXTS = (".edf", ".bdf", ".vhdr", ".fif")
# Channel types that carry brain signal; everything else (ECG, EMG, EOG, MISC, TRIG, ...) is ignored.
BRAIN_TYPES = {"SEEG", "ECOG", "EEG", "DBS"}
NON_BRAIN_NAME = re.compile(r"^(ECG|EKG|EMG|EOG|DC\d*|EVENT|STATUS|TRIG|PHOTIC|RESP|SPO2|PULSE|OSAT|PR|PLETH|X\d+)$|"
                            r"(^|[^A-Z])(ECG|EKG)", re.IGNORECASE)
CONTACT = re.compile(r"^(.*?)(\d+)$")
SEIZURE_TYPES = ("seizure", "ictal", "sz")


@dataclass
class Recording:
    recording_id: str
    path: Path                      # the signal file
    rel: str                        # path relative to INPUT_DIR (for provenance)
    source: str                     # derivative | bids | plain
    sfreq: float | None = None
    onset_s: float | None = None    # expert/clinical seizure onset, seconds from the start
    baseline: tuple[float, float] | None = None
    montage: str = "unknown"        # bipolar | car | as-recorded | unknown
    line_freq: float | None = None
    channels_tsv: Path | None = None
    events_tsv: Path | None = None
    meta: dict = field(default_factory=dict)


@dataclass
class Channel:
    name: str                       # output name (bipolar "LA01-LA02" or the channel itself)
    contacts: tuple[str, ...]       # source channel(s) in the file: (a,) or (a, b) for a - b
    type: str = "SEEG"
    labels: dict = field(default_factory=dict)  # onset/spread/soz/resected -> True/False/None


# ---------------------------------------------------------------------------------------------
# discovery

def _walk(root: Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        for f in sorted(filenames):
            if not f.startswith("."):
                out.append(Path(dirpath) / f)
    return out


def _float(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def read_tsv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def bids_prefix(path: Path) -> str:
    """sub-X_ses-Y_task-Z_run-01[_desc-preproc] from .../<prefix>_ieeg.edf (or the bare stem)."""
    name = path.name
    for ext in SIGNAL_EXTS:
        if name.lower().endswith(ext):
            name = name[: -len(ext)]
            break
    return name[: -len("_ieeg")] if name.endswith("_ieeg") else name


def recording_id_of(path: Path) -> str:
    """The recording id is the BIDS entity chain of the source recording (no desc- entity)."""
    return re.sub(r"_desc-[A-Za-z0-9]+", "", bids_prefix(path))


def _sidecar(path: Path, suffix: str) -> Path | None:
    p = path.parent / f"{bids_prefix(path)}_{suffix}"
    return p if p.exists() else None


def _events(path: Path | None) -> tuple[float | None, tuple[float, float] | None]:
    """(seizure onset, baseline interval) from an _events.tsv."""
    if path is None:
        return None, None
    onset = baseline = None
    try:
        rows = read_tsv(path)
    except OSError:
        return None, None
    for r in rows:
        tt = (r.get("trial_type") or "").strip().lower()
        t0, dur = _float(r.get("onset")), _float(r.get("duration"))
        if t0 is None:
            continue
        if onset is None and (tt in SEIZURE_TYPES or tt.startswith("seizure")):
            onset = t0
        elif baseline is None and tt == "baseline" and dur:
            baseline = (t0, t0 + dur)
    return onset, baseline


def _sidecar_json(path: Path) -> dict:
    p = _sidecar(path, "ieeg.json")
    if p is None:
        return {}
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def find_recordings(input_dir: Path) -> list[Recording]:
    """Every recording under input_dir: those indexed by a recordings.csv first, then any other signal file."""
    files = _walk(input_dir)
    recs: dict[str, Recording] = {}
    seen_paths: set[Path] = set()
    for idx in [f for f in files if f.name == "recordings.csv"]:
        for row in read_csv(idx):
            rel = (row.get("path") or "").strip()
            if not rel:
                continue
            p = (idx.parent / rel)
            if not p.exists():
                log.warning("%s: %s listed in recordings.csv does not exist", idx, rel)
                continue
            rid = (row.get("recording_id") or "").strip() or recording_id_of(p)
            b0, b1 = _float(row.get("baseline_start_s")), _float(row.get("baseline_end_s"))
            js = _sidecar_json(p)
            ev_onset, ev_base = _events(_sidecar(p, "events.tsv"))
            rec = Recording(
                recording_id=rid, path=p, rel=str(p.relative_to(input_dir)) if p.is_relative_to(input_dir) else str(p),
                source="derivative", sfreq=_float(row.get("sfreq")),
                onset_s=_float(row.get("onset_s")) if _float(row.get("onset_s")) is not None else ev_onset,
                baseline=(b0, b1) if b0 is not None and b1 is not None and b1 > b0 else ev_base,
                montage=(row.get("montage") or "unknown").strip() or "unknown",
                line_freq=_float(js.get("PowerLineFrequency")),
                channels_tsv=_sidecar(p, "channels.tsv"), events_tsv=_sidecar(p, "events.tsv"),
                meta={k: row.get(k) for k in ("subject", "session", "task", "run", "kind")},
            )
            if rid in recs:
                log.warning("%s: duplicate recording id, keeping %s", rid, recs[rid].rel)
                continue
            recs[rid] = rec
            seen_paths.add(p.resolve())
    for p in files:
        if p.suffix.lower() not in SIGNAL_EXTS or p.resolve() in seen_paths:
            continue
        rid = recording_id_of(p)
        if rid in recs:
            continue
        js = _sidecar_json(p)
        onset, base = _events(_sidecar(p, "events.tsv"))
        is_bids = p.name.endswith(tuple(f"_ieeg{e}" for e in SIGNAL_EXTS))
        ref = str(js.get("iEEGReference") or "").lower()
        recs[rid] = Recording(
            recording_id=rid, path=p, rel=str(p.relative_to(input_dir)) if p.is_relative_to(input_dir) else str(p),
            source="bids" if is_bids else "plain", sfreq=_float(js.get("SamplingFrequency")),
            onset_s=onset, baseline=base, montage="bipolar" if "bipolar" in ref else "unknown",
            line_freq=_float(js.get("PowerLineFrequency")),
            channels_tsv=_sidecar(p, "channels.tsv"), events_tsv=_sidecar(p, "events.tsv"),
        )
    return sorted(recs.values(), key=lambda r: r.recording_id)


# ---------------------------------------------------------------------------------------------
# channels and montage

def _label(v) -> bool | None:
    s = str(v if v is not None else "").strip().lower()
    if s in ("true", "1", "yes", "y"):
        return True
    if s in ("false", "0", "no", "n"):
        return False
    return None


def open_raw(rec: Recording):
    import mne

    mne.set_log_level("ERROR")
    raw = mne.io.read_raw(str(rec.path), preload=False, verbose="ERROR")
    return raw


def brain_channels(rec: Recording, raw) -> tuple[list[Channel], dict]:
    """
    The analysable channels of a recording after the montage: good brain channels from
    _channels.tsv (or every channel not named like ECG/EMG/... when there is none), bipolar
    when the recording is not bipolar already. Returns (channels, info) where info says what
    was done.
    """
    names_in_file = list(raw.ch_names)
    in_file = set(names_in_file)
    rows = read_tsv(rec.channels_tsv) if rec.channels_tsv else []
    by_name = {r.get("name", ""): r for r in rows}
    good: list[str] = []
    dropped: dict[str, str] = {}
    for n in names_in_file:
        r = by_name.get(n)
        if r is not None:
            typ = (r.get("type") or "").strip().upper()
            if typ and typ not in BRAIN_TYPES:
                dropped[n] = f"type {typ}"
                continue
            if (r.get("status") or "").strip().lower() == "bad":
                dropped[n] = "status bad" + (f" ({r.get('status_description')})" if r.get("status_description") not in (None, "", "n/a") else "")
                continue
        elif NON_BRAIN_NAME.search(n):
            dropped[n] = "not a brain channel (name)"
            continue
        good.append(n)
    labels_of = {n: {k: _label(by_name.get(n, {}).get(k)) for k in ("onset", "spread", "soz", "resected")} for n in good}
    type_of = {n: (by_name.get(n, {}).get("type") or "SEEG").strip().upper() for n in good}
    already_bipolar = rec.montage == "bipolar" or (good and sum("-" in n for n in good) > 0.8 * len(good))
    info = {"montage_in": rec.montage, "n_in_file": len(names_in_file), "dropped": dropped}
    if already_bipolar or rec.montage == "car":
        chans = []
        for n in good:
            chans.append(Channel(n, (n,), type_of[n], labels_of[n]))
        info["montage"] = "bipolar (as given)" if already_bipolar else "car (as given)"
        return chans, info
    # Referential: pair neighbouring contacts on the same lead.
    parsed = {}
    for n in good:
        m = CONTACT.match(n)
        if m and m.group(1):
            parsed[n] = (m.group(1), int(m.group(2)))
    index = {(lead, num): n for n, (lead, num) in parsed.items()}
    chans = []
    for n in good:
        if n not in parsed:
            continue
        lead, num = parsed[n]
        nxt = index.get((lead, num + 1))
        if nxt is None:
            continue
        la, lb = labels_of[n], labels_of[nxt]
        merged = {}
        for k in ("onset", "spread", "soz", "resected"):
            a, b = la.get(k), lb.get(k)
            merged[k] = True if (a or b) else (False if (a is False or b is False) else None)
        chans.append(Channel(f"{n}-{nxt}", (n, nxt), type_of[n], merged))
    info["montage"] = "bipolar (neighbouring contacts)"
    if not chans:
        # Nothing pairs (e.g. scalp EEG names): analyse as given.
        chans = [Channel(n, (n,), type_of[n], labels_of[n]) for n in good]
        info["montage"] = "as given (no contact pairs found)"
    assert all(c in in_file for ch in chans for c in ch.contacts)
    return chans, info


def iter_blocks(raw, chans: list[Channel], start: int = 0, stop: int | None = None,
                block: int = 16) -> Iterator[tuple[list[Channel], np.ndarray]]:
    """Yield (channels, data µV float64 [n_ch, n_samples]) for consecutive blocks of channels."""
    for i in range(0, len(chans), block):
        part = chans[i:i + block]
        need = sorted({c for ch in part for c in ch.contacts})
        data = raw.get_data(picks=need, start=start, stop=stop) * 1e6
        row = {n: j for j, n in enumerate(need)}
        out = np.empty((len(part), data.shape[1]), dtype=np.float64)
        for j, ch in enumerate(part):
            out[j] = data[row[ch.contacts[0]]]
            if len(ch.contacts) == 2:
                out[j] -= data[row[ch.contacts[1]]]
        yield part, out


# ---------------------------------------------------------------------------------------------
# intervals

def baseline_interval(rec: Recording, duration_s: float, default_len: float) -> tuple[tuple[float, float], str]:
    """
    The pre-ictal baseline: recordings.csv / events `baseline`; else, when the onset is known,
    [onset - 120 s, onset - 60 s] clipped to the recording (Ojemann et al. fit on such a
    stretch); else the first `default_len` seconds.
    """
    if rec.baseline is not None:
        b0, b1 = max(0.0, rec.baseline[0]), min(duration_s, rec.baseline[1])
        if b1 - b0 >= 5:
            return (b0, b1), "recording"
    if rec.onset_s is not None and rec.onset_s > 20:
        b1 = max(10.0, rec.onset_s - 60.0) if rec.onset_s >= 70 else rec.onset_s - 10.0
        b0 = max(0.0, b1 - default_len)
        if b1 - b0 >= 5:
            return (b0, b1), "before onset_s"
    return (0.0, min(default_len, duration_s / 3.0)), "start of recording"


def read_onsets_csv(input_dir: Path) -> dict[str, float | None]:
    """seizure_onsets.csv (from seizure-dynasd) anywhere under input_dir: recording_id -> onset_s."""
    out: dict[str, float | None] = {}
    for p in _walk(input_dir):
        if p.name != "seizure_onsets.csv":
            continue
        for r in read_csv(p):
            rid = (r.get("recording_id") or "").strip()
            if rid:
                out[rid] = _float(r.get("onset_s"))
    return out
