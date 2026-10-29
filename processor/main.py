"""
processor-seizure-dynasd: seizure onset time and per-channel onset/spread with DynaSD.

Reads the ieeg-prep derivative (or any BIDS-iEEG / EDF input) under INPUT_DIR and writes

  OUTPUT_DIR/dynasd/seizure_onsets.csv   recording_id, onset_s, method
  OUTPUT_DIR/dynasd/channel_onsets.csv   recording_id, channel, onset_s, recruited, score, method
  OUTPUT_DIR/dynasd/provenance.json

Everything sits in a folder named after the method, so this output can be one of several links
into one fan-in port (seizure-consensus `methods`) without its file names colliding.

One recording at a time; signals are read a block of channels at a time and kept at 256 Hz.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

from processor import detect as D
from processor import resources as R
from processor import ieeg as I

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("processor-seizure-dynasd")
for noisy in ("mne", "matplotlib", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

APP_NAME = "processor-seizure-dynasd"
OUTPUT_FOLDER = "dynasd"  # OUTPUT_DIR/<folder>/: the output can be one link of a fan-in port
REPO_ROOT = Path(__file__).resolve().parent.parent
DYNASD_COMMIT = "48b2b959605a37fb03a4d99ae6fcc1459c04cc37"


def version_of(root: Path) -> str:
    try:
        return (root / "VERSION").read_text().strip()
    except OSError:
        return "unknown"


def tool_versions() -> dict[str, str]:
    out = {"python": platform.python_version(), "dynasd_commit": DYNASD_COMMIT}
    for pkg in ("dynasd", "torch", "mne", "numpy", "scipy", "scikit-learn", "pandas"):
        try:
            out[pkg] = importlib_metadata.version(pkg)
        except importlib_metadata.PackageNotFoundError:
            pass
    return out


def _num(name: str, default: float, cast=float, lo: float | None = None) -> float:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return default
    try:
        v = cast(float(raw)) if cast is int else cast(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a number") from exc
    if lo is not None and v < lo:
        raise ValueError(f"{name}={raw!r} must be >= {lo}")
    return v


def env_params() -> D.Params:
    model = (os.environ.get("MODEL") or "lindda").strip().lower()
    if model not in D.MODELS:
        raise ValueError(f"MODEL={model!r}: one of {', '.join(D.MODELS)}")
    thr = (os.environ.get("THRESHOLD") or "mean").strip().lower()
    if thr not in ("mean", "median"):
        try:
            if float(thr) <= 0:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"THRESHOLD={thr!r}: mean, median or a positive number") from exc
    p = D.Params(
        model=model, threshold=thr,
        filter_w=_num("FILTER_W", 10.0, lo=1.0), rwin_size=_num("RWIN_SIZE", 5.0, lo=1.0),
        rwin_req=_num("RWIN_REQ", 4.0, lo=1.0), min_channels=_num("MIN_CHANNELS", 3, int, lo=1),
        onset_window_s=_num("ONSET_WINDOW_S", 1.0, lo=0.0), score_window_s=_num("SCORE_WINDOW_S", 5.0, lo=0.5),
        search_before_s=_num("SEARCH_BEFORE_S", 30.0, lo=-1.0), baseline_s=_num("BASELINE_S", 60.0, lo=5.0),
        seed=_num("SEED", 42, int), checkpoint_dir=os.environ.get("DYNASD_CHECKPOINTS") or None,
        batch=_num("BATCH", 0, int, lo=0),
    )
    if p.rwin_req > p.rwin_size:
        raise ValueError("RWIN_REQ must be <= RWIN_SIZE")
    device = (os.environ.get("DEVICE") or "auto").strip().lower()
    if device not in D.DEVICES:
        raise ValueError(f"DEVICE={device!r}: one of {', '.join(D.DEVICES)}")
    return p


def choose_device(p: D.Params) -> dict:
    """Pick the device for DEVICE (auto|cuda|cpu), set p.device, log it; the provenance record."""
    requested, p.device, why = D.choose_device(os.environ.get("DEVICE") or "auto")
    (log.warning if requested == "cuda" and p.device == "cpu" else log.info)("%s", why)
    rec = {"requested": requested, "used": p.device, "reason": why}
    if p.device == "cuda":
        import torch

        # ONCET runs fixed-size batches: let cuDNN pick its fastest convolution algorithms.
        torch.backends.cudnn.benchmark = True
        rec.update({"gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
                    "cudnn": torch.backends.cudnn.version()})
    if p.model == "oncet":
        rec["oncet_batch"] = D.oncet_batch(p)
        log.info("ONCET batch: %d (%s)", rec["oncet_batch"], f"BATCH={p.batch}" if p.batch else f"auto for {p.device}")
    return rec


def process_recording(rec: I.Recording, p: D.Params) -> dict:
    t_start = time.monotonic()
    raw = I.open_raw(rec)
    fs = float(raw.info["sfreq"])
    if fs < 200:
        raise ValueError(f"sampling rate {fs:g} Hz < 200 Hz")
    duration = raw.n_times / fs
    chans, minfo = I.brain_channels(rec, raw)
    if len(chans) < 2:
        raise ValueError(f"{len(chans)} usable channel(s)")
    base, base_src = I.baseline_interval(rec, duration, p.baseline_s)
    if p.search_before_s >= 0 and rec.onset_s is not None:
        search = (max(base[1], rec.onset_s - p.search_before_s), None)
        search_src = f"onset_s - {p.search_before_s:g} s"
    else:
        search = (base[1], None)
        search_src = "after the baseline"
    t0 = base[0]
    start = int(np.floor(t0 * fs))
    blocks = []
    for _, x in I.iter_blocks(raw, chans, start=start):
        blocks.append(D.preprocess(x, fs, rec.line_freq))
    x256 = np.concatenate(blocks)
    del blocks
    t_read = time.monotonic() - t_start
    res = D.run(x256, [c.name for c in chans], start / fs, base, search, p)
    del x256
    res["timings"]["read_preprocess_s"] = round(t_read, 2)
    res["timings"]["total_s"] = round(time.monotonic() - t_start, 2)
    res.update({
        "sfreq": fs, "duration_s": round(duration, 3), "n_channels": len(chans), "montage": minfo["montage"],
        "channels_dropped": minfo["dropped"], "baseline": [round(base[0], 3), round(base[1], 3)],
        "baseline_source": base_src, "search_start_s": round(search[0], 3), "search_source": search_src,
        "onset_s_given": rec.onset_s,
    })
    return res


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def main() -> int:
    t_run = time.monotonic()
    input_dir = Path(os.environ.get("INPUT_DIR") or "/data/input")
    output_dir = Path(os.environ.get("OUTPUT_DIR") or "/data/output") / OUTPUT_FOLDER
    try:
        p = env_params()
    except ValueError as exc:
        log.error("%s", exc)
        return 2
    import torch

    # PyTorch threads share the process's memory: one per CPU the container is given (the cgroup
    # quota; os.cpu_count() is the host's), no memory cap.
    n_cpu, cpu_src = R.available_cpus()
    torch.set_num_threads(n_cpu)
    log.info("PyTorch threads: %d (%s)", n_cpu, cpu_src)
    device = choose_device(p)
    output_dir.mkdir(parents=True, exist_ok=True)
    recs = I.find_recordings(input_dir) if input_dir.exists() else []
    log.info("%s %s: %d recording(s) under %s; MODEL=%s THRESHOLD=%s on %s", APP_NAME, version_of(REPO_ROOT),
             len(recs), input_dir, p.model, p.threshold, p.device)
    method = D.METHOD_NAME[p.model]
    statuses = {}
    onset_rows, chan_rows = [], []
    for rec in recs:
        try:
            res = process_recording(rec, p)
        except Exception as exc:  # noqa: BLE001 - a failed recording is logged and skipped
            log.error("%s: failed: %s", rec.recording_id, exc)
            log.debug("%s", traceback.format_exc())
            statuses[rec.recording_id] = {"status": "failed", "error": str(exc), "input": rec.rel}
            continue
        onset_rows.append({"recording_id": rec.recording_id, "onset_s": fmt(res["rec_onset"]), "method": method})
        for r in res["rows"]:
            chan_rows.append({"recording_id": rec.recording_id, "channel": r["channel"], "onset_s": fmt(r["onset_s"]),
                              "recruited": "true" if r["recruited"] else "false",
                              "score": fmt(r["score"], 4), "method": method})
        n_rec = sum(r["recruited"] for r in res["rows"])
        log.info("%s: %d channels (%s), baseline %.1f-%.1f s (%s), %d noisy, threshold %.4g, onset %s, "
                 "%d channels detected, %d recruited, %.1f s", rec.recording_id, res["n_channels"], res["montage"],
                 res["baseline"][0], res["baseline"][1], res["baseline_source"], len(res["noisy"]), res["threshold"],
                 "none" if res["rec_onset"] is None else f"{res['rec_onset']:.2f} s", res["n_detected"], n_rec,
                 res["timings"]["total_s"])
        statuses[rec.recording_id] = {
            "status": "ok", "input": rec.rel, "onset_s": res["rec_onset"], "threshold": res["threshold"],
            "n_channels": res["n_channels"], "n_detected": res["n_detected"], "n_recruited": n_rec,
            "noisy_channels": res["noisy"], "montage": res["montage"], "channels_dropped": res["channels_dropped"],
            "sfreq": res["sfreq"], "duration_s": res["duration_s"], "baseline_s": res["baseline"],
            "baseline_source": res["baseline_source"], "search_start_s": res["search_start_s"],
            "search_source": res["search_source"], "onset_s_given": res["onset_s_given"], "timings": res["timings"],
        }
    with open(output_dir / "seizure_onsets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["recording_id", "onset_s", "method"])
        w.writeheader()
        w.writerows(onset_rows)
    with open(output_dir / "channel_onsets.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["recording_id", "channel", "onset_s", "recruited", "score", "method"])
        w.writeheader()
        w.writerows(chan_rows)
    ok = sum(1 for s in statuses.values() if s["status"] == "ok")
    prov = {
        "app": APP_NAME, "version": version_of(REPO_ROOT), "tools": tool_versions(),
        "parameters": {**{k: v for k, v in vars(p).items() if k not in ("checkpoint_dir", "device")},
                       "DEVICE": device["requested"]},
        # the device actually used, why, and on cuda the GPU and CUDA/cuDNN versions
        "device": device,
        "inputs": [r.rel for r in recs],
        "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seconds": round(time.monotonic() - t_run, 2), "recordings": statuses,
    }
    (output_dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str) + "\n")
    if not recs:
        log.error("no recordings found under %s", input_dir)
        return 1
    if ok == 0:
        log.error("every recording failed")
        return 1
    log.info("done: %d/%d recordings in %.1f s", ok, len(recs), time.monotonic() - t_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
