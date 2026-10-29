"""
Scores this app's output against expert labels (development tool; not in the image).

  python scripts/validate.py <output-dir> <truth> [annotations.tsv]

<truth> is either
  - the Multi-Expert Seizure Annotation dataset folder (Discover 530; has annotations.tsv):
    expert onset = consensus UEO time, onset channels = ueo_consensus contacts (a bipolar channel
    counts when either contact is listed), clip start = approximate_onset - 120 s; or
  - an ieeg-prep derivative (recordings.csv onset_s, _channels.tsv `onset` column).

Per recording: onset error (detected - expert, s), AUC of `score` for expert onset channels,
precision / recall of `recruited`. Prints the table and median [range].
"""
from __future__ import annotations

import ast
import csv
import re
import sys
from pathlib import Path

import numpy as np


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney AUC (ties count 1/2); NaN when one class is empty."""
    pos, neg = scores[labels], scores[~labels]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    diff = pos[:, None] - neg[None, :]
    return float(((diff > 0) + 0.5 * (diff == 0)).mean())


def truth_from_annotations(folder: Path) -> dict:
    rows = list(csv.DictReader(open(folder / "annotations.tsv", encoding="utf-8-sig"), delimiter="\t"))
    out = {}
    for edf in folder.rglob("*_ieeg.edf"):
        rid = edf.name[: -len("_ieeg.edf")]
        pat = re.match(r"sub-([A-Za-z0-9]+)_", rid).group(1)
        n = int(re.search(r"task-[a-z]+(\d+)", rid).group(1))
        hit = [r for r in rows if r["patient"] == pat and int(float(r["approximate_onset"])) == n]
        if len(hit) != 1:
            continue
        r = hit[0]
        start = float(r["approximate_onset"]) - 120.0
        out[rid] = {"onset_s": float(r["ueo_time_consensus"]) - start,
                    "contacts": set(ast.literal_eval(r["ueo_consensus"])),
                    "lvfa": r["lvfa"] == "yes", "stim": r["stim"]}
    return out


def truth_from_derivative(folder: Path) -> dict:
    out = {}
    for idx in folder.rglob("recordings.csv"):
        for r in csv.DictReader(open(idx, encoding="utf-8-sig")):
            edf = idx.parent / r["path"]
            ch = edf.parent / edf.name.replace("_ieeg.edf", "_channels.tsv")
            labels = {}
            if ch.exists():
                for c in csv.DictReader(open(ch, encoding="utf-8-sig"), delimiter="\t"):
                    if (c.get("onset") or "").lower() in ("true", "false"):
                        labels[c["name"]] = c["onset"].lower() == "true"
            out[r["recording_id"]] = {"onset_s": float(r["onset_s"]) if r.get("onset_s") else None,
                                      "labels": labels, "lvfa": None, "stim": None}
    return out


def add_flags(truth: dict, annotations: Path) -> None:
    """lvfa / stim flags from the Multi-Expert annotations.tsv for a derivative's recordings."""
    flags = truth_from_annotations_rows(annotations)
    for rid, t in truth.items():
        f = flags.get(rid)
        if f:
            t["lvfa"], t["stim"] = f


def truth_from_annotations_rows(annotations: Path) -> dict:
    rows = list(csv.DictReader(open(annotations, encoding="utf-8-sig"), delimiter="\t"))
    out = {}
    for r in rows:
        out[(r["patient"], int(float(r["approximate_onset"])))] = (r["lvfa"] == "yes", r["stim"])

    class Lookup(dict):
        def get(self, rid, default=None):
            m = re.match(r"sub-([A-Za-z0-9]+)_.*task-[a-z]+(\d+)", rid)
            return out.get((m.group(1), int(m.group(2)))) if m else default
    return Lookup()


def is_onset(t: dict, channel: str) -> bool | None:
    if "labels" in t:
        return t["labels"].get(channel)
    return bool(set(channel.split("-")) & t["contacts"])


def main(out_dir: Path, truth_dir: Path, annotations: Path | None = None) -> None:
    truth = truth_from_annotations(truth_dir) if (truth_dir / "annotations.tsv").exists() else truth_from_derivative(truth_dir)
    if annotations is not None:
        add_flags(truth, annotations)
    onsets = {r["recording_id"]: (float(r["onset_s"]) if r["onset_s"] else None)
              for r in csv.DictReader(open(out_dir / "seizure_onsets.csv"))}
    chans: dict[str, list[dict]] = {}
    for r in csv.DictReader(open(out_dir / "channel_onsets.csv")):
        chans.setdefault(r["recording_id"], []).append(r)
    res = []
    for rid in sorted(onsets):
        t = truth.get(rid)
        if t is None:
            continue
        rows = [r for r in chans.get(rid, []) if r["score"] != "" and is_onset(t, r["channel"]) is not None]
        lab = np.array([is_onset(t, r["channel"]) for r in rows], bool)
        sc = np.array([float(r["score"]) for r in rows])
        rec = np.array([r["recruited"] == "true" for r in rows], bool)
        tp = int((rec & lab).sum())
        err = onsets[rid] - t["onset_s"] if onsets[rid] is not None and t["onset_s"] is not None else float("nan")
        res.append({"recording": rid, "err_s": err, "auc": auc(sc, lab),
                    "precision": tp / rec.sum() if rec.sum() else float("nan"),
                    "recall": tp / lab.sum() if lab.sum() else float("nan"),
                    "n_recruited": int(rec.sum()), "n_onset": int(lab.sum()), "n_ch": len(rows),
                    "lvfa": t["lvfa"], "stim": t["stim"]})
    print(f"{'recording':55s} {'err_s':>7s} {'auc':>5s} {'prec':>5s} {'rec':>5s} {'n_rec':>5s} {'n_on':>4s} {'n_ch':>4s} lvfa stim")
    for r in res:
        print(f"{r['recording']:55s} {r['err_s']:7.2f} {r['auc']:5.2f} {r['precision']:5.2f} {r['recall']:5.2f} "
              f"{r['n_recruited']:5d} {r['n_onset']:4d} {r['n_ch']:4d} {str(r['lvfa'])[:1]:4s} {str(r['stim'])[:5]}")

    def summ(name, vals):
        v = np.array([x for x in vals if np.isfinite(x)])
        if len(v) == 0:
            return f"{name}: n=0"
        return f"{name}: median {np.median(v):.2f} [{v.min():.2f}, {v.max():.2f}] (n={len(v)})"

    print(summ("onset error (s)", [r["err_s"] for r in res]))
    e = np.array([abs(r["err_s"]) for r in res if np.isfinite(r["err_s"])])
    print(f"|onset error|: median {np.median(e):.2f} s; within 5 s: {int((e <= 5).sum())}/{len(res)}; "
          f"within 10 s: {int((e <= 10).sum())}/{len(res)}; no onset: {sum(not np.isfinite(r['err_s']) for r in res)}")
    for k in ("auc", "precision", "recall"):
        print(summ(k, [r[k] for r in res]))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None)
