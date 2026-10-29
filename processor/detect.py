"""
Seizure onset and spread with DynaSD (Ojemann et al. 2026), one recording at a time.

The pipeline follows the package's API and the preprint's Methods:

1. preprocess every channel as DynaSD's own `preprocess_for_detection` does after the montage:
   notch at the power-line frequency and its first harmonic, band-pass 3-150 Hz
   (dynasd.tools.preprocessing.bandpass_filter), resample to 256 Hz, AR(1) prewhitening (ar_one);
2. `model.fit(baseline)`: per-channel RobustScaler and, for the forecasting models, the baseline
   dynamics (LiNDDA: linear vector autoregressive forecaster of order 3; NDD: an LSTM);
3. `model(X)`: one score per 1 s window (0.5 s stride) and channel;
4. `model.get_threshold(method="pretrained", threshold_agg=...)`: the threshold the authors
   calibrated on their annotated cohort;
5. `model.get_onset_and_spread(scores, threshold, filter_w, rwin_size, rwin_req)`: smoothing,
   threshold, "seizing when rwin_req of the next rwin_size seconds are", per channel.

Added around it, for running unattended on whole recordings:

- channels above threshold for most of the baseline are left out (noisy: they cannot show a
  change; DynaSD's pipeline rejects noisy channels before fitting);
- the search runs from the end of the baseline, or from SEARCH_BEFORE_S before a known onset,
  and a channel's onset is the first time its sustained-activity flag switches on in that span
  (a run already on when the search starts began earlier and is not counted);
- the recording-level onset is the MIN_CHANNELS-th earliest channel onset (one channel alone,
  often an artefact, does not make a seizure);
- `recruited`: channel onset within ONSET_WINDOW_S of the recording onset (the preprint's
  "seizing at onset" = detected in the first second); `score`: the smoothed score over the
  SCORE_WINDOW_S after the recording onset divided by the threshold (> 1 = above threshold),
  so channels can be ranked without a hard cut (the maximum over the search when there is no
  onset).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage, signal

log = logging.getLogger(__name__)

TARGET_FS = 256
MODELS = ("lindda", "ndd", "absslp", "oncet")
METHOD_NAME = {m: f"dynasd-{m}" for m in MODELS}


@dataclass
class Params:
    model: str = "lindda"
    threshold: str = "mean"             # mean | median (the two pretrained thresholds) | a number
    filter_w: float = 10.0
    rwin_size: float = 5.0
    rwin_req: float = 4.0
    min_channels: int = 3
    onset_window_s: float = 1.0
    score_window_s: float = 5.0
    search_before_s: float = 30.0
    baseline_s: float = 60.0
    noisy_fraction: float = 0.5
    num_epochs: int = 10
    seed: int = 42
    checkpoint_dir: str | None = None


def preprocess(x: np.ndarray, fs: float, line_freq: float | None) -> np.ndarray:
    """
    One block of channels [n_ch, n] (µV, at fs) -> [n_ch, n'] at 256 Hz, the way DynaSD's
    preprocess_for_detection does it after the montage: notch, band-pass 3-150 Hz, resample.
    (AR(1) prewhitening is applied later on the whole analysis span, as ar_one needs it.)
    """
    from dynasd.tools.preprocessing import bandpass_filter

    lf = line_freq if line_freq in (50.0, 60.0) else 60.0
    y = x
    for f0 in (lf, 2 * lf):
        if f0 + 2 < fs / 2:
            b, a = signal.butter(4, (f0 - 2, f0 + 2), "bandstop", fs=fs)
            y = signal.filtfilt(b, a, y, axis=-1)
    hi = min(150.0, 0.45 * fs)
    y = bandpass_filter(y, fs, lo=3, hi=hi)
    if abs(fs - TARGET_FS) > 1e-6:
        from fractions import Fraction

        fr = Fraction(TARGET_FS / fs).limit_denominator(1000)
        y = signal.resample_poly(y, fr.numerator, fr.denominator, axis=-1)
    return y.astype(np.float32)


def build_model(p: Params, n_channels: int):
    import torch

    torch.manual_seed(p.seed)
    np.random.seed(p.seed)
    if p.model == "lindda":
        from dynasd import LiNDDA

        return LiNDDA(fs=TARGET_FS, sequence_length=3, forecast_length=2, w_size=1, w_stride=0.5,
                      num_epochs=p.num_epochs, batch_size=1024, lr=0.01, verbose=False)
    if p.model == "ndd":
        from dynasd import NDD

        return NDD(fs=TARGET_FS, hidden_size=10, sequence_length=12, forecast_length=1, w_size=1, w_stride=0.5,
                   num_epochs=p.num_epochs, batch_size=8192, lr=0.01, verbose=False)
    if p.model == "absslp":
        from dynasd import ABSSLP

        return ABSSLP(fs=TARGET_FS, w_size=1, w_stride=0.5)
    if p.model == "oncet":
        from dynasd import ONCET

        kw = {}
        if p.checkpoint_dir:
            kw = {"checkpoint_path": f"{p.checkpoint_dir}/ONCET/best_model.pth",
                  "config_path": f"{p.checkpoint_dir}/ONCET/final_training_config.json"}
        model = ONCET(fs=TARGET_FS, w_size=1, w_stride=0.5, device="cpu", verbose=False, batch_size=128, **kw)
        undilate(model.model)
        return model
    raise ValueError(f"unknown MODEL {p.model!r}; one of {', '.join(MODELS)}")


def undilate(net) -> None:
    """
    Rewrite every dilated Conv1d of a network as the equivalent undilated one (the kernel taps
    spread out with zeros in between). The output is the same to float precision (max |diff|
    4e-7 on ONCET), but PyTorch's CPU backend has no fast kernel for dilated depthwise
    convolutions (it falls back to slow_conv_dilated2d), and this makes ONCET ~5x faster.
    """
    import torch
    from torch import nn

    for mod in net.modules():
        if isinstance(mod, nn.Conv1d) and mod.dilation[0] > 1:
            d, k = mod.dilation[0], mod.kernel_size[0]
            w = mod.weight.data
            wide = torch.zeros(w.shape[0], w.shape[1], (k - 1) * d + 1, dtype=w.dtype)
            wide[:, :, ::d] = w
            mod.weight = nn.Parameter(wide, requires_grad=False)
            mod.kernel_size = (wide.shape[-1],)
            mod.dilation = (1,)


def ar_one(x: np.ndarray) -> np.ndarray:
    """DynaSD's AR(1) prewhitening (dynasd.tools.preprocessing.ar_one) on [n, n_ch]."""
    from dynasd.tools.preprocessing import ar_one as _ar

    return _ar(x)


def rising_onsets(clf: pd.DataFrame) -> dict[str, float | None]:
    """
    First onset per channel from DynaSD's sustained-activity matrix (get_onset_and_spread's
    ret_smooth_mat): the first window where the channel turns on, not counting a run that is
    already on when the search starts (that activity began before the search window).
    """
    out = {}
    idx = clf.index.to_numpy()
    for c in clf.columns:
        v = clf[c].to_numpy() > 0
        rise = np.flatnonzero(v[1:] & ~v[:-1]) + 1
        out[c] = float(idx[rise[0]]) if len(rise) else None
    return out


def run(x256: np.ndarray, names: list[str], t0: float, baseline: tuple[float, float],
        search: tuple[float, float | None], p: Params) -> dict:
    """
    x256: [n_ch, n] at 256 Hz covering [t0, ...) seconds of the recording; search = (start,
    end or None) in recording seconds. Returns per-channel onset/recruited/score and the
    recording onset, all in recording seconds.
    """
    timings = {}
    t = time.monotonic()
    white = ar_one(x256.T.astype(np.float64))          # [n-1, n_ch]; sample k+1 of x
    t0w = t0 + 1.0 / TARGET_FS
    X = pd.DataFrame(white.astype(np.float32), columns=names)
    del white
    b0 = max(0, int(round((baseline[0] - t0w) * TARGET_FS)))
    b1 = int(round((baseline[1] - t0w) * TARGET_FS))
    model = build_model(p, len(names))
    model.fit(X.iloc[b0:b1].reset_index(drop=True))
    timings["fit_s"] = round(time.monotonic() - t, 2)
    t = time.monotonic()
    scores = model(X).astype(float)                     # index: window start (s) from t0w
    del X
    scores.index = scores.index.to_numpy() + t0w
    scores = scores.ffill().bfill().fillna(0.0)
    timings["forward_s"] = round(time.monotonic() - t, 2)
    try:
        thr = float(p.threshold)
        model.threshold = thr
    except ValueError:
        thr = float(model.get_threshold(method="pretrained", threshold_agg=p.threshold))
    fw = int(np.floor((p.filter_w - 1.0) / 0.5)) + 1
    smooth_all = pd.DataFrame(ndimage.uniform_filter1d(scores.to_numpy(), size=fw, mode="nearest", axis=0, origin=0),
                              index=scores.index, columns=scores.columns)
    # Channels above threshold for most of the baseline cannot show a change: noisy, left out.
    inb = (smooth_all.index >= baseline[0]) & (smooth_all.index < baseline[1])
    above = (smooth_all.loc[inb] > thr).mean(axis=0)
    noisy = [c for c in names if above[c] > p.noisy_fraction]
    usable = [c for c in names if c not in noisy]
    if not usable:
        raise ValueError("every channel is above the threshold during the baseline")
    keep = scores.index >= max(search[0], baseline[1])
    if search[1] is not None:
        keep &= scores.index < search[1]
    sub = scores.loc[keep, usable]
    need = int(np.floor((p.rwin_size - 1.0) / 0.5)) + 1
    if len(sub) < 2 * need:
        raise ValueError(f"only {len(sub)} windows to search after the baseline; recording too short")
    _, clf = model.get_onset_and_spread(sub, threshold=thr, ret_smooth_mat=True, filter_w=p.filter_w,
                                        rwin_size=p.rwin_size, rwin_req=p.rwin_req)
    ch_onset = rising_onsets(clf)
    detected = sorted(v for v in ch_onset.values() if v is not None)
    k = max(1, p.min_channels)
    rec_onset = detected[k - 1] if len(detected) >= k else None
    smooth = smooth_all.loc[sub.index, usable]
    if rec_onset is not None:
        w = (smooth.index >= rec_onset) & (smooth.index < rec_onset + max(p.score_window_s, 0.5))
        score = smooth.loc[w].mean(axis=0) / thr
    else:
        score = smooth.max(axis=0) / thr
    rows = []
    for c in names:
        on = ch_onset.get(c)
        rows.append({
            "channel": c,
            "onset_s": on,
            "recruited": bool(on is not None and rec_onset is not None and on <= rec_onset + p.onset_window_s),
            "score": float(score[c]) if c in usable else None,
        })
    return {"rec_onset": rec_onset, "threshold": thr, "rows": rows, "timings": timings, "noisy": noisy,
            "n_detected": len(detected), "n_windows": int(len(sub))}
