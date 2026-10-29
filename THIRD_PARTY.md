# Third-party software

This app is Apache-2.0 (see `LICENSE`). The image it builds contains the following third-party
components, installed from PyPI (PyTorch from the PyTorch CPU wheel index) at the versions
pinned in `requirements.txt`, DynaSD from a pinned GitHub commit, and DynaSD's ONCET checkpoint.
Nothing GPL is linked in-process (PyTorch's bundled `libgomp` is GPL-3.0 with the GCC Runtime
Library Exception, which permits this use).

| Component | Version | What it does here | Licence | Source |
|---|---|---|---|---|
| DynaSD | commit `48b2b959605a37fb03a4d99ae6fcc1459c04cc37` (package 0.1.0) | The detectors (LiNDDA, NDD, ABSSLP, ONCET), thresholds, `get_onset_and_spread`, band-pass and AR(1) prewhitening helpers | MIT (Copyright (c) 2025 William Ojemann, Katherine Walsh, Haoer Shi). The author confirmed use of the code and the released checkpoints. | https://github.com/wojemann/DynaSD |
| DynaSD ONCET checkpoint | release `v0.1.0-checkpoints`: `best_model.pth` (sha256 `34afd9c5…5c2f`), `final_training_config.json` (sha256 `0481bbcc…7dda`) | Pretrained weights of the ONCET classifier (`MODEL=oncet`), vendored at build time into `/opt/dynasd` | MIT (released with DynaSD; use confirmed by the author) | https://github.com/wojemann/DynaSD/releases/tag/v0.1.0-checkpoints |
| PyTorch | 2.11.0+cpu | DynaSD's forecasters and classifier (CPU-only build, no CUDA) | BSD-3-Clause; bundled third-party libraries (oneDNN, sleef, pthreadpool, …) under their own permissive licences, `libgomp` GPL-3.0 with GCC Runtime Library Exception | https://pytorch.org |
| MNE-Python | 1.13.2 | Reads EDF / BrainVision / FIF | BSD-3-Clause | https://mne.tools |
| NumPy | 2.4.6 | Arrays | BSD-3-Clause (bundled parts 0BSD, MIT, Zlib, CC0-1.0) | https://numpy.org |
| SciPy | 1.18.1 | Notch filters, resampling, smoothing | BSD-3-Clause | https://scipy.org |
| scikit-learn | 1.9.1 | DynaSD's RobustScaler | BSD-3-Clause | https://scikit-learn.org |
| pandas | 3.1.0 | DynaSD's data frames | BSD-3-Clause | https://pandas.pydata.org |
| tqdm | 4.70.1 | DynaSD dependency | MPL-2.0 AND MIT | https://github.com/tqdm/tqdm |
| joblib, threadpoolctl | 1.6.0, 3.7.0 | scikit-learn dependencies | BSD-3-Clause | PyPI |
| Matplotlib (+ contourpy, cycler, fonttools, kiwisolver, pillow, pyparsing) | 3.11.2 (1.4.0, 0.12.1, 4.68.0, 1.5.1, 13.0.0, 3.3.3) | MNE dependency (not used for plots here) | Matplotlib License (PSF-based); BSD-3-Clause; BSD-3-Clause; MIT; BSD-3-Clause; MIT-CMU; MIT | https://matplotlib.org |
| sympy, mpmath, networkx, filelock, fsspec, Jinja2, MarkupSafe, typing_extensions | 1.14.0, 1.3.0, 3.7, 4.2.4, 2026.9.0, 3.1.6, 3.0.4, 4.16.0 | PyTorch dependencies | BSD-3-Clause; BSD-3-Clause; BSD-3-Clause; MIT; BSD-3-Clause; BSD-3-Clause; BSD-3-Clause; PSF-2.0 | PyPI |
| decorator, lazy-loader, pooch, platformdirs, packaging, requests, urllib3, idna, charset-normalizer, certifi | 5.3.1, 0.6, 1.9.0, 4.14.0, 26.3, 2.34.2, 2.8.0, 3.20, 3.5.2, 2026.10.23 | MNE dependencies (no network is used at run time) | BSD-2-Clause; BSD-3-Clause; BSD-3-Clause; MIT; Apache-2.0 OR BSD-2-Clause; Apache-2.0; MIT; BSD-3-Clause; MIT; MPL-2.0 | PyPI |
| cloudpickle, narwhals | 3.1.2, 2.28.0 | dependencies | BSD-3-Clause; MIT | PyPI |
| python-dateutil, six, tzdata | 3.0.0, 1.17.0, 2026.6 | pandas dependencies | Apache-2.0 / BSD dual; MIT; Apache-2.0 | PyPI |
| python:3.12.16-slim-bookworm | — | Base image (Debian 12 + CPython 3.12, pip) | PSF-2.0 (Python); Debian packages under their own free licences; pip MIT | https://hub.docker.com/_/python |

Not bundled: DynaSD's WVNT detector (needs TensorFlow, which would more than double the image)
and its `[test]` extras (`ieeg`, `neurodsp`, `seaborn`).

Demo data used for testing (not in the image or the repository): Ojemann WK, Zhou D,
Armstrong C, Sinha N, Litt B, Conrad E. Multi-Expert Seizure Annotation. Pennsieve Discover
2026, doi:10.26275/dely-fzwx, CC BY 4.0.
