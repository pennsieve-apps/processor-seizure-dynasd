# processor-seizure-dynasd

The seizure-detection step of the Seizure Onset Localisation workflow
(`ieeg-prep → seizure-dynasd → {seizure-ei, seizure-fragility} → seizure-consensus → ieeg-report`).
It wraps **DynaSD** (Ojemann et al. 2026, github.com/wojemann/DynaSD, MIT).

**For an epileptologist.** Given SEEG or ECoG recordings that contain a seizure, the app marks
when the seizure starts and, for every bipolar channel, when that channel starts seizing,
whether it is among the channels seizing at onset (`recruited`), and a score to rank channels by.
It learns what each recording looks like before the seizure (a pre-ictal baseline) and flags
the moment each channel's signal stops behaving like it. It is meant to put a first, reproducible
mark on the onset and the onset channels for review, not to replace reading the EEG; on the
demo data it is within 5 s of the epileptologists' consensus onset in two thirds of spontaneous
seizures and ranks their onset channels near the top (median AUC 0.99), but it is unreliable on
stimulation-induced seizures (see *Validation*).

**For a researcher.** DynaSD's "neural dynamic divergence" models are fitted per recording on
its own baseline and scored on 1 s windows (0.5 s stride); the default `MODEL=lindda` is LiNDDA,
a linear vector-autoregressive forecaster (order 3) whose per-window forecast error is the
score, the model the preprint found best on intracranial data and cheapest to run. The app uses
the package's own API end to end (`fit` on the baseline → `forward` → `get_threshold(pretrained)`
→ `get_onset_and_spread`) and its preprocessing helpers, and adds what is needed to run
unattended on whole recordings: a baseline and search window, rejection of channels that are
already above threshold at baseline, a recording-level onset that one artefactual channel cannot
trigger, and a per-channel score.

## Ports

| Port | Direction | Placement | What |
|---|---|---|---|
| `ieeg` | input | root | The ieeg-prep derivative (`recordings.csv`, EDF + `_channels.tsv` + `_events.tsv` + `_ieeg.json`), or any BIDS-iEEG dataset or EDF / BrainVision / FIF files, found by name anywhere under `INPUT_DIR` (symlinks followed). |
| `onsets` | output | `OUTPUT_DIR/dynasd/` | `seizure_onsets.csv`, `channel_onsets.csv`, `provenance.json`, under a folder named after the method so the output can be one of several links into a fan-in port (seizure-consensus `methods`) without name collisions. |

## Parameters

| Name | Default | What it does |
|---|---|---|
| `MODEL` | `lindda` | `lindda`: LiNDDA, linear forecaster of the baseline dynamics. `ndd`: NDD, the same with an LSTM forecaster. `absslp`: baseline-normalised mean absolute slope (Schindler et al. 2007), no training. `oncet`: DynaSD's pretrained convolutional seizure classifier (checkpoint bundled; ~5 min per recording on CPU: run it on the gpu runtime, see *GPU*). |
| `THRESHOLD` | `mean` | `mean` or `median`: the two pretrained thresholds DynaSD ships per model (phi-optimal, mean over patients; F1-optimal, median over patients). `mean` is stricter and did better on the demo (below). A number sets the threshold on the model's score. |
| `MIN_CHANNELS` | `3` | The recording-level onset is the onset of the k-th earliest channel. `1` takes the first channel, which a single artefactual channel can set. |
| `SEARCH_BEFORE_S` | `30` | When the input gives a seizure onset (`recordings.csv onset_s`, or a `seizure`/`ictal` event), onsets are searched from this long before it; negative: ignore the given onset and search everything after the baseline. |
| `ONSET_WINDOW_S` | `1` | `recruited` = channel onset within this long after the recording onset (the preprint's "seizing at onset": detected in the first second). |
| `SCORE_WINDOW_S` | `5` | `score` = smoothed model output over this long after the recording onset, divided by the threshold. |
| `BASELINE_S` | `60` | Baseline length when the input gives none: the stretch ending 60 s before the given onset (as in the preprint), else the start of the recording. |
| `FILTER_W` | `10` | DynaSD `filter_w`: moving average (s) over the window scores before thresholding. |
| `RWIN_SIZE`, `RWIN_REQ` | `5`, `4` | DynaSD `rwin_size`/`rwin_req`: a window is seizing when `RWIN_REQ` of the next `RWIN_SIZE` seconds are above threshold. |
| `SEED` | `42` | Seed of the forecasters' initial weights and training order. |
| `DEVICE` | `auto` | `auto`, `cuda` or `cpu`: where PyTorch runs. `auto` takes `cuda` when `torch.cuda.is_available()` (the gpu runtime) and `cpu` otherwise; `cuda` on a host without a GPU falls back to `cpu` with a WARNING rather than failing; `cpu` stays on the CPU even with a GPU. Logged at start and recorded in `provenance.json` as `device` (`requested`, `used`, `reason`, and on cuda the GPU name and CUDA/cuDNN versions). See *GPU*. |
| `BATCH` | `0` | ONCET's inference batch (windows × channels per forward pass). `0` = automatic: 8192 on cuda (sized for a 16 GB T4: under 1 GB of activations), 128 on cpu (v0.1.0's). Only float rounding depends on it. LiNDDA's (1024) and NDD's (8192) batch sizes are training hyperparameters that shape the fitted model, so they are fixed on every device. |

## What it does, step by step

1. **Finds** the recordings (`recordings.csv` first, then any other EDF/BrainVision/FIF), with
   their sampling rate, onset, baseline and power-line frequency from the BIDS sidecars.
2. **Channels.** Good brain channels (`_channels.tsv` `status`/`type`; channels named like
   ECG/EMG/… are dropped when there is no `_channels.tsv`). A recording that is not bipolar
   already is re-referenced to neighbouring contacts on the same lead (`LA01-LA02`, …), the
   montage DynaSD was built and calibrated on.
3. **Preprocessing** (DynaSD's `preprocess_for_detection` after the montage, a block of 16
   channels at a time): notch at the line frequency and its harmonic, band-pass 3–150 Hz
   (`dynasd.tools.preprocessing.bandpass_filter`), polyphase resampling to 256 Hz, AR(1)
   prewhitening (`ar_one`).
4. **Baseline**: `recordings.csv baseline_start_s/baseline_end_s` or a `baseline` event; else
   [onset − 120 s, onset − 60 s] when the onset is known (the preprint fits on −120 to −60 s);
   else the first `BASELINE_S` seconds.
5. **Model**: `fit(baseline)`; `forward` over the recording from the baseline on; threshold
   from `get_threshold(method="pretrained", threshold_agg=THRESHOLD)`.
6. **Noisy channels**: channels whose smoothed score is above the threshold for more than half
   of the baseline cannot show a change and are left out (listed in `provenance.json`; their
   rows have an empty `score`).
7. **Onsets**: `get_onset_and_spread` (smoothing `FILTER_W`, sustained activity
   `RWIN_REQ`/`RWIN_SIZE`) over the search window (from the end of the baseline, or from
   `SEARCH_BEFORE_S` before the given onset, to the end). A channel's onset is the first time
   its sustained-activity flag switches **on** in the search window (a run already on when the
   search starts began before it and is not counted). The recording onset is the
   `MIN_CHANNELS`-th earliest channel onset.

## Outputs

**`seizure_onsets.csv`**: one row per recording: `recording_id`, `onset_s` (seconds from the
recording start; empty when no onset was found), `method` (`dynasd-lindda`, …).

**`channel_onsets.csv`**: one row per channel and recording.

| Column | |
|---|---|
| `recording_id` | the recording (BIDS entity chain) |
| `channel` | the bipolar channel (`LA01-LA02`) |
| `onset_s` | when the channel starts seizing (s from the recording start); empty if never in the search window |
| `recruited` | `true` when `onset_s` ≤ recording onset + `ONSET_WINDOW_S` |
| `score` | smoothed model output over `SCORE_WINDOW_S` after the recording onset ÷ threshold (> 1 = above threshold); the maximum over the search window when there is no onset; empty for noisy channels |
| `method` | `dynasd-<model>` |

**`provenance.json`**: app and version, tool versions (DynaSD commit, PyTorch, MNE, NumPy, SciPy,
scikit-learn, pandas), parameters, inputs, timings, and per recording: status, onset, threshold,
channels (detected, recruited, noisy, dropped and why), montage, baseline and search window and
where they came from.

## Validation on the demo data

24 SEEG seizures from 6 patients (Multi-Expert Seizure Annotation, doi:10.26275/dely-fzwx; four
seizures each, one of them stimulation-induced), run on the raw clips (re-referenced here to
85–208 neighbouring-contact bipolar channels, 1024 Hz, 4.2–7.7 min each, 120 s before the
clinically marked onset). Truth: the consensus of 3–5 epileptologists, onset time =
`ueo_time_consensus` (median of their unequivocal electrographic onsets), onset channels =
`ueo_consensus` (a bipolar channel counts when either contact is listed). `scripts/validate.py`
computes per seizure the onset error (detected − expert), the AUC of `score` for the expert
onset channels, and precision/recall of `recruited` against them.

Default settings (`MODEL=lindda`, `THRESHOLD=mean`, `MIN_CHANNELS=3`, `SEARCH_BEFORE_S=30`
around the clip's clinically marked onset):

| | onset error (s), median [range] | \|error\| median; ≤ 5 s; ≤ 10 s | AUC onset channels | precision | recall |
|---|---|---|---|---|---|
| all 24 | −3.8 [−70.2, 44.0] | 5.0 s; 12/24; 14/24 | **0.98** [0.41, 1.00] | 0.33 [0, 1] | 0.60 [0, 1] |
| 18 spontaneous | −3.1 [−26.6, 44.0] | 3.9 s; 12/18; 14/18 | 0.99 [0.41, 1.00] | 0.33 | 0.64 |
| 6 stimulation-induced | −23.7 [−70.2, 36.3] | 30.9 s; 0/6; 0/6 | 0.85 [0.41, 0.99] | 0.00 | 0.00 |

Variants (all 24 seizures, otherwise default):

| Setting | \|error\| median; ≤ 5 s; ≤ 10 s | AUC median [range] | precision / recall |
|---|---|---|---|
| default (above) | 5.0 s; 12; 14 | 0.98 [0.41, 1.00] | 0.33 / 0.60 |
| search anchored on the expert onset instead (as ieeg-prep's `onset_s` will give) | 5.0 s; 12; 16 | 0.98 [0.41, 1.00] | 0.29 / 0.63 |
| no anchor (`SEARCH_BEFORE_S=-1`: everything after the 60 s baseline) | 17.7 s; 10; 11 | 0.97 [0.41, 1.00] | 0.25 / 0.60 |
| `THRESHOLD=median` (F1-optimal, lower) | 14.5 s; 5; 12 | 0.95 [0.41, 1.00] | 0.10 / 0.42 |
| `MODEL=ndd` | 18.9 s; 5; 10 | 0.92 [0.05, 1.00] | 0.11 / 0.17 |
| `MODEL=absslp` | 15.0 s; 6; 11 | 0.78 [0.10, 1.00] | 0.18 / 0.20 |
| `MODEL=oncet` (pretrained threshold 0.62)¹ | 6.7 s; 10; 13 | 0.95 [0.70, 1.00] | 0.29 / 0.25 |

¹ replayed with the same rules on ONCET scores cached from the same preprocessing (a full app
run takes ~1 h natively for the 24 clips); the same replay gives LiNDDA 5.0 s; 12; 15 and
AUC 0.98, matching the app run.

What this says:

- On spontaneous seizures LiNDDA puts the onset within 5 s of the experts' consensus in 12 of
  18 and ranks their onset channels at the top (median AUC 0.99). Errors are mostly **early**
  (median −3 s): pre-ictal changes on a few channels cross the threshold before the experts'
  unequivocal onset; `MIN_CHANNELS=3` and the `mean` threshold were chosen against exactly that.
- **Stimulation-induced seizures fail** (0 of 6 within 10 s): the 1 Hz stimulation train before
  the onset is a large departure from the baseline, and in some clips it is inside the baseline.
- `recruited` is a hard cut and imprecise (precision 0.33): the first channels DynaSD sees
  are often neighbours of the experts' channels or the first-recruited contacts of a fast
  spread. Use `score` to rank.
- LiNDDA beats NDD and ABSSLP here, as in the preprint; the pretrained ONCET classifier comes
  close on onset time and has the best worst case (AUC ≥ 0.70) but is ~30× slower on CPU.
- These defaults (`THRESHOLD=mean`, `MIN_CHANNELS=3`, `SEARCH_BEFORE_S=30`, noisy-channel rule)
  were chosen on these same 24 seizures from a small grid (threshold mean/median/per-recording
  GMM × k = 1–3 × search start), so the numbers are in-sample. They are not tuned per patient.

**On the ieeg-prep derivative** (`workflow-epilepsy-ieeg/data/work/prep/ieeg`: the same 24
clips, bipolar with bad channels marked, `onset_s` = the expert consensus onset, baseline =
recording start to onset − 10 s, at most 90 s). Because `onset_s` there *is* the expert onset,
anchoring on it is optimistic; the honest number is the onset-blind run (onset and baseline
removed from the input: baseline = first 60 s, search everything after it):

| Run | \|error\| median; ≤ 5 s; ≤ 10 s | signed error median [range] | AUC median [range] | precision / recall |
|---|---|---|---|---|
| default, search from 30 s before the **expert** onset (optimistic) | 5.0 s; 12/24; 15/24 | −3.9 [−29.0, 59.5] | 0.98 [0.49, 1.00] | 0.33 / 0.63 |
| … spontaneous (18) / stimulation-induced (6) | 3.9 s; 12/18; 15/18 / 25.8 s; 0/6; 0/6 | | 0.98 / 0.90 | 0.36 / 0.71 ; 0 / 0 |
| `SEARCH_BEFORE_S=-1` (search from the end of ieeg-prep's baseline, which still ends 10 s before the expert onset) | 5.8 s; 11/24; 13/24 | −4.4 [−31.4, 59.5] | 0.98 [0.47, 1.00] | 0.33 / 0.60 |
| **onset-blind** (no `onset_s`, no baseline in the input) | 17.7 s; 9/24; 11/24 | −11.0 [−70.2, 23.5] | 0.97 [0.42, 1.00] | 0.29 / 0.71 |

These match the raw-clip runs above (default 5.0 s / AUC 0.98; unanchored 17.7 s / 0.97): the
channel ranking holds up without any onset information, the onset *time* needs a rough
clinical mark to be within seconds.

**Compared with the preprint.** Ojemann et al. evaluated on the same kind of data (HUP SEEG,
consensus of five epileptologists; 46 seizures of 23 patients) but report channel-level
metrics, not an onset-time error: for LiNDDA onset-channel AUROC 0.88 ± 0.06, AUPRC 0.47 ± 0.11,
and phi 0.58 at the per-seizure optimal threshold vs 0.35 at the fixed (pretrained) threshold;
in their large cohort LiNDDA's first detection came 9.8 ± 1.5 s after the clinical onset
("recruitment latency"). Our AUC (0.98) is higher than their 0.88, but measures something easier:
we rank channels by the score in the 5 s after *our* detected onset and count a bipolar channel
as an onset channel when either contact is, while they classify every channel's seizing state
in the onset window. Our onset is early rather than late on average because the search starts
30 s before the clinical mark and the k-th channel rule fires on pre-ictal activity.

## Runtime

`computeTypes: [standard, gpu]` (see *GPU*), 4 vCPU / 8 GiB, 4 h timeout. One recording at a time; signals
are read a block of 16 channels at a time and kept at 256 Hz (float32), so memory is ~2 GB per
hour of 128-channel data plus the model's working copies.

Measured in Docker (linux/amd64 **emulated on an M-series Mac**, `--cpus 4 --memory 4g`, with
other jobs running on the host):

| Model | Clips | Time per clip | cgroup `memory.peak` |
|---|---|---|---|
| LiNDDA (default) | 3 demo clips: 208 ch / 338 s, 116 ch / 386 s, 106 ch / 463 s | 38.8, 19.5, 16.9 s (88 s run) | 1.60 GB |
| NDD | 1 clip, 116 ch / 386 s | 28.9 s | 2.19 GB |
| ONCET | 1 clip, 116 ch / 386 s | 279 s | 1.84 GB |
| LiNDDA (default), no other load | all 24 ieeg-prep clips (72–204 channels) | 240 s for the run, ~10 s per clip | — |

Natively (arm64, same Mac) all 24 demo clips take ~2 min with LiNDDA (5–10 s each);
ONCET takes 1.5–7 min per clip natively, even after rewriting its dilated convolutions as
undilated ones (identical output, ~5× faster: PyTorch's CPU backend has no fast kernel for
dilated depthwise convolutions). 8 GiB leaves room for hour-long recordings with LiNDDA; for
ONCET on many long recordings use the gpu runtime or raise the timeout. The image is
3.8 GB compressed with CUDA PyTorch (0.43 GB for the CPU-only build; see *GPU*).

## GPU

Two variants of the app, built from one release (app.yml `variants`; the Dockerfile's
`TORCH_FLAVOUR` build argument picks the PyTorch build):

| Variant | Image | Runs on | Default `MODEL` |
|---|---|---|---|
| `cpu` (default) | CPU PyTorch, 0.43 GB compressed | a standard Fargate task (4 vCPU / 8 GiB) | `lindda` |
| `gpu` | CUDA 12.6 PyTorch, 3.8 GB compressed | the GPU tier, EC2 g4dn.xlarge (NVIDIA T4 16 GB; 4 vCPU / 14 GiB) | `oncet` |

A workflow step picks the variant; the parameters and ports are the same in both, so a step
can switch without breaking its links. `DEVICE=auto` (the default) puts the models on the GPU
when there is one and on the CPU otherwise, so the CUDA image also runs on CPU.

- **cpu** is for LiNDDA (5–40 s per clip), NDD and the slope detector; none of them gain
  from a GPU.
- **gpu** is for `MODEL=oncet`, DynaSD's pretrained convolutional classifier: ~90,000–110,000
  independent 1 s single-channel windows per clip through a small 1-D CNN. **Measured on dev:
  59 s for the ONCET step on the GPU against 2163 s on CPU (~37×)** for the same clips; the
  run is then dominated by reading and filtering the EDF.
- **Cost.** g4dn.xlarge is ~$0.53/h on demand, roughly 3× a 4 vCPU / 8 GiB Fargate task, so
  ONCET on the GPU costs a fraction of ONCET on CPU; LiNDDA on the GPU only costs more. The
  first GPU step waits for an EC2 instance and the image pull (about a minute when the
  compute node keeps a warm GPU instance, a few minutes from cold).
- **Image size.** Before the variants, one CUDA image served both runtimes and every CPU run
  pulled 3.8 GB (~94 s on Fargate). The cpu variant now pulls 0.43 GB. Locally,
  `make image-cpu` builds the cpu image and `make image` the CUDA one.

**What changes on the GPU.** The same models with the same weights and seeds; only the
arithmetic differs. The forecasters (LiNDDA, NDD) are initialised and their training batches
shuffled on the CPU (same `SEED`), then trained in float32 on the GPU; ONCET keeps its dilated
convolutions (cuDNN has fast kernels for them; on CPU they are rewritten as equivalent
undilated ones, see *Runtime*) with `cudnn.benchmark` on. float32 on a GPU differs from the CPU
in the last bits (different summation order, cuDNN algorithms): expect scores to agree to about
1e-5 relative (ONCET probabilities to ~1e-6 absolute), so `score` in `channel_onsets.csv` (4
decimals) can move in its last digit and an onset can move by one 0.5 s window when a smoothed
score sits within that rounding of the threshold, which is rare. For LiNDDA/NDD the ten training
epochs can amplify the rounding slightly more. GPU results are not bit-reproducible from run to
run (cuDNN autotuning); CPU results are, and are byte-identical to v0.1.0
(`tests/test_processor.py::test_cpu_results_match_v0_1_0`, against `tests/golden/`).

**Implementation notes.** DynaSD's ONCET moves a whole recording's windows to the device and
then gives that tensor to a `DataLoader(pin_memory=True)`, which cannot pin a CUDA tensor; the
app keeps the windows in host memory and copies one batch at a time (`detect.oncet_class`).
Scoring runs under `torch.no_grad()`. The cuda path is unit-tested by mocking
`torch.cuda.is_available` (there is no GPU on the build machine). Checked on the Mac (Docker,
amd64 emulation, `--cpus 4 --memory 4g`, no GPU): the CUDA image starts, logs
`DEVICE=auto: cuda not available, using cpu` (and a WARNING for `DEVICE=cuda`), and its outputs
are byte-identical to v0.1.0's image for LiNDDA on two demo clips and for ONCET on one (85
channels, 256 s: ~105 s of ONCET scoring either way, peak 1.4 GB). On CPU the CUDA build of
PyTorch is as fast as the CPU build (LiNDDA and ONCET timings within run-to-run noise); the cost
for CPU runs is the pull, not the run.

To measure on dev: run the workflow (or a one-step workflow) with dynasd `MODEL=oncet` and the
step's `executionTarget` `gpu`, then compare `timings.forward_s` per recording in
`dynasd/provenance.json` (and `device.used` = `cuda`) with a `standard` run of the same input.

## Running it locally

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
make test PYTHON=.venv/bin/python        # unit tests on a synthetic fixture
# ONCET's tests need its checkpoint in data/checkpoints/ONCET (else skipped), e.g. from the image:
#   docker cp $(docker create processor-seizure-dynasd:dev):/opt/dynasd/ONCET data/checkpoints/
make run PYTHON=.venv/bin/python         # Docker over the fixture -> data/output/
make run-demo DEMO=../workflow-epilepsy-ieeg/data/work/prep/ieeg     # or .../data/seizures/DATA
make run-demo IMAGE=processor-seizure-dynasd:dev TORCH_FLAVOUR=cpu      # the small CPU-only image
make run-demo DOCKER_GPU='--gpus all'                                 # on a host with an NVIDIA GPU
make validate TRUTH=../workflow-epilepsy-ieeg/data/seizures/DATA
```

## Limitations

- DynaSD finds where a recording **stops resembling its baseline**. Anything that does that
  counts: stimulation artefacts (the stimulation-induced demo seizures are its worst cases),
  movement, pre-ictal bursts, an earlier subclinical event. Give it a clean baseline and, when
  the clinical onset is known, keep the search window around it (`SEARCH_BEFORE_S`).
- The baseline must be interictal-like and come before the seizure; a recording that starts in
  a seizure, or a baseline full of artefacts, gives nonsense.
- The pretrained thresholds were calibrated by the authors on their own cohort (HUP; their
  preprocessing). They transfer reasonably to this demo (from the same centre), but not
  necessarily to other amplifiers, montages or ECoG; `THRESHOLD` can be set directly.
- Onset times are quantised to the 0.5 s window stride and smoothed over `FILTER_W`; expect
  seconds, not milliseconds. `recruited` is a hard cut (precision is low on the demo, see
  above); rank by `score` instead when possible.
- Not every DynaSD model is offered: WVNT needs TensorFlow (not bundled: it would more than
  double the image); GIN (no pretrained threshold for its default sequence length), IMPRINT
  (calibrated on a 0.125 s stride) and HFER (its own threshold API) are not wired up.
- The preprint is not yet peer reviewed.

## Licences

Apache-2.0 for this app's code (`LICENSE`). The image bundles DynaSD (MIT; the author, Will
Ojemann, confirmed use of the code and the released checkpoints) at a pinned commit, its ONCET
checkpoint (pinned URL + SHA-256), PyTorch (CUDA 12.6 build, BSD-3-Clause) with NVIDIA's CUDA
libraries (NVIDIA EULA, redistributable), MNE-Python, NumPy,
SciPy, scikit-learn and pandas (BSD); see `THIRD_PARTY.md`.

## Citations

DynaSD:

> Ojemann WKS, Xu Z, Shi H, Walsh K, Pattnaik A, Sinha N, Lavelle S, Aguila C, Gallagher R,
> Revell A, LaRocque JJ, Korzun J, Kulick-Soper CV, Zhou DJ, Galer PD, Sinha SR, Shinohara RT,
> Davis KA, Litt B, Conrad EC. **Unsupervised seizure annotation and detection with neural
> dynamic divergence.** medRxiv (2026, preprint posted 17 Feb 2026).
> doi:10.64898/2026.02.15.26346325.

> DynaSD, https://github.com/wojemann/DynaSD, commit 48b2b959605a37fb03a4d99ae6fcc1459c04cc37 (MIT).

`MODEL=absslp`:

> Schindler K, Leung H, Elger CE, Lehnertz K. **Assessing seizure dynamics by analysing the
> correlation structure of multichannel intracranial EEG.** Brain 130(1):65–77 (2007).
> doi:10.1093/brain/awl304, PMID 17082199.

Reading EDF / BrainVision / FIF:

> Gramfort A, Luessi M, Larson E, Engemann DA, Strohmeier D, Brodbeck C, Goj R, Jas M, Brooks T,
> Parkkonen L, Hämäläinen M. **MEG and EEG data analysis with MNE-Python.** Front Neurosci 7:267
> (2013). doi:10.3389/fnins.2013.00267, PMID 24431986.

PyTorch (bundled, runs the forecasters and ONCET):

> Paszke A, Gross S, Massa F, Lerer A, Bradbury J, Chanan G, Killeen T, Lin Z, Gimelshein N,
> Antiga L, Desmaison A, Köpf A, Yang E, DeVito Z, Raison M, Tejani A, Chilamkurthy S, Steiner B,
> Fang L, Bai J, Chintala S. **PyTorch: An Imperative Style, High-Performance Deep Learning
> Library.** arXiv (2019). doi:10.48550/arXiv.1912.01703.

Demo data:

> Ojemann WK, Zhou D, Armstrong C, Sinha N, Litt B, Conrad E. **Multi-Expert Seizure
> Annotation.** Pennsieve Discover (2026). doi:10.26275/dely-fzwx (CC BY 4.0).

DOIs were checked against Crossref (articles, the medRxiv preprint) and DataCite (dataset,
arXiv): title, authors and year. Crossref lists only the first author of the MNE-Python paper;
the full list is from PubMed (PMID 24431986). Schindler et al. was published online in 2006
(Crossref's date) in the January 2007 issue.
