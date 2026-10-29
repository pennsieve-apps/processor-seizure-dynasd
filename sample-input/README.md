# sample-input

Real input for this app is the `ieeg` output of ieeg-prep: a BIDS-iEEG derivative with
`recordings.csv` and, per recording, an EDF plus `_channels.tsv`, `_events.tsv` and
`_ieeg.json` (see `docs/conventions.md` in workflow-epilepsy-ieeg). Any plain BIDS-iEEG
dataset or folder of EDF files works too.

The workflow's demo data are public (no login):

> Ojemann WK, Zhou D, Armstrong C, Sinha N, Litt B, Conrad E. Multi-Expert Seizure Annotation.
> Pennsieve Discover dataset 530 v3 (2026). doi:10.26275/dely-fzwx (CC BY 4.0)

24 SEEG seizure clips from 6 patients (HUP224, HUP230, HUP235, HUP238, HUP250, HUP257; four
seizures each, one of them stimulation-induced), 1024 Hz, 85-230 contacts, each with 120 s
before the clinically marked onset, plus `annotations.tsv` with the onset time and channels
3-5 epileptologists marked and their consensus. Fetch them with

```sh
cd ../workflow-epilepsy-ieeg && scripts/fetch-sample-data.sh     # -> data/seizures/ (~2.4 GB)
```

then run ieeg-prep on them (its output lands in `data/work/prep/ieeg/`), or point this app at
the raw clips directly (they are re-referenced to bipolar here, and the events `ictal` row is
used as the onset):

```sh
make run-demo DEMO=../workflow-epilepsy-ieeg/data/work/prep/ieeg      # or .../data/seizures/DATA
make validate TRUTH=../workflow-epilepsy-ieeg/data/seizures/DATA      # scores against annotations.tsv
```

For tests and `make run`, `tests/make_fixture.py` writes a small synthetic input instead
(`make fixture` -> `data/fixture/`): a two-recording ieeg-prep derivative (6 bipolar channels,
256 Hz, 150 s, a seizure starting at 90 s on two channels and spreading to a third), a plain
referential BIDS recording and a broken EDF.
