"""
Unit tests on a synthetic fixture (tests/make_fixture.py); no Docker, no real data.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from processor import detect as D
from processor import ieeg as I
from processor import main as M
from tests import make_fixture

PARAMS = ("MODEL", "THRESHOLD", "FILTER_W", "RWIN_SIZE", "RWIN_REQ", "MIN_CHANNELS", "ONSET_WINDOW_S",
          "SCORE_WINDOW_S", "SEARCH_BEFORE_S", "BASELINE_S", "SEED", "DEVICE", "BATCH", "DYNASD_CHECKPOINTS")
GOLDEN = Path(__file__).parent / "golden"
# ONCET's checkpoint: DYNASD_CHECKPOINTS, else data/checkpoints/ (copy /opt/dynasd/ONCET out of the image)
CHECKPOINTS = Path(__import__("os").environ.get("DYNASD_CHECKPOINTS") or Path(__file__).parent.parent / "data" / "checkpoints")


@pytest.fixture(scope="module")
def fixture_dir(tmp_path_factory) -> Path:
    return make_fixture.make(tmp_path_factory.mktemp("fixture"))


def run(monkeypatch, input_dir: Path, output_dir: Path, **env) -> int:
    for k in PARAMS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return M.main()


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


# --- input ------------------------------------------------------------------------

def test_find_recordings_derivative(fixture_dir):
    recs = I.find_recordings(fixture_dir / "linked")
    assert [r.recording_id for r in recs] == ["sub-01_task-ictal_run-01", "sub-01_task-ictal_run-02"]
    r = recs[0]
    assert r.source == "derivative" and r.montage == "bipolar"
    assert r.onset_s == make_fixture.ONSET and r.baseline == make_fixture.BASELINE and r.line_freq == 60
    raw = I.open_raw(r)
    chans, info = I.brain_channels(r, raw)
    assert [c.name for c in chans] == ["A01-A02", "A02-A03", "A03-A04", "B01-B02", "B02-B03", "B03-B04"]
    assert chans[0].labels["onset"] is True and chans[3].labels["onset"] is False
    assert info["montage"] == "bipolar (as given)"


def test_find_recordings_plain_bipolar(fixture_dir):
    recs = {r.recording_id: r for r in I.find_recordings(fixture_dir / "plain")}
    assert set(recs) == {"sub-02_task-ictal", "sub-02_task-broken"}
    r = recs["sub-02_task-ictal"]
    assert r.source == "bids" and r.onset_s == make_fixture.ONSET and r.baseline is None
    chans, info = I.brain_channels(r, I.open_raw(r))
    assert [c.name for c in chans] == ["A01-A02", "A02-A03", "A03-A04", "B01-B02", "B02-B03", "B03-B04"]
    assert chans[0].contacts == ("A01", "A02")
    assert info["montage"] == "bipolar (neighbouring contacts)"
    # the bipolar signal is the difference of the two contacts
    raw = I.open_raw(r)
    (_, x), = list(I.iter_blocks(raw, chans[:1], start=0, stop=512))
    a = raw.get_data(picks=["A01", "A02"], start=0, stop=512) * 1e6
    assert np.allclose(x[0], a[0] - a[1])


def test_bad_channels_and_types(tmp_path):
    out = make_fixture.make(tmp_path)
    ch = out / "plain" / "sub-02" / "ieeg" / "sub-02_task-ictal_channels.tsv"
    rows = I.read_tsv(ch)
    rows[1]["status"] = "bad"        # A02 bad -> A01-A02 and A02-A03 disappear
    rows[7]["type"] = "ECG"          # B04 not brain -> B03-B04 disappears
    make_fixture.write_tsv(ch, rows)
    r = [r for r in I.find_recordings(out / "plain") if r.recording_id == "sub-02_task-ictal"][0]
    chans, info = I.brain_channels(r, I.open_raw(r))
    assert [c.name for c in chans] == ["A03-A04", "B01-B02", "B02-B03"]
    assert info["dropped"]["A02"].startswith("status bad") and info["dropped"]["B04"] == "type ECG"


def test_baseline_interval():
    r = I.Recording("x", Path("x.edf"), "x.edf", "plain")
    assert I.baseline_interval(r, 300, 60) == ((0.0, 60.0), "start of recording")
    r.onset_s = 120.0
    assert I.baseline_interval(r, 300, 60) == ((0.0, 60.0), "before onset_s")
    r.onset_s = 400.0
    assert I.baseline_interval(r, 600, 60) == ((280.0, 340.0), "before onset_s")
    r.baseline = (10.0, 50.0)
    assert I.baseline_interval(r, 600, 60) == ((10.0, 50.0), "recording")


# --- detection ----------------------------------------------------------------------

def test_preprocess_resamples_to_256():
    fs = 1024.0
    t = np.arange(int(10 * fs)) / fs
    x = np.vstack([np.sin(2 * np.pi * 20 * t), np.sin(2 * np.pi * 60 * t)]) * 50
    y = D.preprocess(x, fs, 60.0)
    assert y.shape == (2, 2560) and y.dtype == np.float32
    mid = slice(500, 2000)
    assert np.std(y[0, mid]) > 25           # 20 Hz passes
    assert np.std(y[1, mid]) < 2            # 60 Hz notched


def test_undilate_is_exact():
    torch = pytest.importorskip("torch")
    conv = torch.nn.Conv1d(4, 4, kernel_size=5, dilation=3, padding=6, groups=4, bias=False)
    net = torch.nn.Sequential(conv)
    x = torch.randn(2, 4, 64)
    with torch.no_grad():
        before = net(x)
        D.undilate(net)
        after = net(x)
    assert conv.dilation == (1,) and conv.kernel_size == (13,)
    assert torch.allclose(before, after, atol=1e-5)


def test_rising_onsets_ignores_runs_already_on():
    import pandas as pd

    clf = pd.DataFrame({"a": [1, 1, 0, 0, 1, 1], "b": [0, 0, 0, 1, 1, 1], "c": [0] * 6},
                       index=[10.0, 10.5, 11.0, 11.5, 12.0, 12.5])
    assert D.rising_onsets(clf) == {"a": 12.0, "b": 11.5, "c": None}


# --- end to end -----------------------------------------------------------------------

def test_main_on_derivative(fixture_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run(monkeypatch, fixture_dir / "linked", out) == 0
    # everything under a folder named after the method (one link of a fan-in port)
    assert [q.name for q in out.iterdir()] == ["dynasd"]
    onsets = read_csv(out / "dynasd" / "seizure_onsets.csv")
    assert [r["recording_id"] for r in onsets] == ["sub-01_task-ictal_run-01", "sub-01_task-ictal_run-02"]
    for r in onsets:
        assert r["method"] == "dynasd-lindda"
        # 3rd channel to start (MIN_CHANNELS=3) is A03-A04, recruited 8 s after the onset at 90 s
        assert make_fixture.ONSET - 5 <= float(r["onset_s"]) <= make_fixture.ONSET + 12
    rows = read_csv(out / "dynasd" / "channel_onsets.csv")
    assert set(rows[0]) == {"recording_id", "channel", "onset_s", "recruited", "score", "method"}
    for r in rows:
        if r["channel"].startswith("B"):
            assert r["recruited"] == "false" and r["onset_s"] == "" and float(r["score"]) < 1
        if r["channel"] in ("A01-A02", "A02-A03"):
            assert r["recruited"] == "true" and float(r["score"]) > 1
            assert make_fixture.ONSET - 5 <= float(r["onset_s"]) <= make_fixture.ONSET + 8
    prov = json.loads((out / "dynasd" / "provenance.json").read_text())
    assert prov["app"] == "processor-seizure-dynasd" and prov["parameters"]["model"] == "lindda"
    st = prov["recordings"]["sub-01_task-ictal_run-01"]
    assert st["status"] == "ok" and st["baseline_source"] == "recording" and st["search_start_s"] == 60.0


def test_main_plain_with_broken_file(fixture_dir, tmp_path, monkeypatch):
    out = tmp_path / "out"
    assert run(monkeypatch, fixture_dir / "plain", out, MODEL="absslp", MIN_CHANNELS="1") == 0
    prov = json.loads((out / "dynasd" / "provenance.json").read_text())
    assert prov["recordings"]["sub-02_task-broken"]["status"] == "failed"
    assert prov["recordings"]["sub-02_task-ictal"]["status"] == "ok"
    assert prov["recordings"]["sub-02_task-ictal"]["baseline_source"] == "before onset_s"
    rows = read_csv(out / "dynasd" / "channel_onsets.csv")
    assert {r["method"] for r in rows} == {"dynasd-absslp"}


def test_whole_input_root_and_reproducible(fixture_dir, tmp_path, monkeypatch):
    """Everything merged at the root (no links): both layouts are found; same SEED, same output."""
    a, b = tmp_path / "a", tmp_path / "b"
    assert run(monkeypatch, fixture_dir, a) == 0
    assert run(monkeypatch, fixture_dir, b) == 0
    assert len(read_csv(a / "dynasd" / "seizure_onsets.csv")) == 3
    assert (a / "dynasd" / "channel_onsets.csv").read_text() == (b / "dynasd" / "channel_onsets.csv").read_text()


def test_no_input_fails(tmp_path, monkeypatch):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(monkeypatch, empty, tmp_path / "out") == 1


@pytest.mark.parametrize("env", [{"MODEL": "wvnt"}, {"THRESHOLD": "high"}, {"RWIN_REQ": "6"}, {"MIN_CHANNELS": "x"},
                                 {"DEVICE": "gpu"}, {"BATCH": "-1"}])
def test_bad_parameters(fixture_dir, tmp_path, monkeypatch, env):
    assert run(monkeypatch, fixture_dir / "linked", tmp_path / "out", **env) == 2


# --- device (DEVICE=auto|cuda|cpu) -----------------------------------------------------
# There is no GPU in CI or on a Mac: the cuda path is tested by mocking torch.cuda.

@pytest.fixture
def fake_gpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Tesla T4")
    return torch


def test_choose_device_without_gpu(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert D.choose_device("cpu")[:2] == ("cpu", "cpu")
    assert D.choose_device("auto")[:2] == ("auto", "cpu")
    assert D.choose_device("")[:2] == ("auto", "cpu")
    req, dev, why = D.choose_device("CUDA")          # asked for cuda, none there: falls back
    assert (req, dev) == ("cuda", "cpu") and "not available" in why
    with pytest.raises(ValueError):
        D.choose_device("gpu")


def test_choose_device_with_gpu(fake_gpu):
    assert D.choose_device("auto") == ("auto", "cuda", "DEVICE=auto: using cuda: Tesla T4")
    assert D.choose_device("cuda")[1] == "cuda"
    assert D.choose_device("cpu")[1] == "cpu"        # cpu is honoured even with a GPU


def test_choose_device_broken_cuda_runtime(monkeypatch):
    torch = pytest.importorskip("torch")

    def boom():
        raise RuntimeError("driver too old")

    monkeypatch.setattr(torch.cuda, "is_available", boom)
    req, dev, why = D.choose_device("auto")
    assert dev == "cpu" and "driver too old" in why


def test_oncet_batch():
    assert D.oncet_batch(D.Params(device="cpu")) == 128      # v0.1.0's batch on CPU
    assert D.oncet_batch(D.Params(device="cuda")) == 8192
    assert D.oncet_batch(D.Params(device="cuda", batch=1000)) == 1000


@pytest.mark.parametrize("model", ["lindda", "ndd"])
def test_forecasters_on_cuda(fake_gpu, model):
    """LiNDDA and NDD get use_cuda; their (training) batch sizes do not depend on the device."""
    m_cpu = D.build_model(D.Params(model=model, device="cpu"), 4)
    m_gpu = D.build_model(D.Params(model=model, device="cuda"), 4)
    assert m_cpu.device.type == "cpu" and m_gpu.device.type == "cuda"
    assert m_cpu.batch_size == m_gpu.batch_size


@pytest.fixture
def oncet_checkpoints():
    if not (CHECKPOINTS / "ONCET" / "best_model.pth").exists():
        pytest.skip(f"no ONCET checkpoint under {CHECKPOINTS}")
    return str(CHECKPOINTS)


def test_oncet_on_cuda_is_built_for_the_gpu(fake_gpu, monkeypatch, oncet_checkpoints):
    """
    DEVICE=cuda: ONCET gets device cuda and the GPU batch, keeps its dilated convolutions (cuDNN),
    and its windows stay in host memory (upstream would pin a CUDA tensor and fail).
    """
    torch = fake_gpu
    import pandas as pd

    load = torch.load
    monkeypatch.setattr(torch, "load", lambda f, map_location=None, **kw: load(f, map_location="cpu", **kw))
    monkeypatch.setattr(torch.nn.Module, "to", lambda self, *a, **kw: self)   # no GPU to move to
    m = D.build_model(D.Params(model="oncet", device="cuda", checkpoint_dir=oncet_checkpoints), 3)
    assert m.device == "cuda" and m.batch_size == 8192
    convs = [c for c in m.model.modules() if isinstance(c, torch.nn.Conv1d)]
    assert any(c.dilation[0] > 1 for c in convs)
    x = pd.DataFrame(np.random.default_rng(0).standard_normal((256 * 4, 3)).astype(np.float32), columns=list("abc"))
    m.fit(x)
    t = m._prepare_oncet_segment(x)
    assert t.device.type == "cpu" and t.shape == (7 * 3, 1, 256) and m.device == "cuda"


def test_oncet_on_cpu_unchanged(oncet_checkpoints):
    m = D.build_model(D.Params(model="oncet", device="cpu", checkpoint_dir=oncet_checkpoints), 3)
    assert m.device == "cpu" and m.batch_size == 128
    assert all(c.dilation == (1,) for c in m.model.modules() if c.__class__.__name__ == "Conv1d")


def test_device_in_provenance_and_cuda_falls_back(fixture_dir, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "out"
    assert run(monkeypatch, fixture_dir / "linked", out, MODEL="absslp", DEVICE="cuda") == 0
    prov = json.loads((out / "dynasd" / "provenance.json").read_text())
    assert prov["device"]["requested"] == "cuda" and prov["device"]["used"] == "cpu"
    assert prov["parameters"]["DEVICE"] == "cuda"


# --- CPU results unchanged since v0.1.0 ---------------------------------------------------
# tests/golden/<model>/ are v0.1.0's outputs on the synthetic fixture (macOS arm64, torch 2.11.0).
# ONCET flags no channel on the synthetic burst at its pretrained threshold, so it runs with a
# fixed THRESHOLD=0.99 and MIN_CHANNELS=1 to give scores to compare.

GOLDEN_ENV = {"lindda": {}, "ndd": {}, "absslp": {}, "oncet": {"THRESHOLD": "0.99", "MIN_CHANNELS": "1"}}


@pytest.mark.parametrize("model", list(GOLDEN_ENV))
@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_cpu_results_match_v0_1_0(fixture_dir, tmp_path, monkeypatch, model, device):
    env = dict(GOLDEN_ENV[model], MODEL=model, DEVICE=device)
    if model == "oncet":
        if not (CHECKPOINTS / "ONCET" / "best_model.pth").exists():
            pytest.skip(f"no ONCET checkpoint under {CHECKPOINTS}")
        env["DYNASD_CHECKPOINTS"] = str(CHECKPOINTS)
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    out = tmp_path / "out"
    assert run(monkeypatch, fixture_dir / "linked", out, **env) == 0
    for name in ("seizure_onsets.csv", "channel_onsets.csv"):
        got, want = read_csv(out / "dynasd" / name), read_csv(GOLDEN / model / name)
        assert len(got) == len(want)
        for g, w in zip(got, want):
            assert {k: v for k, v in g.items() if k != "score"} == {k: v for k, v in w.items() if k != "score"}
            if "score" in w:
                assert (g["score"] == w["score"]) or abs(float(g["score"]) - float(w["score"])) <= 2e-4
