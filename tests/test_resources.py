"""THREADS=0 resolves from the cgroup limits (fake cgroup trees), not the host's CPU count."""
from __future__ import annotations

import os

import pytest

from processor import resources as R


@pytest.fixture
def host(monkeypatch):
    """A 15-CPU host with 64 GB, as the container sees it through os.*."""
    monkeypatch.setattr(os, "cpu_count", lambda: 15)
    monkeypatch.setattr(os, "sched_getaffinity", lambda pid: set(range(15)), raising=False)
    monkeypatch.setattr(R, "machine_memory_bytes", lambda: 64 * 2**30)


def cgroup(tmp_path, **files):
    for name, text in files.items():
        f = tmp_path / name.replace("__", "/")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text + "\n")
    return tmp_path


def test_v2_quota_and_memory_cap(tmp_path, host):
    root = cgroup(tmp_path, **{"cpu.max": "400000 100000", "memory.max": str(3 * 2**30)})
    assert R.cgroup_cpus(root) == 4 and R.cgroup_memory_bytes(root) == 3 * 2**30
    n, why = R.resolve_workers(0, per_worker_mb=320, reserve_mb=850, root=root)
    assert n == 4 and "cgroup CPU quota 4" in why
    # 3 GB holds (3072 - 850) // 700 = 3 workers of 700 MB: memory wins over the 4 CPUs
    n, why = R.resolve_workers(0, per_worker_mb=700, reserve_mb=850, root=root)
    assert n == 3 and "capped by memory" in why


def test_v2_no_limits_falls_back_to_affinity(tmp_path, host):
    root = cgroup(tmp_path, **{"cpu.max": "max 100000", "memory.max": "max"})
    assert R.cgroup_cpus(root) is None and R.cgroup_memory_bytes(root) is None
    n, why = R.resolve_workers(0, per_worker_mb=320, reserve_mb=850, root=root)
    assert n == 15 and "CPU affinity" in why


def test_v1_fractional_quota_and_limit(tmp_path, host):
    root = cgroup(tmp_path, **{"cpu__cpu.cfs_quota_us": "150000", "cpu__cpu.cfs_period_us": "100000",
                               "memory__memory.limit_in_bytes": str(2**63 - 4096)})
    assert R.cgroup_cpus(root) == 1.5 and R.cgroup_memory_bytes(root) is None
    assert R.resolve_workers(0, 320, 850, root)[0] == 1
    root2 = cgroup(tmp_path / "b", **{"cpu__cpu.cfs_quota_us": "-1", "cpu__cpu.cfs_period_us": "100000"})
    assert R.cgroup_cpus(root2) is None


def test_explicit_threads_kept_with_a_note(tmp_path, host):
    root = cgroup(tmp_path, **{"cpu.max": "200000 100000", "memory.max": str(3 * 2**30)})
    n, why = R.resolve_workers(15, 320, 850, root)
    assert n == 15 and "more than the 6" in why
    assert R.resolve_workers(2, 320, 850, root) == (2, "THREADS=2")


def test_no_cgroup_at_all(tmp_path, host, monkeypatch):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    n, why = R.resolve_workers(0, 320, 850, tmp_path / "none")
    assert n == 15 and "os.cpu_count()" in why
