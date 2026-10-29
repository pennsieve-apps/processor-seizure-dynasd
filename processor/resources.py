"""
How many CPUs and how much memory this container may use: its cgroup limits, not the host's.

os.cpu_count() reports the host (or VM) CPUs, which inside a CPU-limited container (Docker
--cpus, an ECS/Fargate task) can be several times the allocation; starting that many worker
processes under a memory limit gets the run OOM-killed without a word. THREADS=0 ("one per
vCPU") is therefore resolved from the cgroup CPU quota (v2 cpu.max, v1 cpu.cfs_quota_us /
cpu.cfs_period_us), else the CPU affinity, else os.cpu_count(), and capped so that the
estimated per-worker memory fits the cgroup memory limit (v2 memory.max, v1
memory.limit_in_bytes), else the machine's memory.

The same file is kept in every epilepsy iEEG app that runs workers in parallel.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

CGROUP = Path("/sys/fs/cgroup")
UNLIMITED = 1 << 60  # cgroup v1 writes a huge number for "no limit"


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def cgroup_cpus(root: Path = CGROUP) -> float | None:
    """CPUs allowed by the cgroup CPU quota, or None when there is no quota."""
    v2 = _read(root / "cpu.max")
    if v2:
        quota, _, period = v2.partition(" ")
        if quota != "max":
            try:
                return int(quota) / int(period or 100000)
            except (ValueError, ZeroDivisionError):
                return None
        return None
    for d in (root / "cpu", root / "cpu,cpuacct", root):
        q, p = _read(d / "cpu.cfs_quota_us"), _read(d / "cpu.cfs_period_us")
        if q and p:
            try:
                q_, p_ = int(q), int(p)
            except ValueError:
                return None
            return q_ / p_ if q_ > 0 and p_ > 0 else None
    return None


def cgroup_memory_bytes(root: Path = CGROUP) -> int | None:
    """The cgroup memory limit in bytes, or None when there is none."""
    for f in (root / "memory.max", root / "memory" / "memory.limit_in_bytes", root / "memory.limit_in_bytes"):
        v = _read(f)
        if v is None:
            continue
        if v == "max":
            return None
        try:
            n = int(v)
        except ValueError:
            return None
        return n if 0 < n < UNLIMITED else None
    return None


def machine_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def available_cpus(root: Path = CGROUP) -> tuple[int, str]:
    """Whole CPUs this process may use (at least 1) and where the number comes from."""
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    quota = cgroup_cpus(root)
    if quota is not None:
        n = max(1, math.floor(quota + 1e-9))
        if affinity:
            n = min(n, affinity)
        return n, f"cgroup CPU quota {quota:g}"
    if affinity:
        return affinity, "CPU affinity"
    return max(1, os.cpu_count() or 1), "os.cpu_count()"


def resolve_workers(requested: int, per_worker_mb: float, reserve_mb: float,
                    root: Path = CGROUP) -> tuple[int, str]:
    """
    The number of parallel workers and why. requested > 0 is used as given (with a note when the
    memory estimate says it will not fit); 0 = one per available CPU, capped so that reserve_mb
    (the main process) plus per_worker_mb per worker fits the memory limit.
    """
    limit = cgroup_memory_bytes(root)
    mem_src = "cgroup memory limit"
    if limit is None:
        limit, mem_src = machine_memory_bytes(), "machine memory"
    fit = None
    if limit:
        fit = max(1, int((limit / 2**20 - reserve_mb) // per_worker_mb))
    mem_note = (f"{mem_src} {limit / 2**20:.0f} MB fits {fit} at ~{per_worker_mb:g} MB each "
                f"+ {reserve_mb:g} MB") if fit else "memory limit unknown"
    if requested > 0:
        why = f"THREADS={requested}"
        if fit is not None and requested > fit:
            why += f" (more than the {fit} the memory estimate allows: {mem_note})"
        return requested, why
    cpus, cpu_src = available_cpus(root)
    if fit is not None and fit < cpus:
        return fit, f"THREADS=0: {cpus} CPU(s) ({cpu_src}), capped by memory: {mem_note}"
    return cpus, f"THREADS=0: {cpus} CPU(s) ({cpu_src}); {mem_note}"
