"""Optional CPU / GPU / VRAM snapshots for tracker benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import resource
import subprocess


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    cpu_percent: float | None
    cpu_time_seconds: float
    gpu_memory_mb: float | None
    rss_mb: float | None


class ResourceSampler:
    def snapshot(self) -> ResourceSnapshot:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        cpu_time = float(usage.ru_utime + usage.ru_stime)
        rss_mb = float(usage.ru_maxrss) / 1024.0
        if rss_mb > 10_000:
            rss_mb = float(usage.ru_maxrss) / (1024.0 * 1024.0)
        return ResourceSnapshot(
            cpu_percent=_cpu_percent(),
            cpu_time_seconds=cpu_time,
            gpu_memory_mb=_gpu_memory_mb(),
            rss_mb=rss_mb,
        )


def _cpu_percent() -> float | None:
    stat = Path("/proc/self/stat")
    if not stat.is_file():
        return None
    try:
        load = os.getloadavg()[0]
        return float(load)
    except OSError:
        return None


def _gpu_memory_mb() -> float | None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            stderr=subprocess.DEVNULL,
            timeout=1.0,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    line = output.strip().splitlines()
    if not line:
        return None
    try:
        return float(line[0].split(",")[0])
    except ValueError:
        return None
