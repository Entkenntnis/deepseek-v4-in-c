# SPDX-License-Identifier: Apache-2.0
"""Per-core utilization and package temperature from procfs/sysfs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

from .defaults import clamp


class CpuMixin:
    def _discover_cpu_temp(self) -> None:
        """Find the most likely CPU package sensor in sysfs, once."""
        self.cpu_temp_checked = True
        best_score = 0
        best_path: Optional[str] = None

        for hwmon in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
            name = self._read_text(str(hwmon / "name")).lower()
            for sensor in sorted(hwmon.glob("temp*_input")):
                label_path = sensor.with_name(
                    sensor.name.replace("_input", "_label"))
                label = self._read_text(str(label_path)).lower()
                text = f"{name} {label}"
                if "package" in text:
                    score = 100
                elif "tctl" in text or "tdie" in text:
                    score = 90
                elif "x86_pkg_temp" in text or "pkg" in text:
                    score = 80
                elif name in ("coretemp", "k10temp", "zenpower"):
                    score = 70
                elif "core" in label or "cpu" in text or "tcpu" in text:
                    score = 60
                elif "acpitz" in text:
                    score = 10
                else:
                    score = 0
                if score > best_score:
                    best_score = score
                    best_path = str(sensor)

        for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
            ztype = self._read_text(str(zone / "type")).lower()
            if "x86_pkg_temp" in ztype or "pkg" in ztype:
                score = 80
            elif "tcpu" in ztype:
                score = 75
            elif "cpu" in ztype:
                score = 70
            elif "acpitz" in ztype:
                score = 10
            else:
                score = 0
            if score > best_score:
                best_score = score
                best_path = str(zone / "temp")

        self.cpu_temp_path = best_path

    def _read_cpu_temp(self) -> Optional[float]:
        if not self.cpu_temp_checked:
            self._discover_cpu_temp()
        if not self.cpu_temp_path:
            return None
        text = self._read_text(self.cpu_temp_path)
        try:
            value = int(text)
        except ValueError:
            try:
                value = int(float(text))
            except ValueError:
                self.cpu_temp_path = None
                return None
        if value <= 0:
            return None
        # hwmon and thermal_zone temperatures are millidegrees Celsius.
        return value / 1000.0 if value > 1000 else float(value)

    @staticmethod
    def _read_cpu_times() -> dict[int, tuple[int, int]]:
        """Busy/total jiffies for cores 0-3 from /proc/stat."""
        out: dict[int, tuple[int, int]] = {}
        try:
            lines = Path("/proc/stat").read_text().splitlines()
        except OSError:
            return out
        for line in lines:
            parts = line.split()
            if len(parts) < 5 or not parts[0].startswith("cpu"):
                continue
            if parts[0] == "cpu":
                continue
            try:
                core = int(parts[0][3:])
                values = [int(x) for x in parts[1:]]
            except ValueError:
                continue
            if core > 3 or len(values) < 4:
                continue
            idle = values[3] + (values[4] if len(values) > 4 else 0)
            out[core] = (sum(values) - idle, sum(values))
        return out

    def update_cpu(self) -> None:
        """Sample per-core utilization and the package temperature."""
        now = time.monotonic()
        current = self._read_cpu_times()
        if self.cpu_prev is not None and now - self.cpu_prev_t >= 0.2:
            for core in range(4):
                before = self.cpu_prev.get(core)
                after = current.get(core)
                if not before or not after:
                    self.core_pct[core] = None
                    continue
                busy = after[0] - before[0]
                total = after[1] - before[1]
                self.core_pct[core] = (
                    clamp(busy / total * 100.0, 0.0, 100.0)
                    if total > 0 else self.core_pct[core])
        self.cpu_prev = current
        self.cpu_prev_t = now
        self.cpu_temp_c = self._read_cpu_temp()
