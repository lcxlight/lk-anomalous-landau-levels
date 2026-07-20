"""Zoomed Stage 3 traces for the first three high-field oscillation windows.

This figure documents the field windows selected for the next thermal-damping
analysis.  It zooms the smooth-detrended chi_T and M_no0LL traces to W0-W2 for
each density, shades three zero-crossing-aligned windows, and marks the local
max/min points of the lowest-temperature trace in each window.
"""
from __future__ import annotations

from pathlib import Path
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lambda1_deep_response import (
    response_for_rho,
)
from step4_lambda1_stage3_thermal_damping_traces import (
    RHO_VALUES,
    T_GRID,
    expected_frequency,
    oscillatory_part,
)


INV_B_MIN = 10.0
INV_B_MAX = 125.0
N_INV_B = 420
N_WINDOWS = 3


def zero_crossings_with_slope(x: np.ndarray, y: np.ndarray,
                              xmin: float = INV_B_MIN
                              ) -> list[tuple[float, int]]:
    """Linear-interpolated zero crossings and their crossing direction.

    Direction is +1 for upward crossing and -1 for downward crossing.
    """
    crossings: list[tuple[float, int]] = []
    valid = np.isfinite(x) & np.isfinite(y)
    xx = x[valid]
    yy = y[valid]
    for i in range(xx.size - 1):
        x0, x1 = float(xx[i]), float(xx[i + 1])
        y0, y1 = float(yy[i]), float(yy[i + 1])
        if x1 < xmin:
            continue
        if y0 == 0.0:
            xc = x0
        elif y0 * y1 > 0.0 or y1 == y0:
            continue
        else:
            xc = x0 - y0 * (x1 - x0) / (y1 - y0)
        if xc < xmin:
            continue
        direction = 1 if y1 > y0 else -1
        if not crossings or abs(xc - crossings[-1][0]) > 1e-6:
            crossings.append((float(xc), direction))
    return crossings


def zero_aligned_windows(x: np.ndarray, y: np.ndarray, freq: float,
                         n_windows: int = N_WINDOWS
                         ) -> list[tuple[float, float, str]]:
    """Choose full cycles bounded by zero crossings with the same slope.

    If the zero-crossing search fails, fall back to nominal frequency windows.
    """
    crossings = zero_crossings_with_slope(x, y)
    windows: list[tuple[float, float, str]] = []
    used_until = -np.inf
    for i, (x0, sign0) in enumerate(crossings):
        if x0 < used_until:
            continue
        for j in range(i + 1, len(crossings)):
            x1, sign1 = crossings[j]
            if sign1 == sign0:
                if x1 > x0 and x1 <= float(np.max(x)):
                    windows.append((x0, x1, "zero"))
                    used_until = x1
                break
        if len(windows) >= n_windows:
            break

    if len(windows) < n_windows:
        period = 1.0 / freq
        start = INV_B_MIN
        while len(windows) < n_windows:
            i = len(windows)
            windows.append((start + i * period, start + (i + 1) * period, "freq"))
    return windows[:n_windows]


def local_extrema(x: np.ndarray, y: np.ndarray, x0: float, x1: float
                  ) -> tuple[float, float, float, float]:
    mask = (x >= x0) & (x <= x1)
    xx = x[mask]
    yy = y[mask]
    if xx.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    max_i = int(np.nanargmax(yy))
    min_i = int(np.nanargmin(yy))
    return float(xx[max_i]), float(yy[max_i]), float(xx[min_i]), float(yy[min_i])


def set_symmetric_ylim(ax, values: list[np.ndarray]) -> None:
    finite = np.concatenate([v[np.isfinite(v)] for v in values if np.any(np.isfinite(v))])
    if finite.size == 0:
        return
    vmax = float(np.max(np.abs(finite)))
    if vmax <= 0:
        vmax = 1.0
    ax.set_ylim(-1.12 * vmax, 1.12 * vmax)


def main() -> None:
    outdir = (
        ROOT
        / "figures"
        / "default_a1.115_c0.215_E0_0.0849_Lambda_1.0"
        / "step4_lk_theory_tests"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    inv_b = np.linspace(INV_B_MIN, INV_B_MAX, N_INV_B)
    b_values = 1.0 / inv_b
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, T_GRID.size))
    shade_colors = ["#dbe9f6", "#f8e6c1", "#d8efdf"]
    rows = []

    fig, axes = plt.subplots(
        len(RHO_VALUES), 2,
        figsize=(12.2, 2.05 * len(RHO_VALUES)),
        sharex=False,
        squeeze=False,
    )

    for row_idx, rho in enumerate(RHO_VALUES):
        F_exp, target_label, target_density = expected_frequency(rho)
        chi_traces = []
        mag_traces = []

        for T in T_GRID:
            chi_raw, mag_raw, _, _ = response_for_rho(rho, b_values, float(T))
            x_chi, chi_osc = oscillatory_part(inv_b, chi_raw)
            x_mag, mag_osc = oscillatory_part(inv_b, mag_raw)
            chi_traces.append((x_chi, chi_osc))
            mag_traces.append((x_mag, mag_osc))

        max_color = "#d62728"
        min_color = "#1f77b4"
        for col, (obs_name, traces) in enumerate([
            (r"$\widetilde\chi_T$", chi_traces),
            (r"$\widetilde M_{\rm no\,0LL}$", mag_traces),
        ]):
            ax = axes[row_idx, col]
            x0_trace, y0_trace = traces[0]
            windows = zero_aligned_windows(x0_trace, y0_trace, F_exp)
            x_right = windows[-1][1]
            for win_idx, (x0, x1, method) in enumerate(windows):
                ax.axvspan(x0, x1, color=shade_colors[win_idx], alpha=0.32, lw=0)
                ax.axvline(x0, color="0.45", lw=0.45, alpha=0.55)
                ax.text(
                    0.5 * (x0 + x1), 0.94, f"W{win_idx}",
                    transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8.5,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55},
                )
            ax.axvline(windows[-1][1], color="0.45", lw=0.45, alpha=0.55)

            for t_idx, (T, (x_vals, y_vals)) in enumerate(zip(T_GRID, traces)):
                mask = (x_vals >= INV_B_MIN) & (x_vals <= x_right)
                lw = 1.35 if t_idx == 0 else 0.75
                alpha = 0.95 if t_idx == 0 else 0.58
                ax.plot(x_vals[mask], y_vals[mask], color=colors[t_idx], lw=lw, alpha=alpha)

            # Mark local max/min on the lowest-temperature trace.
            for win_idx, (x0, x1, method) in enumerate(windows):
                xmax, ymax, xmin, ymin = local_extrema(x0_trace, y0_trace, x0, x1)
                ax.plot(xmax, ymax, marker="o", ms=4.8, color=max_color,
                        mec="white", mew=0.55, zorder=5)
                ax.plot(xmin, ymin, marker="v", ms=5.2, color=min_color,
                        mec="white", mew=0.55, zorder=5)
                rows.append({
                    "rho": rho,
                    "observable": "chi" if col == 0 else "M_no0LL",
                    "window": f"W{win_idx}",
                    "x_min": x0,
                    "x_max": x1,
                    "window_method": method,
                    "Delta_invB": x1 - x0,
                    "B_high": 1.0 / x0,
                    "B_low": 1.0 / x1,
                    "Bbar": 1.0 / (0.5 * (x0 + x1)),
                    "F_expected": F_exp,
                    "target_label": target_label,
                    "target_density": target_density,
                    "T_for_extrema": float(T_GRID[0]),
                    "x_at_max": xmax,
                    "y_max": ymax,
                    "x_at_min": xmin,
                    "y_min": ymin,
                    "peak_to_peak": ymax - ymin,
                    "half_peak_to_peak": 0.5 * (ymax - ymin),
                })

            local_values = []
            for x_vals, y_vals in traces:
                mask = (x_vals >= INV_B_MIN) & (x_vals <= x_right)
                local_values.append(y_vals[mask])
            set_symmetric_ylim(ax, local_values)
            ax.set_xlim(INV_B_MIN, x_right)
            ax.axhline(0.0, color="0.35", lw=0.45, alpha=0.55)
            ax.grid(True, alpha=0.18)
            if row_idx == 0:
                ax.set_title(obs_name)
            if col == 0:
                ax.set_ylabel(rf"$\rho={rho:+.3f}$", rotation=0, ha="right", va="center")
            if row_idx == len(RHO_VALUES) - 1:
                ax.set_xlabel(r"$1/B$")

    handles = [
        plt.Line2D([0], [0], color=colors[0], lw=1.5,
                   label=rf"$T_{{\min}}={T_GRID[0]:.1e}$"),
        plt.Line2D([0], [0], color=colors[-1], lw=1.2,
                   label=rf"$T_{{\max}}={T_GRID[-1]:.1e}$"),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#d62728",
                   markeredgecolor="white", markersize=5, label="local max at Tmin"),
        plt.Line2D([0], [0], marker="v", color="none", markerfacecolor="#1f77b4",
                   markeredgecolor="white", markersize=5, label="local min at Tmin"),
    ]
    axes[0, 1].legend(handles=handles, fontsize=7.5, loc="upper right")
    fig.suptitle(
        "Stage 3: first three zero-aligned high-field windows and local extrema"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))

    fig_path = outdir / "S3_first_three_window_traces_extrema_chi_M_no_zeroLL.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    csv_path = outdir / "S3_first_three_window_extrema_chi_M_no_zeroLL.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(fig_path)
    print(csv_path)


if __name__ == "__main__":
    main()
