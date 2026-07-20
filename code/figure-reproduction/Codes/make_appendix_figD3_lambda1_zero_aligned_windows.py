"""Generate Appendix Fig. D.3: zero-aligned W0-W2 windows.

This is a diagnostic figure for choosing local thermal-damping windows.  It
uses the same Lambda=1 fixed-density convention as Appendix Fig. D.1, computes
the zero-mode-excluded oscillatory magnetization, and marks the first three
zero-aligned windows selected from the lowest-temperature trace.
"""
from __future__ import annotations

from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from make_appendix_figD1_lambda1_deltaOmega_M import (  # noqa: E402
    INV_B,
    OUTDIR,
    RHO_VALUES,
    T_VALUES,
    expected_frequency,
    oscillatory_part,
    trace_for_density,
)
from step4_lk_thermal import THEORY_PARAMS  # noqa: E402
from step4_lambda1_stage3_first_three_window_traces import (  # noqa: E402
    local_extrema,
    zero_aligned_windows,
)


MANUAL_WINDOWS: dict[float, list[tuple[float, float]]] = {
    0.020: [
        (15.933120633, 31.861829664),
        (31.861829664, 47.783525188),
        (47.783525188, 63.702544898),
    ],
    -0.010: [
        (24.237832872, 39.150130313),
        (39.150130313, 55.160533851),
        (55.160533851, 71.994823747),
    ],
}

WINDOW_FREQUENCY_OVERRIDES: dict[float, float] = {
    0.020: np.pi * 0.020,
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


class FixedScalarFormatter(ScalarFormatter):
    def __init__(self) -> None:
        super().__init__(useMathText=True)
        self.set_powerlimits((-2, 2))
        self.set_useOffset(False)


def set_symmetric_ylim(ax, values: list[np.ndarray]) -> None:
    finite_parts = [v[np.isfinite(v)] for v in values if np.any(np.isfinite(v))]
    if not finite_parts:
        return
    finite = np.concatenate(finite_parts)
    vmax = float(np.nanmax(np.abs(finite)))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    ax.set_ylim(-1.15 * vmax, 1.15 * vmax)


def windows_for_density(rho: float, x: np.ndarray, y: np.ndarray,
                        F_exp: float) -> list[tuple[float, float, str]]:
    for rho_key, windows in MANUAL_WINDOWS.items():
        if abs(rho - rho_key) < 1e-12:
            return [(x0, x1, "manual") for x0, x1 in windows]
    return zero_aligned_windows(x, y, F_exp)


def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    b_values = 1.0 / INV_B
    temp_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(T_VALUES)))
    shade_colors = ["#dbe9f6", "#f8e6c1", "#d8efdf"]

    fig, axes = plt.subplots(
        len(RHO_VALUES), 1,
        figsize=(7.8, 8.9),
        sharex=False,
        squeeze=False,
    )
    rows = []

    for row_idx, rho in enumerate(RHO_VALUES):
        ax = axes[row_idx, 0]
        traces: list[tuple[np.ndarray, np.ndarray]] = []
        regime_mid = ""
        F_exp = np.nan
        density_label = ""
        active_density = np.nan

        for temp in T_VALUES:
            data = trace_for_density(rho, b_values, float(temp))
            regime_mid = str(data["regime"][len(data["regime"]) // 2])
            F_exp, density_label, active_density = expected_frequency(rho, regime_mid)
            x_mag, mag_osc, _ = oscillatory_part(INV_B, data["magnetization"])
            traces.append((x_mag, mag_osc))

        x0_trace, y0_trace = traces[0]
        windows = windows_for_density(rho, x0_trace, y0_trace, float(F_exp))
        F_window = float(F_exp)
        for rho_key, freq in WINDOW_FREQUENCY_OVERRIDES.items():
            if abs(rho - rho_key) < 1e-12:
                F_window = float(freq)
                break
        x_right = min(float(INV_B.max()), windows[-1][1] + 0.18 / F_window)
        visible_values = []

        for win_idx, (x0, x1, method) in enumerate(windows):
            ax.axvspan(x0, x1, color=shade_colors[win_idx], alpha=0.35, lw=0)
            ax.axvline(x0, color="0.35", lw=0.65, alpha=0.65)
            ax.text(
                0.5 * (x0 + x1), 0.92, f"W{win_idx}",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8.8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65},
            )
            xmax, ymax, xmin, ymin = local_extrema(x0_trace, y0_trace, x0, x1)
            ax.plot(xmax, ymax, marker="o", ms=4.8, color="#d62728",
                    mec="white", mew=0.55, zorder=5)
            ax.plot(xmin, ymin, marker="v", ms=5.1, color="#1f77b4",
                    mec="white", mew=0.55, zorder=5)
            rows.append({
                "rho": rho,
                "Lambda": THEORY_PARAMS.Lambda,
                "regime_mid": regime_mid,
                "window": f"W{win_idx}",
                "x_min": x0,
                "x_max": x1,
                "window_length": x1 - x0,
                "expected_period": 1.0 / F_window,
                "length_over_expected_period": (x1 - x0) * F_window,
                "window_method": method,
                "B_high": 1.0 / x0,
                "B_low": 1.0 / x1,
                "Bbar": 1.0 / (0.5 * (x0 + x1)),
                "F_expected": F_window,
                "frequency_density_label": density_label.replace("$", ""),
                "active_density": active_density,
                "T_for_extrema": float(T_VALUES[0]),
                "x_at_max": xmax,
                "y_max": ymax,
                "x_at_min": xmin,
                "y_min": ymin,
                "peak_to_peak": ymax - ymin,
            })

        ax.axvline(windows[-1][1], color="0.35", lw=0.65, alpha=0.65)

        for t_idx, (temp, (x_vals, y_vals)) in enumerate(zip(T_VALUES, traces)):
            mask = (x_vals >= float(INV_B.min())) & (x_vals <= x_right)
            lw = 1.35 if t_idx == 0 else 0.85
            alpha = 0.98 if t_idx == 0 else 0.62
            ax.plot(x_vals[mask], y_vals[mask], color=temp_colors[t_idx],
                    lw=lw, alpha=alpha)
            visible_values.append(y_vals[mask])

        set_symmetric_ylim(ax, visible_values)
        ax.axhline(0.0, color="0.35", lw=0.55, alpha=0.65)
        ax.grid(True, alpha=0.18)
        ax.set_xlim(float(INV_B.min()), x_right)
        ax.yaxis.set_major_formatter(FixedScalarFormatter())
        ax.set_ylabel(r"$\widetilde M_{\rm no0LL}$")
        ax.text(
            0.015, 0.88, f"({chr(ord('a') + row_idx)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
        )
        ax.text(
            0.985, 0.88, rf"$\rho={rho:+.3f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55},
        )
        if row_idx == len(RHO_VALUES) - 1:
            ax.set_xlabel(r"$1/B$")

    fig.subplots_adjust(
        left=0.11,
        right=0.985,
        bottom=0.075,
        top=0.985,
        hspace=0.34,
    )

    png = OUTDIR / "appD3_lambda1_zero_aligned_windows.png"
    pdf = OUTDIR / "appD3_lambda1_zero_aligned_windows.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)

    csv_path = OUTDIR / "appD3_lambda1_zero_aligned_windows.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
