"""Generate Appendix Fig. D.2: thermodynamic FFT frequency check.

The figure uses the same fixed-density convention as Appendix Fig. D.1, then
Fourier transforms the smooth-detrended grand potential and zero-mode-excluded
magnetization at the lowest temperature.
"""
from __future__ import annotations

from pathlib import Path
import csv
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from make_appendix_figD1_lambda1_deltaOmega_M import (  # noqa: E402
    INV_B,
    OUTDIR,
    RHO_VALUES,
    expected_frequency,
    oscillatory_part,
    trace_for_density,
)
from step4_lk_thermal import THEORY_PARAMS  # noqa: E402


T_MIN = 2.0e-4


def fft_from_oscillation(x_uniform: np.ndarray, y_osc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(y_osc.size)
    pad_n = 4 * y_osc.size
    fft_vals = np.abs(np.fft.rfft(y_osc * window, n=pad_n))
    freqs = np.fft.rfftfreq(pad_n, d=float(x_uniform[1] - x_uniform[0]))
    return freqs, fft_vals


def peak_near(freqs: np.ndarray, fft_vals: np.ndarray, target: float, width: float = 0.08) -> float:
    mask = (freqs >= max(0.0, target - width)) & (freqs <= target + width)
    if not np.any(mask):
        return float("nan")
    idxs = np.where(mask)[0]
    return float(freqs[idxs[int(np.argmax(fft_vals[idxs]))]])


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    b_values = 1.0 / INV_B
    fig, axes = plt.subplots(
        len(RHO_VALUES), 2,
        figsize=(7.8, 9.15),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    rows = []

    for idx, rho in enumerate(RHO_VALUES):
        data = trace_for_density(rho, b_values, T_MIN)
        regime = str(data["regime"][len(data["regime"]) // 2])
        F_exp, density_label, active_density = expected_frequency(rho, regime)

        color = "#2364aa" if rho > 0.0 else "#b23a48"
        peak_values = {}
        for col, (key, label) in enumerate((
            ("omega", "deltaOmega"),
            ("magnetization", "M_no0LL"),
        )):
            ax = axes[idx, col]
            x, y_osc, _ = oscillatory_part(INV_B, data[key])
            freqs, fft_vals = fft_from_oscillation(x, y_osc)
            mask = (freqs >= 0.0) & (freqs <= 0.35)
            scale = float(np.nanmax(fft_vals[mask])) if np.any(mask) else 1.0
            if scale <= 0.0 or not np.isfinite(scale):
                scale = 1.0
            ax.plot(freqs[mask], fft_vals[mask] / scale, color=color)
            ax.axvline(F_exp, color="0.15", lw=0.95, ls="--")
            if rho > 0.0 and abs(rho - 0.020) < 1e-12:
                ax.axvline(np.pi * abs(rho), color="#2ca02c", lw=1.05, ls="--")
            ax.grid(True, alpha=0.24)
            ax.set_xlim(0.0, 0.35)
            ax.set_ylim(0.0, 1.08)
            panel_idx = 2 * idx + col
            ax.text(
                0.03, 0.88, f"({chr(ord('a') + panel_idx)})",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.5,
                fontweight="bold",
            )
            ax.text(
                0.96, 0.88, rf"$\rho={rho:+.3f}$",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9.0,
            )
            peak_values[f"{label}_frequency_peak"] = peak_near(freqs, fft_vals, F_exp)

        rows.append({
            "rho": rho,
            "Lambda": THEORY_PARAMS.Lambda,
            "T": T_MIN,
            "regime": regime,
            "frequency_expected": F_exp,
            "frequency_density_label": density_label.replace("$", ""),
            "active_density": active_density,
            **peak_values,
        })

    for ax in axes[-1, :]:
        ax.set_xlabel(r"Frequency $F$")
    axes[0, 0].set_title(r"FFT of $\delta\Omega_\rho$")
    axes[0, 1].set_title(r"FFT of $\widetilde M_{\rm no0LL}$")
    for ax in axes[:, 0]:
        ax.set_ylabel("Normalized FFT")
    fig.subplots_adjust(
        left=0.105,
        right=0.985,
        bottom=0.075,
        top=0.95,
        wspace=0.18,
        hspace=0.38,
    )

    png = OUTDIR / "appD2_lambda1_magnetization_fft.png"
    pdf = OUTDIR / "appD2_lambda1_magnetization_fft.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)

    csv_path = OUTDIR / "appD2_lambda1_magnetization_fft_peaks.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
