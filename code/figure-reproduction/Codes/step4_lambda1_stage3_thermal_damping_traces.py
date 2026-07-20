"""Stage 3 thermal-damping diagnostics for Lambda=1.

This script uses the frequency-verified response observables:

  * compressibility chi_T(B, mu_rho(B))
  * zero-mode-excluded magnetization M_no0LL(B)

For each selected density and temperature, it subtracts a smooth background in
1/B, extracts the FFT amplitude near the verified counting frequency, and
plots both the oscillatory traces and the temperature-dependent amplitudes.
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

from step4_lk_thermal import THEORY_PARAMS, smooth_background
from lambda1_deep_response import (
    fixed_total_capacity,
    response_for_rho,
)


RHO_VALUES = [-0.010, -0.020, -0.140, -0.150, 0.010, 0.015]
T_GRID = np.geomspace(2.0e-4, 4.0e-3, 8)
LOG_T_GRID = np.log10(T_GRID)


def label_log_temperature_axis(ax) -> None:
    tick_t = np.array([2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 4.0e-3])
    ax.set_xticks(np.log10(tick_t))
    ax.set_xticklabels([
        r"$2{\times}10^{-4}$",
        r"$5{\times}10^{-4}$",
        r"$10^{-3}$",
        r"$2{\times}10^{-3}$",
        r"$4{\times}10^{-3}$",
    ])


def expected_frequency(rho: float) -> tuple[float, str, float]:
    """Return expected frequency, counting label, and active density."""
    if rho <= -0.09:
        target = fixed_total_capacity() - abs(rho)
        return 2.0 * np.pi * target, "deep: rho_low_e", target
    target = abs(rho)
    label = "electron |rho|" if rho > 0.0 else "shallow |rho|"
    return 2.0 * np.pi * target, label, target


def oscillatory_part(inv_b: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(inv_b, dtype=float)
    yy = np.asarray(y, dtype=float)
    valid = np.isfinite(yy)
    x = x[valid]
    yy = yy[valid]
    order = np.argsort(x)
    x = x[order]
    yy = yy[order]
    x_uniform = np.linspace(float(x.min()), float(x.max()), x.size)
    y_uniform = np.interp(x_uniform, x, yy)
    background = smooth_background(y_uniform, window_frac=0.31, polyorder=3)
    return x_uniform, y_uniform - background


def fft_amplitude_near(x_uniform: np.ndarray, y_osc: np.ndarray,
                       target: float, width: float = 0.08
                       ) -> tuple[float, float]:
    window = np.hanning(y_osc.size)
    pad_n = 4 * y_osc.size
    fft_vals = np.abs(np.fft.rfft(y_osc * window, n=pad_n))
    freqs = np.fft.rfftfreq(pad_n, d=float(x_uniform[1] - x_uniform[0]))
    mask = (freqs >= max(0.0, target - width)) & (freqs <= target + width)
    if not np.any(mask):
        return float("nan"), float("nan")
    idxs = np.where(mask)[0]
    idx = idxs[int(np.argmax(fft_vals[idxs]))]
    return float(freqs[idx]), float(fft_vals[idx])


def fft_spectrum(x_uniform: np.ndarray, y_osc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(y_osc.size)
    pad_n = 4 * y_osc.size
    fft_vals = np.abs(np.fft.rfft(y_osc * window, n=pad_n))
    freqs = np.fft.rfftfreq(pad_n, d=float(x_uniform[1] - x_uniform[0]))
    return freqs, fft_vals


def main() -> None:
    p = THEORY_PARAMS
    outdir = (
        ROOT
        / "figures"
        / "default_a1.115_c0.215_E0_0.0849_Lambda_1.0"
        / "step4_lk_theory_tests"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    inv_b = np.linspace(12.5, 125.0, 420)
    b_values = 1.0 / inv_b
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, T_GRID.size))

    trace_data: dict[float, dict[str, list[np.ndarray]]] = {}
    amp_rows = []

    for rho in RHO_VALUES:
        F_exp, target_label, target_density = expected_frequency(rho)
        trace_data[rho] = {
            "chi": [],
            "mag": [],
            "x": [],
            "chi_freq": [],
            "chi_fft": [],
            "mag_freq": [],
            "mag_fft": [],
        }
        for T in T_GRID:
            chi_raw, mag_raw, regimes, targets = response_for_rho(rho, b_values, float(T))
            x_chi, chi_osc = oscillatory_part(inv_b, chi_raw)
            x_mag, mag_osc = oscillatory_part(inv_b, mag_raw)
            f_chi, a_chi = fft_amplitude_near(x_chi, chi_osc, F_exp)
            f_mag, a_mag = fft_amplitude_near(x_mag, mag_osc, F_exp)
            chi_freq, chi_fft = fft_spectrum(x_chi, chi_osc)
            mag_freq, mag_fft = fft_spectrum(x_mag, mag_osc)
            trace_data[rho]["x"].append(x_chi)
            trace_data[rho]["chi"].append(chi_osc)
            trace_data[rho]["mag"].append(mag_osc)
            trace_data[rho]["chi_freq"].append(chi_freq)
            trace_data[rho]["chi_fft"].append(chi_fft)
            trace_data[rho]["mag_freq"].append(mag_freq)
            trace_data[rho]["mag_fft"].append(mag_fft)
            amp_rows.append({
                "rho": rho,
                "T": float(T),
                "target_label": target_label,
                "target_density": target_density,
                "F_expected": F_exp,
                "chi_F_peak": f_chi,
                "chi_fft_amplitude": a_chi,
                "M_no0LL_F_peak": f_mag,
                "M_no0LL_fft_amplitude": a_mag,
                "regime_mid": str(regimes[len(regimes) // 2]),
            })

    # Normalize amplitudes density-by-density.
    for rho in RHO_VALUES:
        rows = [r for r in amp_rows if r["rho"] == rho]
        chi0 = rows[0]["chi_fft_amplitude"]
        mag0 = rows[0]["M_no0LL_fft_amplitude"]
        for r in rows:
            r["chi_amp_norm"] = r["chi_fft_amplitude"] / chi0 if chi0 else np.nan
            r["M_no0LL_amp_norm"] = r["M_no0LL_fft_amplitude"] / mag0 if mag0 else np.nan

    # Figure 1: oscillatory traces versus 1/B.
    fig, axes = plt.subplots(
        len(RHO_VALUES), 2, figsize=(11.5, 1.85 * len(RHO_VALUES)),
        sharex=True, squeeze=False
    )
    for row, rho in enumerate(RHO_VALUES):
        F_exp, target_label, target_density = expected_frequency(rho)
        for k, T in enumerate(T_GRID):
            x = trace_data[rho]["x"][k]
            axes[row, 0].plot(x, trace_data[rho]["chi"][k], color=colors[k], lw=0.75)
            axes[row, 1].plot(x, trace_data[rho]["mag"][k], color=colors[k], lw=0.75)
        axes[row, 0].set_ylabel(f"{rho:+.3f}", rotation=0, ha="right", va="center")
        for ax in axes[row]:
            ax.axhline(0.0, color="0.45", lw=0.45, alpha=0.5)
            ax.grid(True, alpha=0.20)
    axes[0, 0].set_title(r"$\widetilde\chi_T(1/B)$")
    axes[0, 1].set_title(r"$\widetilde M_{\rm no\,0LL}(1/B)$")
    axes[-1, 0].set_xlabel(r"$1/B$")
    axes[-1, 1].set_xlabel(r"$1/B$")
    handles = [plt.Line2D([0], [0], color=colors[k], lw=1.5,
                          label=f"{T_GRID[k]:.1e}") for k in range(T_GRID.size)]
    axes[0, 1].legend(handles=handles, title="T", fontsize=7, title_fontsize=8,
                      ncol=2, loc="upper right")
    fig.suptitle(
        rf"Stage 3: smooth-detrended response oscillations vs temperature "
        rf"($\Lambda={p.Lambda:g}$)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig_path = outdir / "S3_temperature_traces_chi_M_no_zeroLL.png"
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    # Figure 2: FFT spectra for the same oscillatory traces.
    fig_fft, axes_fft = plt.subplots(
        len(RHO_VALUES), 2, figsize=(11.5, 1.85 * len(RHO_VALUES)),
        sharex=True, squeeze=False
    )
    for row, rho in enumerate(RHO_VALUES):
        F_exp, target_label, target_density = expected_frequency(rho)
        chi_scale = max(
            np.nanmax(s[(f >= 0.0) & (f <= 0.45)])
            for f, s in zip(trace_data[rho]["chi_freq"], trace_data[rho]["chi_fft"])
        )
        mag_scale = max(
            np.nanmax(s[(f >= 0.0) & (f <= 0.45)])
            for f, s in zip(trace_data[rho]["mag_freq"], trace_data[rho]["mag_fft"])
        )
        chi_scale = chi_scale if chi_scale > 0.0 else 1.0
        mag_scale = mag_scale if mag_scale > 0.0 else 1.0
        for k, T in enumerate(T_GRID):
            f_chi = trace_data[rho]["chi_freq"][k]
            s_chi = trace_data[rho]["chi_fft"][k] / chi_scale
            f_mag = trace_data[rho]["mag_freq"][k]
            s_mag = trace_data[rho]["mag_fft"][k] / mag_scale
            axes_fft[row, 0].plot(f_chi, s_chi, color=colors[k], lw=0.75)
            axes_fft[row, 1].plot(f_mag, s_mag, color=colors[k], lw=0.75)
        axes_fft[row, 0].set_ylabel(f"{rho:+.3f}", rotation=0, ha="right", va="center")
        for ax in axes_fft[row]:
            ax.axvline(F_exp, color="#d62728", lw=0.75, ls="--", alpha=0.85)
            ax.set_xlim(0.0, 0.45)
            ax.set_ylim(bottom=0.0)
            ax.grid(True, alpha=0.20)
    axes_fft[0, 0].set_title(r"$|\mathrm{FFT}[\widetilde\chi_T]|$")
    axes_fft[0, 1].set_title(r"$|\mathrm{FFT}[\widetilde M_{\rm no\,0LL}]|$")
    axes_fft[-1, 0].set_xlabel(r"Frequency $F$ in $1/B$ units")
    axes_fft[-1, 1].set_xlabel(r"Frequency $F$ in $1/B$ units")
    handles = [plt.Line2D([0], [0], color=colors[k], lw=1.5,
                          label=f"{T_GRID[k]:.1e}") for k in range(T_GRID.size)]
    axes_fft[0, 1].legend(handles=handles, title="T", fontsize=7, title_fontsize=8,
                          ncol=2, loc="upper right")
    fig_fft.suptitle(
        rf"Stage 3: FFT spectra of smooth-detrended response traces "
        rf"($\Lambda={p.Lambda:g}$; red dashed = expected frequency)"
    )
    fig_fft.tight_layout(rect=(0, 0, 1, 0.985))
    fft_fig_path = outdir / "S3_fft_spectra_chi_M_no_zeroLL.png"
    fig_fft.savefig(fft_fig_path, dpi=220)
    plt.close(fig_fft)

    # Figure 3: FFT amplitudes versus T.
    fig2, axes2 = plt.subplots(
        len(RHO_VALUES), 2, figsize=(10.5, 1.8 * len(RHO_VALUES)),
        sharex=True, squeeze=False
    )
    for row, rho in enumerate(RHO_VALUES):
        rows = [r for r in amp_rows if r["rho"] == rho]
        T = np.array([r["T"] for r in rows])
        log_T = np.log10(T)
        chi = np.array([r["chi_amp_norm"] for r in rows])
        mag = np.array([r["M_no0LL_amp_norm"] for r in rows])
        axes2[row, 0].plot(log_T, chi, "o-", color="#2a6f97", ms=3.5, lw=1.0)
        axes2[row, 1].plot(log_T, mag, "o-", color="#b02a30", ms=3.5, lw=1.0)
        axes2[row, 0].set_ylabel(f"{rho:+.3f}", rotation=0, ha="right", va="center")
        for ax in axes2[row]:
            ax.grid(True, alpha=0.22)
            ax.axhline(1.0, color="0.45", lw=0.45, alpha=0.5)
            label_log_temperature_axis(ax)
    axes2[0, 0].set_title(r"$A_\chi(T)/A_\chi(T_{\min})$")
    axes2[0, 1].set_title(r"$A_M(T)/A_M(T_{\min})$")
    axes2[-1, 0].set_xlabel(r"$\log_{10}T$  (ticks labeled by $T$)")
    axes2[-1, 1].set_xlabel(r"$\log_{10}T$  (ticks labeled by $T$)")
    fig2.suptitle("Stage 3: FFT amplitude near verified frequency")
    fig2.tight_layout(rect=(0, 0, 1, 0.985))
    amp_fig_path = outdir / "S3_fft_amplitude_vs_temperature_chi_M_no_zeroLL.png"
    fig2.savefig(amp_fig_path, dpi=220)
    plt.close(fig2)

    csv_path = outdir / "S3_fft_amplitude_vs_temperature_chi_M_no_zeroLL.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(amp_rows[0].keys()))
        writer.writeheader()
        writer.writerows(amp_rows)

    print(fig_path)
    print(fft_fig_path)
    print(amp_fig_path)
    print(csv_path)


if __name__ == "__main__":
    main()
