"""Diagnostics for spin-sector counting in Appendix Fig. D.

Outputs:
  figures/spin_counting_diagnostics/lambda1_mu_fixed_density.pdf
  figures/spin_counting_diagnostics/lambda1_rho_p0020_branch_magnetization.pdf
  figures/spin_counting_diagnostics/lambda1_rho_p0020_branch_magnetization_fft.pdf

These plots use the same fixed-density solver and background subtraction as
Appendix Fig. D.1, but expose the chemical-potential trajectories and the
branch-resolved electron-side magnetization.
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

from ideal_flatband_model import (  # noqa: E402
    landau_level_energies_LL,
    landau_level_energies_LLTR,
)
from make_appendix_figD1_lambda1_deltaOmega_M import (  # noqa: E402
    INV_B,
    RHO_VALUES,
    T_VALUES,
    active_sector_for_density,
    oscillatory_part,
    trace_for_density,
)
from step4_lk_thermal import (  # noqa: E402
    THEORY_PARAMS,
    landau_level_energy_LLL,
    sector_filling_finite_T,
    sector_magnetization_finite_T,
    solve_mu_for_sector,
)


OUTDIR = ROOT / "figures" / "spin_counting_diagnostics"
T_MIN = float(T_VALUES[0])


def normal_branch_levels_with_dB(B: float, branch: str) -> tuple[np.ndarray, np.ndarray]:
    """Return normal-branch energies and analytical B derivatives.

    branch = "LL+" or "LLTR+".  Zero modes are deliberately not included here,
    because Fig. D.1 uses the zero-mode-excluded magnetization convention.
    """
    p = THEORY_PARAMS
    n_arr = np.arange(1, max(1, int(np.floor(p.Lambda**2 / (2.0 * B)))) + 1, dtype=float)
    c2 = p.c**2

    if branch == "LL+":
        energies, _ = landau_level_energies_LL(B, n_arr, p.a, p.c, p.E0)
        U = c2 / p.a + (2.0 * n_arr + 1.0) * p.a * B
        Up = (2.0 * n_arr + 1.0) * p.a
        Delta = np.sqrt(np.maximum(U**2 - 4.0 * c2 * B, 1e-300))
        denergies = 0.5 * (Up + (U * Up - 2.0 * c2) / Delta)
    elif branch == "LLTR+":
        energies, _ = landau_level_energies_LLTR(B, n_arr, p.a, p.c, p.E0)
        V = c2 / p.a + (2.0 * n_arr - 1.0) * p.a * B
        Vp = (2.0 * n_arr - 1.0) * p.a
        W = np.sqrt(V**2 + 4.0 * c2 * B)
        denergies = 0.5 * (Vp + (V * Vp + 2.0 * c2) / W)
    else:
        raise ValueError(f"Unknown branch: {branch}")

    order = np.argsort(energies)
    return np.asarray(energies[order], dtype=float), np.asarray(denergies[order], dtype=float)


def electron_zero_occupation(B: float, mu: float, T: float) -> float:
    p = THEORY_PARAMS
    D = B / (2.0 * np.pi)
    e_zero = landau_level_energy_LLL(B, p.a, p.E0)
    if T <= 0:
        return D * float(e_zero <= mu)
    x = (e_zero - mu) / T
    if x >= 0:
        occ = np.exp(-x) / (1.0 + np.exp(-x))
    else:
        occ = 1.0 / (1.0 + np.exp(x))
    return D * float(occ)


def fft_from_oscillation(x_uniform: np.ndarray, y_osc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(y_osc.size)
    pad_n = 4 * y_osc.size
    fft_vals = np.abs(np.fft.rfft(y_osc * window, n=pad_n))
    freqs = np.fft.rfftfreq(pad_n, d=float(x_uniform[1] - x_uniform[0]))
    return freqs, fft_vals


def dominant_peak(freqs: np.ndarray, fft_vals: np.ndarray,
                  fmin: float = 0.005, fmax: float = 0.30) -> tuple[float, float]:
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return float("nan"), float("nan")
    idxs = np.where(mask)[0]
    peak_idx = idxs[int(np.argmax(fft_vals[idxs]))]
    return float(freqs[peak_idx]), float(fft_vals[peak_idx])


def branch_magnetization_traces(
    rho: float = 0.020,
    T: float = T_MIN,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray]]:
    b_values = 1.0 / INV_B
    combined = trace_for_density(rho, b_values, T)
    mu_vals = np.asarray(combined["mu"], dtype=float)

    branch_mag: dict[str, np.ndarray] = {}
    branch_occ: dict[str, np.ndarray] = {}
    for branch in ("LL+", "LLTR+"):
        mag = np.empty_like(b_values)
        occ = np.empty_like(b_values)
        for i, (B, mu) in enumerate(zip(b_values, mu_vals)):
            energies, denergies = normal_branch_levels_with_dB(float(B), branch)
            mag[i] = sector_magnetization_finite_T(mu, float(B), T, energies, denergies)
            occ[i] = sector_filling_finite_T(mu, float(B), T, energies)
        branch_mag[branch] = mag
        branch_occ[branch] = occ

    zero_occ = np.array([electron_zero_occupation(float(B), float(mu), T) for B, mu in zip(b_values, mu_vals)])
    return mu_vals, branch_mag, branch_occ, zero_occ, combined


def plot_mu_trajectories() -> None:
    b_values = 1.0 / INV_B
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(T_VALUES)))
    fig, axes = plt.subplots(3, 2, figsize=(8.0, 8.2), sharex=True, constrained_layout=True)
    rows: list[dict[str, float | str]] = []

    for ax, rho in zip(axes.ravel(), RHO_VALUES):
        for color, T in zip(colors, T_VALUES):
            mu_vals = np.empty_like(b_values)
            residuals = np.empty_like(b_values)
            regimes: list[str] = []
            for i, B in enumerate(b_values):
                regime, energies, _, target = active_sector_for_density(float(rho), float(B))
                mu = solve_mu_for_sector(float(target), float(B), float(T), energies)
                fill = sector_filling_finite_T(mu, float(B), float(T), energies)
                mu_vals[i] = mu
                residuals[i] = fill - float(target)
                regimes.append(regime)
            ax.plot(INV_B, mu_vals, color=color, lw=0.95)
            rows.append({
                "rho": rho,
                "T": float(T),
                "regime_mid": regimes[len(regimes) // 2],
                "max_abs_density_residual": float(np.max(np.abs(residuals))),
                "mu_min": float(np.min(mu_vals)),
                "mu_max": float(np.max(mu_vals)),
            })

        ax.set_title(rf"$\rho={rho:+.3f}\,\mathrm{{nm}}^{{-2}}$")
        ax.set_ylabel(r"$\mu_{\mathcal{S}}$ (eV)")
        ax.grid(True, alpha=0.25)

    for ax in axes[-1, :]:
        ax.set_xlabel(r"$1/B$ ($\mathrm{nm}^{2}$)")

    fig.suptitle(r"Fixed-density chemical-potential trajectories, $\Lambda=1.0\,\mathrm{nm}^{-1}$")
    fig.savefig(OUTDIR / "lambda1_mu_fixed_density.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUTDIR / "lambda1_mu_fixed_density.pdf", bbox_inches="tight")
    plt.close(fig)

    with (OUTDIR / "lambda1_mu_fixed_density_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def rho_tag(rho: float) -> str:
    sign = "p" if rho >= 0.0 else "m"
    return f"{sign}{abs(rho):.3f}".replace(".", "")


def plot_electron_branch_magnetization(rho: float) -> None:
    T = T_MIN
    b_values = 1.0 / INV_B
    mu_vals, branch_mag, branch_occ, zero_occ, combined = branch_magnetization_traces(rho, T)

    x_comb, combined_osc, _ = oscillatory_part(INV_B, np.asarray(combined["magnetization"], dtype=float))
    branch_osc = {
        branch: oscillatory_part(INV_B, mag)[1]
        for branch, mag in branch_mag.items()
    }
    branch_sum_osc = branch_osc["LL+"] + branch_osc["LLTR+"]

    fig, axes = plt.subplots(3, 1, figsize=(7.6, 7.8), sharex=True, constrained_layout=True)

    axes[0].plot(INV_B, branch_occ["LL+"], color="#2364aa", label=r"$E^{\uparrow}_{+}$")
    axes[0].plot(INV_B, branch_occ["LLTR+"], color="#b23a48", label=r"$E^{\downarrow}_{+}$")
    axes[0].plot(INV_B, zero_occ, color="0.35", ls=":", label=r"$E^{\uparrow}_{0}$")
    axes[0].plot(INV_B, branch_occ["LL+"] + branch_occ["LLTR+"] + zero_occ,
                 color="0.05", lw=1.1, label="sum")
    axes[0].axhline(rho, color="0.05", ls="--", lw=0.8, label=r"target $\rho$")
    axes[0].set_ylabel(r"density ($\mathrm{nm}^{-2}$)")
    axes[0].legend(ncol=3, fontsize=8)
    axes[0].grid(True, alpha=0.24)

    axes[1].plot(INV_B, branch_mag["LL+"], color="#2364aa", label=r"$M_{E^{\uparrow}_{+}}$")
    axes[1].plot(INV_B, branch_mag["LLTR+"], color="#b23a48", label=r"$M_{E^{\downarrow}_{+}}$")
    axes[1].plot(INV_B, np.asarray(combined["magnetization"], dtype=float),
                 color="0.05", lw=1.1, label=r"$M_{\mathrm{no0LL}}$ used in Fig. D.1(b)")
    axes[1].set_ylabel(r"$M$ (eV)")
    axes[1].legend(ncol=3, fontsize=8)
    axes[1].grid(True, alpha=0.24)

    axes[2].plot(x_comb, combined_osc, color="0.05", lw=1.25,
                 label=r"$\widetilde M_{\mathrm{no0LL}}$ (Fig. D.1(b))")
    axes[2].plot(x_comb, branch_osc["LL+"], color="#2364aa", alpha=0.9,
                 label=r"$\widetilde M_{E^{\uparrow}_{+}}$")
    axes[2].plot(x_comb, branch_osc["LLTR+"], color="#b23a48", alpha=0.9,
                 label=r"$\widetilde M_{E^{\downarrow}_{+}}$")
    axes[2].plot(x_comb, branch_sum_osc, color="0.45", ls="--", lw=1.0,
                 label="branch sum")
    axes[2].axhline(0.0, color="0.65", lw=0.7)
    axes[2].set_xlabel(r"$1/B$ ($\mathrm{nm}^{2}$)")
    axes[2].set_ylabel(r"osc. $M$ (eV)")
    axes[2].legend(ncol=2, fontsize=8)
    axes[2].grid(True, alpha=0.24)

    tag = rho_tag(rho)
    fig.suptitle(
        rf"Electron-side branch decomposition at $\rho={rho:+.3f}$, "
        rf"$k_BT=2.0\times10^{{-4}}$ eV"
    )
    fig.savefig(OUTDIR / f"lambda1_rho_{tag}_branch_magnetization.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUTDIR / f"lambda1_rho_{tag}_branch_magnetization.pdf", bbox_inches="tight")
    plt.close(fig)

    rows = []
    for i, invB in enumerate(INV_B):
        rows.append({
            "invB": float(invB),
            "B": float(b_values[i]),
            "rho": rho,
            "T": T,
            "mu": float(mu_vals[i]),
            "rho_LLplus": float(branch_occ["LL+"][i]),
            "rho_LLTRplus": float(branch_occ["LLTR+"][i]),
            "rho_LLzero": float(zero_occ[i]),
            "M_LLplus": float(branch_mag["LL+"][i]),
            "M_LLTRplus": float(branch_mag["LLTR+"][i]),
            "M_no0LL_combined": float(combined["magnetization"][i]),
        })
    with (OUTDIR / f"lambda1_rho_{tag}_branch_magnetization.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_electron_branch_magnetization_fft(rho: float) -> None:
    T = T_MIN
    _, branch_mag, _, _, combined = branch_magnetization_traces(rho, T)

    x_comb, combined_osc, _ = oscillatory_part(INV_B, np.asarray(combined["magnetization"], dtype=float))
    traces = [
        (r"$\widetilde M_{\mathrm{no0LL}}$", combined_osc, "0.05"),
        (r"$\widetilde M_{E^{\uparrow}_{+}}$", oscillatory_part(INV_B, branch_mag["LL+"])[1], "#2364aa"),
        (r"$\widetilde M_{E^{\downarrow}_{+}}$", oscillatory_part(INV_B, branch_mag["LLTR+"])[1], "#b23a48"),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 3.6), constrained_layout=True)
    rows = []
    for label, y_osc, color in traces:
        freqs, fft_vals = fft_from_oscillation(x_comb, y_osc)
        mask = (freqs >= 0.0) & (freqs <= 0.30)
        scale = float(np.max(fft_vals[mask])) if np.any(mask) else 1.0
        if scale <= 0.0 or not np.isfinite(scale):
            scale = 1.0
        peak_f, peak_amp = dominant_peak(freqs, fft_vals)
        ax.plot(freqs[mask], fft_vals[mask] / scale, color=color, lw=1.35, label=label)
        ax.axvline(peak_f, color=color, lw=0.75, ls=":", alpha=0.85)
        rows.append({
            "rho": rho,
            "T": T,
            "trace": label.replace("$", ""),
            "dominant_peak_frequency": peak_f,
            "dominant_peak_amplitude": peak_amp,
            "normalization_amplitude": scale,
        })

    ax.set_xlim(0.0, 0.30)
    ax.set_ylim(0.0, 1.08)
    ax.set_xlabel(r"Frequency $F$")
    ax.set_ylabel("Normalized FFT")
    tag = rho_tag(rho)
    ax.set_title(rf"Branch-resolved FFT at $\rho={rho:+.3f}$, $k_BT=2.0\times10^{{-4}}$ eV")
    ax.grid(True, alpha=0.24)
    ax.legend(fontsize=8.0)

    fig.savefig(OUTDIR / f"lambda1_rho_{tag}_branch_magnetization_fft.png", dpi=320, bbox_inches="tight")
    fig.savefig(OUTDIR / f"lambda1_rho_{tag}_branch_magnetization_fft.pdf", bbox_inches="tight")
    plt.close(fig)

    with (OUTDIR / f"lambda1_rho_{tag}_branch_magnetization_fft_peaks.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "legend.fontsize": 8.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
    })
    plot_mu_trajectories()
    for rho in (0.020, 0.010):
        plot_electron_branch_magnetization(rho)
        plot_electron_branch_magnetization_fft(rho)
    print(f"Wrote diagnostics to {OUTDIR}")


if __name__ == "__main__":
    main()
