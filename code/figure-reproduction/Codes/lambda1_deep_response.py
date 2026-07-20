"""Corrected shallow/deep-hole response FFTs with fixed deep-hole target.

For deep holes, the active carriers are counted as lower-sector electrons from
the lowest anomalous LLs upward.  The fixed target is

    rho_low_e = Lambda^2/(2*pi) - |rho|,

rather than rho_up(B)+rho_down(B)-|rho|, which contains N_max(B) staircase
effects.  The magnetization shown here also removes the two zero LL
contributions.
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

from step4_lk_thermal import (
    THEORY_PARAMS,
    E_ref_B,
    analytical_LL_spectrum_with_dB,
    carrier_sector_at_B,
    dos_at_mu_finite_T,
    landau_level_energy_LLL,
    landau_level_energies_LLTRZero,
    sector_magnetization_finite_T,
    solve_mu_for_sector,
    smooth_background,
    thermodynamics_sector_vs_B,
)


RHO_VALUES = [-0.020, -0.030, -0.130, -0.140, -0.150]


def fixed_total_capacity() -> float:
    """Combined two-sector anomalous capacity.

    One flat band within the circular cutoff has capacity Lambda^2/(4*pi).
    The deep-hole convention counts both anomalous sectors relative to E_ref,
    so the total capacity used here is twice that value.
    """
    p = THEORY_PARAMS
    return p.Lambda ** 2 / (2.0 * np.pi)


def deep_lower_sector_at_B(rho: float, B: float):
    p = THEORY_PARAMS
    ll, dll_dB = analytical_LL_spectrum_with_dB(B, p)
    mask = ll < p.E0
    energies = ll[mask]
    denergies_dB = dll_dB[mask]
    order = np.argsort(energies)
    target = fixed_total_capacity() - abs(rho)
    return energies[order], denergies_dB[order], target


def zero_mode_mask_in_active_sector(B: float, energies: np.ndarray,
                                    denergies_dB: np.ndarray,
                                    regime: str) -> np.ndarray:
    p = THEORY_PARAMS
    E_ref = E_ref_B(B, p)
    zero_E = np.array([
        landau_level_energy_LLL(B, p.a, p.E0),
        landau_level_energies_LLTRZero(p.a, p.c, p.E0),
    ])
    zero_dE = np.array([p.a, 0.0])

    mask = np.zeros_like(energies, dtype=bool)
    for Ez, dEz in zip(zero_E, zero_dE):
        if regime == "shallow_hole":
            target_E = E_ref - Ez
            target_dE = -dEz
        else:
            target_E = Ez
            target_dE = dEz
        mask |= (
            np.isclose(energies, target_E, rtol=0.0, atol=1e-10)
            & np.isclose(denergies_dB, target_dE, rtol=0.0, atol=1e-10)
        )
    return mask


def response_for_rho(rho: float, B_grid: np.ndarray, T: float):
    p = THEORY_PARAMS
    regimes = []
    chi = np.empty_like(B_grid, dtype=float)
    mag = np.empty_like(B_grid, dtype=float)
    targets = np.empty_like(B_grid, dtype=float)

    for i, B in enumerate(B_grid):
        classifier = carrier_sector_at_B(rho, float(B), p)
        regime = str(classifier["regime"])
        if regime == "deep_hole":
            energies, denergies_dB, target = deep_lower_sector_at_B(rho, float(B))
        else:
            energies = np.asarray(classifier["energies"], dtype=float)
            denergies_dB = np.asarray(classifier["denergies_dB"], dtype=float)
            target = float(classifier["target"])
        mu = solve_mu_for_sector(target, float(B), T, energies)
        chi[i] = dos_at_mu_finite_T(mu, float(B), T, energies)

        zero_mask = zero_mode_mask_in_active_sector(
            float(B), energies, denergies_dB, regime
        )
        keep = ~zero_mask
        mag[i] = sector_magnetization_finite_T(
            mu, float(B), T, energies[keep], denergies_dB[keep]
        )
        targets[i] = target
        regimes.append(regime)

    return chi, mag, np.asarray(regimes, dtype=object), targets


def oscillatory_part(inv_b: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(inv_b, dtype=float)
    yy = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x = x[order]
    yy = yy[order]
    x_uniform = np.linspace(float(x.min()), float(x.max()), x.size)
    y_uniform = np.interp(x_uniform, x, yy)
    background = smooth_background(y_uniform, window_frac=0.31, polyorder=3)
    return x_uniform, y_uniform - background


def fft_from_oscillation(x_uniform: np.ndarray, y_osc: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray]:
    window = np.hanning(y_osc.size)
    pad_n = 4 * y_osc.size
    fft_vals = np.abs(np.fft.rfft(y_osc * window, n=pad_n))
    freqs = np.fft.rfftfreq(pad_n, d=float(x_uniform[1] - x_uniform[0]))
    return freqs, fft_vals


def peak_near(freqs: np.ndarray, fft_vals: np.ndarray, target: float,
              width: float = 0.08) -> float:
    mask = (freqs >= max(0.0, target - width)) & (freqs <= target + width)
    if not np.any(mask):
        return float("nan")
    idxs = np.where(mask)[0]
    idx = idxs[int(np.argmax(fft_vals[idxs]))]
    return float(freqs[idx])


def main() -> None:
    p = THEORY_PARAMS
    t_min = 2.0e-4
    outdir = (
        ROOT
        / "figures"
        / "default_a1.115_c0.215_E0_0.0849_Lambda_1.0"
        / "step4_lk_theory_tests"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    inv_b = np.linspace(12.5, 125.0, 420)
    b_values = 1.0 / inv_b
    fig, axes = plt.subplots(
        len(RHO_VALUES), 4, figsize=(15.5, 2.05 * len(RHO_VALUES)),
        sharex="col", squeeze=False
    )
    rows = []

    for row, rho in enumerate(RHO_VALUES):
        chi_raw, mag_nozero, regimes, targets = response_for_rho(rho, b_values, t_min)
        regime = str(regimes[len(regimes) // 2])
        if regime == "deep_hole":
            f_target = 2.0 * np.pi * (fixed_total_capacity() - abs(rho))
            target_label = "fixed_lower_electron"
        else:
            f_target = 2.0 * np.pi * abs(rho)
            target_label = "hole_density"

        x_chi, chi_osc = oscillatory_part(inv_b, chi_raw)
        x_mag, mag_osc = oscillatory_part(inv_b, mag_nozero)
        f_chi, a_chi = fft_from_oscillation(x_chi, chi_osc)
        f_mag, a_mag = fft_from_oscillation(x_mag, mag_osc)
        chi_peak = peak_near(f_chi, a_chi, f_target)
        mag_peak = peak_near(f_mag, a_mag, f_target)
        rows.append({
            "rho": rho,
            "regime_at_mid_window": regime,
            "target_convention": target_label,
            "active_target_min": float(np.min(targets)),
            "active_target_max": float(np.max(targets)),
            "F_expected": f_target,
            "compressibility_F_peak": chi_peak,
            "magnetization_no_zero_LL_F_peak": mag_peak,
        })

        color = "#b02a30" if regime == "shallow_hole" else "#5b2a86"
        ax_chi, ax_chif, ax_mag, ax_magf = axes[row]
        ax_chi.plot(x_chi, chi_osc, color=color, lw=0.9)
        ax_mag.plot(x_mag, mag_osc, color=color, lw=0.9)
        ax_chif.plot(f_chi, a_chi, color=color, lw=0.9)
        ax_magf.plot(f_mag, a_mag, color=color, lw=0.9)
        for ax in (ax_chif, ax_magf):
            ax.axvline(f_target, color="#ff7f0e", ls="--", lw=0.85)
            ax.set_xlim(0.0, 0.55)
        if np.isfinite(chi_peak):
            ax_chif.axvline(chi_peak, color="#009e73", ls=":", lw=1.0)
        if np.isfinite(mag_peak):
            ax_magf.axvline(mag_peak, color="#009e73", ls=":", lw=1.0)
        ax_chi.axhline(0.0, color="0.45", lw=0.5, alpha=0.5)
        ax_mag.axhline(0.0, color="0.45", lw=0.5, alpha=0.5)
        ax_chi.set_ylabel(
            f"{rho:+.3f}\n{regime.replace('_', ' ')}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
        )
        for ax in axes[row]:
            ax.grid(True, alpha=0.22)

    axes[0, 0].set_title(r"$\widetilde\chi_T=\chi_T-\chi_{\rm sm}$")
    axes[0, 1].set_title(r"FFT of $\widetilde\chi_T$")
    axes[0, 2].set_title(r"$\widetilde M_{\rm no\,0LL}$")
    axes[0, 3].set_title(r"FFT of $\widetilde M_{\rm no\,0LL}$")
    axes[-1, 0].set_xlabel(r"$1/B$")
    axes[-1, 2].set_xlabel(r"$1/B$")
    axes[-1, 1].set_xlabel(r"Frequency $F$")
    axes[-1, 3].set_xlabel(r"Frequency $F$")
    fig.suptitle(
        rf"Lambda = {p.Lambda:g}: corrected deep-hole response FFTs "
        rf"($T={t_min:.0e}$, fixed lower-electron target)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = outdir / "00p_hole_shallow_deep_fixed_deep_chi_magnetization_no_zeroLL_fft_lambda1.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)

    csv_out = outdir / "00p_hole_shallow_deep_fixed_deep_chi_magnetization_no_zeroLL_fft_peaks_lambda1.csv"
    with csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(out)
    print(csv_out)


if __name__ == "__main__":
    main()
