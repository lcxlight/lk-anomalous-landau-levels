"""Generate Appendix Fig. D.1: wide-range thermodynamic oscillations.

The figure compares the smooth-background-subtracted grand potential and
zero-mode-excluded magnetization along the same fixed-density trajectory.
It uses the Lambda=1 convention and the main-text representative densities.
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

from step4_lk_thermal import (  # noqa: E402
    THEORY_PARAMS,
    carrier_sector_at_B,
    sector_grand_potential_finite_T,
    sector_magnetization_finite_T,
    solve_mu_for_sector,
    smooth_background,
)
from lambda1_deep_response import (  # noqa: E402
    deep_lower_sector_at_B,
    fixed_total_capacity,
    zero_mode_mask_in_active_sector,
)


RHO_VALUES = [0.020, 0.010, -0.010, -0.020, -0.140, -0.150]
T_VALUES = np.geomspace(2.0e-4, 4.0e-3, 8)
INV_B = np.linspace(10.0, 100.0, 900)
OUTDIR = (
    ROOT
    / "figures"
    / "appendix_numerics_fresh"
)


def oscillatory_part(inv_b: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return uniform 1/B grid, oscillatory component, and smooth background."""
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
    return x_uniform, y_uniform - background, background


def active_sector_for_density(rho: float, B: float):
    """Use the fixed deep-hole convention already adopted for Lambda=1."""
    p = THEORY_PARAMS
    classifier = carrier_sector_at_B(rho, B, p)
    regime = str(classifier["regime"])
    if regime == "deep_hole":
        energies, denergies_dB, target = deep_lower_sector_at_B(rho, B)
    else:
        energies = np.asarray(classifier["energies"], dtype=float)
        denergies_dB = np.asarray(classifier["denergies_dB"], dtype=float)
        target = float(classifier["target"])
    return regime, energies, denergies_dB, target


def expected_frequency(rho: float, regime: str) -> tuple[float, str, float]:
    """Return frequency and active density used by the fixed-density counting."""
    if regime == "deep_hole":
        active_density = fixed_total_capacity() - abs(rho)
        return 2.0 * np.pi * active_density, r"$\rho_{\rm low,e}$", active_density
    active_density = abs(rho)
    label = r"$|\rho|$"
    return 2.0 * np.pi * active_density, label, active_density


def trace_for_density(rho: float, b_values: np.ndarray, T: float):
    omega = np.empty_like(b_values, dtype=float)
    mag = np.empty_like(b_values, dtype=float)
    mu = np.empty_like(b_values, dtype=float)
    target = np.empty_like(b_values, dtype=float)
    regimes: list[str] = []

    for i, B in enumerate(b_values):
        regime, energies, denergies_dB, target_i = active_sector_for_density(rho, float(B))
        mu_i = solve_mu_for_sector(target_i, float(B), T, energies)
        zero_mask = zero_mode_mask_in_active_sector(
            float(B), energies, denergies_dB, regime
        )
        keep = ~zero_mask
        omega[i] = sector_grand_potential_finite_T(
            mu_i, float(B), T, energies
        )
        mag[i] = sector_magnetization_finite_T(
            mu_i, float(B), T, energies[keep], denergies_dB[keep]
        )
        mu[i] = mu_i
        target[i] = target_i
        regimes.append(regime)

    return {
        "omega": omega,
        "magnetization": mag,
        "mu": mu,
        "target": target,
        "regime": np.asarray(regimes, dtype=object),
    }


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 9.0,
        "lines.linewidth": 1.35,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


class FixedScalarFormatter(ScalarFormatter):
    """Keep scientific notation local to each axis without offset clutter."""

    def __init__(self) -> None:
        super().__init__(useMathText=True)
        self.set_powerlimits((-2, 2))
        self.set_useOffset(False)


def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    b_values = 1.0 / INV_B
    normal_color = "#2364aa"
    anomalous_color = "#b23a48"
    temp_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(T_VALUES)))

    n_rows = len(RHO_VALUES)
    n_pairs = 1
    fig, axes = plt.subplots(
        n_rows, 2 * n_pairs,
        figsize=(7.8, 9.15),
        sharex=True,
        constrained_layout=False,
        squeeze=False,
    )
    rows = []

    for idx, rho in enumerate(RHO_VALUES):
        row = idx
        ax_omega = axes[row, 0]
        ax_mag = axes[row, 1]
        for t_idx, temp in enumerate(T_VALUES):
            data = trace_for_density(rho, b_values, float(temp))
            regime_mid = str(data["regime"][len(data["regime"]) // 2])
            F_exp, density_label, active_density = expected_frequency(rho, regime_mid)

            x_omega, omega_osc, _ = oscillatory_part(INV_B, data["omega"])
            x_mag, mag_osc, _ = oscillatory_part(INV_B, data["magnetization"])
            line_color = temp_colors[t_idx]
            ax_omega.plot(x_omega, omega_osc, color=line_color, lw=0.95)
            ax_mag.plot(x_mag, mag_osc, color=line_color, lw=0.95)

            rows.append({
                "rho": rho,
                "Lambda": THEORY_PARAMS.Lambda,
                "T": float(temp),
                "regime_mid": regime_mid,
                "frequency_expected": F_exp,
                "frequency_density_label": density_label.replace("$", ""),
                "active_density": active_density,
                "target_min": float(np.min(data["target"])),
                "target_max": float(np.max(data["target"])),
                "mu_min": float(np.min(data["mu"])),
                "mu_max": float(np.max(data["mu"])),
                "deltaOmega_min": float(np.min(omega_osc)),
                "deltaOmega_max": float(np.max(omega_osc)),
                "Mtilde_min": float(np.min(mag_osc)),
                "Mtilde_max": float(np.max(mag_osc)),
            })

        for local_col, ax in enumerate((ax_omega, ax_mag)):
            ax.axhline(0.0, color="0.45", lw=0.7, alpha=0.7)
            ax.grid(True, alpha=0.22)
            ax.yaxis.set_major_formatter(FixedScalarFormatter())
            panel_idx = 2 * idx + local_col
            ax.text(
                0.99, 1.04, f"({chr(ord('a') + panel_idx)})",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9.0,
                fontweight="bold",
                clip_on=False,
            )
            ax.tick_params(labelbottom=(row == n_rows - 1))

    axes[0, 0].set_title(r"$\delta\Omega_\rho(1/B)$")
    axes[0, 1].set_title(r"$\widetilde M_{\rm no0LL}(1/B)$")
    for col in range(2):
        axes[-1, col].set_xlabel(r"$1/B$")
    for row in range(n_rows):
        axes[row, 0].set_ylabel(r"$\delta\Omega_\rho$")
        axes[row, 1].set_ylabel(r"$\widetilde M_{\rm no0LL}$")
    for ax in axes.flat:
        ax.set_xlim(float(INV_B.min()), float(INV_B.max()))
    fig.subplots_adjust(
        left=0.115,
        right=0.985,
        bottom=0.075,
        top=0.95,
        wspace=0.32,
        hspace=0.38,
    )

    png = OUTDIR / "appD1_lambda1_deltaOmega_M_wide.png"
    pdf = OUTDIR / "appD1_lambda1_deltaOmega_M_wide.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)

    csv_path = OUTDIR / "appD1_lambda1_deltaOmega_M_wide_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
