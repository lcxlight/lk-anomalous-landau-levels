"""Compare fixed-density chemical potential and nearest-Fermi LL splitting.

This figure combines, for rho=+0.010 and +0.020:
  1. the fixed-density chemical-potential trajectory mu(1/B), and
  2. the LL spacing closest to the Fermi level.

The bottom panel is read from the exported LL-near-Fermi CSV; the top panel is
computed with the same fixed-density solver used in Appendix Fig. D.1.
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
    active_sector_for_density,
)
from make_spin_counting_diagnostics import T_MIN  # noqa: E402
from step4_lk_thermal import solve_mu_for_sector  # noqa: E402


OUTDIR = ROOT / "figures" / "spin_counting_diagnostics"
SPLIT_CSV = OUTDIR / "electron_LL_near_fermi_summary_dense_public.csv"
RHO_VALUES = [0.010, 0.020]
COLORS = {0.010: "#2364aa", 0.020: "#b23a48"}


def mu_trace(rho: float) -> tuple[np.ndarray, np.ndarray]:
    b_values = 1.0 / INV_B
    mu_vals = np.empty_like(INV_B, dtype=float)
    for i, B in enumerate(b_values):
        _, energies, _, target = active_sector_for_density(rho, float(B))
        mu_vals[i] = solve_mu_for_sector(target, float(B), T_MIN, energies)
    return INV_B.copy(), mu_vals


def read_splitting_rows() -> dict[float, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[float, list[tuple[float, float]]] = {rho: [] for rho in RHO_VALUES}
    with SPLIT_CSV.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rho = float(row["rho"])
            for target in RHO_VALUES:
                if abs(rho - target) < 1e-12:
                    grouped[target].append((
                        float(row["invB"]),
                        float(row["fermi_bracketing_spacing"]),
                    ))
    out: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    for rho, pairs in grouped.items():
        pairs = sorted(pairs)
        out[rho] = (
            np.array([p[0] for p in pairs], dtype=float),
            np.array([p[1] for p in pairs], dtype=float),
        )
    return out


def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    split = read_splitting_rows()

    plt.rcParams.update({
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.8,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.35,
    })

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 5.1), sharex=True, constrained_layout=True)
    ax_mu, ax_gap = axes

    for rho in RHO_VALUES:
        x_mu, y_mu = mu_trace(rho)
        ax_mu.plot(
            x_mu, y_mu,
            color=COLORS[rho],
            label=rf"$\rho={rho:+.3f}\,\mathrm{{nm}}^{{-2}}$",
        )
        x_gap, y_gap = split[rho]
        ax_gap.plot(
            x_gap, y_gap,
            marker=None,
            color=COLORS[rho],
            label=rf"$\rho={rho:+.3f}\,\mathrm{{nm}}^{{-2}}$",
        )

    ax_mu.set_ylabel(r"$\mu_S$ (eV)")
    ax_mu.set_title(r"Fixed-density Fermi level")
    ax_mu.grid(True, alpha=0.25)
    ax_mu.legend(loc="best", frameon=True)

    ax_gap.set_xlabel(r"$1/B$ ($\mathrm{nm}^{2}$)")
    ax_gap.set_ylabel(r"$\Delta E_{\mathrm{near}}$ (eV)")
    ax_gap.set_title(r"LL spacing closest to $\mu_S$")
    ax_gap.grid(True, alpha=0.25)

    for ax, label in zip(axes, ("(a)", "(b)")):
        ax.text(
            0.015, 0.93, label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.0,
            fontweight="bold",
        )

    png = OUTDIR / "electron_mu_and_near_fermi_splitting_comparison.png"
    pdf = OUTDIR / "electron_mu_and_near_fermi_splitting_comparison.pdf"
    fig.savefig(png, dpi=320, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
