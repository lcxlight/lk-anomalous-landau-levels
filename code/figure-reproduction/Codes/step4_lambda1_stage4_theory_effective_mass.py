"""Stage 4 theory expectation for apparent LK effective mass.

This is a theory-only diagnostic.  It asks: if the branch-resolved LK
thermal argument X_p is forced into the standard form

    X_std = 2*pi^2*p*T*m_eff/B,

what apparent m_eff(B) should be expected for each branch?
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

from step4_lk_thermal import THEORY_PARAMS
from lambda1_deep_response import (
    fixed_total_capacity,
)


RHO_NORMAL = [0.010, 0.015]
RHO_ANOMALOUS = [-0.010, -0.020, -0.140, -0.150]
INV_B_GRID = np.linspace(10.0, 125.0, 360)
B_GRID = 1.0 / INV_B_GRID


def normal_delta(rho: float) -> float:
    p = THEORY_PARAMS
    return p.c ** 2 / p.a + 4.0 * np.pi * p.a * rho


def m_eff_ll_plus(delta: np.ndarray | float, B: np.ndarray) -> np.ndarray:
    p = THEORY_PARAMS
    d2 = np.asarray(delta) ** 2
    return (d2 - p.c ** 2 * B) / (2.0 * p.a * d2)


def m_eff_lltr_plus(delta: np.ndarray | float, B: np.ndarray) -> np.ndarray:
    p = THEORY_PARAMS
    d2 = np.asarray(delta) ** 2
    return (d2 + p.c ** 2 * B) / (2.0 * p.a * d2)


def anomalous_delta_branch_vs_B(rho: float, B_values: np.ndarray
                                ) -> tuple[np.ndarray, str, float]:
    """Smooth fixed-density crossing used for the Stage-3 theory curves."""
    p = THEORY_PARAMS
    if rho <= -0.09:
        active_density = fixed_total_capacity() - abs(rho)
        n_star = 2.0 * np.pi * active_density / B_values
        V = p.c ** 2 / p.a + (2.0 * n_star - 1.0) * p.a * B_values
        delta = 0.5 * (V - np.sqrt(V ** 2 + 4.0 * p.c ** 2 * B_values))
        return delta, "LLTR-", active_density

    active_density = abs(rho)
    n_star = 2.0 * np.pi * active_density / B_values
    U = p.c ** 2 / p.a + (2.0 * n_star + 1.0) * p.a * B_values
    rad = np.maximum(U ** 2 - 4.0 * p.c ** 2 * B_values, 0.0)
    delta = 0.5 * (U - np.sqrt(rad))
    return delta, "LL-", active_density


def m_eff_ll_minus(delta: np.ndarray, B: np.ndarray) -> np.ndarray:
    p = THEORY_PARAMS
    d2 = delta ** 2
    return (p.c ** 2 * B - d2) / (2.0 * p.a * d2)


def m_eff_lltr_minus(delta: np.ndarray, B: np.ndarray) -> np.ndarray:
    p = THEORY_PARAMS
    d2 = delta ** 2
    return (p.c ** 2 * B + d2) / (2.0 * p.a * d2)


def summarize_curve(rho: float, branch: str, delta: np.ndarray, m: np.ndarray,
                    active_density: float | None = None) -> dict[str, float | str]:
    idx_low = int(np.argmin(B_GRID))
    idx_high = int(np.argmax(B_GRID))
    idx_mid = int(np.argmin(np.abs(INV_B_GRID - 68.75)))
    return {
        "rho": rho,
        "branch": branch,
        "active_density": "" if active_density is None else active_density,
        "B_min": float(B_GRID[idx_low]),
        "B_mid_invB_68p75": float(B_GRID[idx_mid]),
        "B_max": float(B_GRID[idx_high]),
        "delta_at_Bmin": float(np.asarray(delta)[idx_low]),
        "delta_at_Bmid": float(np.asarray(delta)[idx_mid]),
        "delta_at_Bmax": float(np.asarray(delta)[idx_high]),
        "m_eff_at_Bmin": float(m[idx_low]),
        "m_eff_at_Bmid": float(m[idx_mid]),
        "m_eff_at_Bmax": float(m[idx_high]),
        "m_eff_min": float(np.nanmin(m)),
        "m_eff_max": float(np.nanmax(m)),
    }


def main() -> None:
    p = THEORY_PARAMS
    outdir = (
        ROOT
        / "figures"
        / "default_a1.115_c0.215_E0_0.0849_Lambda_1.0"
        / "step4_lk_theory_tests"
    )
    outdir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    parabolic = 1.0 / (2.0 * p.a)

    fig, (ax_n, ax_a) = plt.subplots(1, 2, figsize=(12.0, 4.6), sharex=True)

    normal_colors = {0.010: "#1f77b4", 0.015: "#2ca02c"}
    anomalous_colors = {
        -0.010: "#d62728",
        -0.020: "#ff7f0e",
        -0.140: "#9467bd",
        -0.150: "#8c564b",
    }

    for rho in RHO_NORMAL:
        delta = normal_delta(rho)
        delta_arr = np.full_like(B_GRID, delta)
        m_ll = m_eff_ll_plus(delta, B_GRID)
        m_lltr = m_eff_lltr_plus(delta, B_GRID)
        color = normal_colors[rho]
        ax_n.plot(B_GRID, m_ll, color=color, lw=1.8,
                  label=rf"$\rho={rho:+.3f}$, LL+")
        ax_n.plot(B_GRID, m_lltr, color=color, lw=1.8, ls="--",
                  label=rf"$\rho={rho:+.3f}$, LLTR+")
        rows.append(summarize_curve(rho, "LL+", delta_arr, m_ll))
        rows.append(summarize_curve(rho, "LLTR+", delta_arr, m_lltr))

    ax_n.axhline(parabolic, color="0.2", lw=1.0, ls=":", label=r"$1/(2a)$")
    ax_n.grid(True, alpha=0.25)
    ax_n.set_ylabel(r"apparent $m_{\rm eff}(B)$")
    ax_n.set_title("normal/dispersive upper branches")
    ax_n.legend(fontsize=7.5, ncol=2)

    for rho in RHO_ANOMALOUS:
        delta, branch, active = anomalous_delta_branch_vs_B(rho, B_GRID)
        if branch == "LL-":
            m = m_eff_ll_minus(delta, B_GRID)
        else:
            m = m_eff_lltr_minus(delta, B_GRID)
        color = anomalous_colors[rho]
        label = rf"$\rho={rho:+.3f}$, {branch}"
        ax_a.plot(B_GRID, m, color=color, lw=1.8, label=label)
        rows.append(summarize_curve(rho, branch, delta, m, active))

    for ax in (ax_n, ax_a):
        ax.set_xlabel(r"$B$")
    ax_a.axhline(parabolic, color="0.2", lw=1.0, ls=":")
    ax_a.grid(True, alpha=0.25)
    ax_a.set_ylabel(r"apparent $m_{\rm eff}(B)$")
    ax_a.set_title("anomalous lower branches")
    ax_a.legend(fontsize=7.5, ncol=2)

    fig.suptitle(
        "Stage 4 theory expectation: apparent effective mass from branch-resolved $X_p$"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig_path = outdir / "S4_theory_apparent_effective_mass_branches.png"
    fig.savefig(fig_path, dpi=230)
    plt.close(fig)

    csv_path = outdir / "S4_theory_apparent_effective_mass_branches.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {fig_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
