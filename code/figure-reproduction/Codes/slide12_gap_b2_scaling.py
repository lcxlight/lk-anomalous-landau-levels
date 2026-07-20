"""Appendix figure: anomalous-LL gaps defined in Eq. B.5.

The plotted curves track fixed gap definitions as B varies:

    G_0^up(B) = E_LLL(B) - E_-^LL(n=1, B),
    G_n^up(B) = E_-^LL(n, B) - E_-^LL(n+1, B).

This avoids sorting the cutoff-truncated spectrum at each B, which can introduce
artificial steps when N_max=floor(Lambda^2/2B) changes.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ideal_flatband_model import landau_level_energies_LL, landau_level_energy_LLL
from step4_analytical_gap_scaling import asym_top_largeB, asym_top_smallB


PUBLICATION_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = PUBLICATION_ROOT / "figures" / "appendix_numerics_fresh"


def main() -> None:
    a, c, E0 = 1.115, 0.215, 0.0849
    Lambda = 2.0
    B_star = c**2 / a**2
    B_values = np.geomspace(1.0e-5, 0.08, 130)
    bulk_indices = [1, 2, 5, 10]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = OUT_DIR / "slide12_gap_b2_scaling.png"

    def e_minus(B: np.ndarray, n: int) -> np.ndarray:
        _, em = landau_level_energies_LL(B, n, a, c, E0)
        return em

    def top_gap(B: np.ndarray) -> np.ndarray:
        return landau_level_energy_LLL(B, a, E0) - e_minus(B, 1)

    def fixed_index_gap(B: np.ndarray, n: int) -> np.ndarray:
        return e_minus(B, n) - e_minus(B, n + 1)

    gap_curves = {"0": top_gap(B_values)}
    for n in bulk_indices:
        gap_curves[str(n)] = fixed_index_gap(B_values, n)

    fig, ax0 = plt.subplots(figsize=(8.2, 6.0))

    labels = [r"$G^\uparrow_{0}$", r"$G^\uparrow_{1}$", r"$G^\uparrow_{2}$", r"$G^\uparrow_{5}$", r"$G^\uparrow_{10}$"]
    keys = ["0", "1", "2", "5", "10"]
    colors = plt.cm.viridis(np.linspace(0.08, 0.86, len(keys)))
    for color, key, label in zip(colors, keys, labels):
        ax0.loglog(B_values, gap_curves[key], "-", lw=2.4, color=color, label=label)

    pred = asym_top_smallB(B_values, a, c)
    ax0.loglog(B_values, pred, "k--", lw=2.3, label=r"$2a^3B^2/c^2$")
    largeB = asym_top_largeB(B_values, a, c)
    ax0.loglog(B_values, np.where(largeB > 0, largeB, np.nan), "k:", lw=2.1, label=r"$aB-c^2/(3a)$")

    ax0.axvline(B_star, color="0.45", lw=1.5, ls="-.")
    ax0.text(B_star * 1.04, ax0.get_ylim()[0] * 1.7, r"$B^*$", fontsize=16, color="0.35")
    ax0.set_xlabel(r"$B$", fontsize=18)
    ax0.set_ylabel(r"$G^\uparrow_n(B)$", fontsize=18)
    ax0.tick_params(axis="both", which="major", labelsize=15, length=6)
    ax0.tick_params(axis="both", which="minor", length=3)
    ax0.grid(True, which="both", alpha=0.25)
    ax0.legend(fontsize=14, loc="upper left", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(fig_path, dpi=220)
    plt.close(fig)

    print(f"Wrote {fig_path}")


if __name__ == "__main__":
    main()
