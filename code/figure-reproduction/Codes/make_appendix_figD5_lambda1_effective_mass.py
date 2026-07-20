"""Generate Appendix Fig. D.5: effective masses from D.4 window fits."""
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

from make_appendix_figD1_lambda1_deltaOmega_M import OUTDIR  # noqa: E402
from step4_lk_thermal import THEORY_PARAMS  # noqa: E402
from step4_lambda1_stage4_theory_effective_mass import (  # noqa: E402
    B_GRID,
    anomalous_delta_branch_vs_B,
    m_eff_ll_minus,
    m_eff_ll_plus,
    m_eff_lltr_minus,
    m_eff_lltr_plus,
    normal_delta,
)


FIT_CSV = OUTDIR / "appD4_lambda1_window_p1_fit_summary.csv"
NORMAL_RHOS = [0.020, 0.010]
ANOMALOUS_RHOS = [-0.010, -0.020, -0.140, -0.150]
COLORS = {
    0.020: "#2364aa",
    0.010: "#5aa9e6",
    -0.010: "#b23a48",
    -0.020: "#e76f51",
    -0.140: "#7b2cbf",
    -0.150: "#6d4c41",
}
WINDOW_MARKERS = {"W0": "o", "W1": "s", "W2": "^"}
OMIT_UNRELIABLE_POINTS = {
    (-0.020, "W2"),
    (-0.140, "W2"),
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 9.0,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "lines.linewidth": 1.45,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_fit_rows() -> list[dict[str, str]]:
    with FIT_CSV.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def theory_mass_for_anomalous(rho: float, B: np.ndarray) -> tuple[np.ndarray, str]:
    delta, branch, _ = anomalous_delta_branch_vs_B(rho, B)
    if branch == "LL-":
        return m_eff_ll_minus(delta, B), branch
    return m_eff_lltr_minus(delta, B), branch


def theory_mass_at_point(rho: float, Bbar: float, regime: str) -> tuple[float, str, float | str]:
    B = np.array([Bbar], dtype=float)
    if rho > 0.0:
        ll = float(m_eff_ll_plus(normal_delta(rho), B)[0])
        lltr = float(m_eff_lltr_plus(normal_delta(rho), B)[0])
        return ll, "LL+", lltr
    mass, branch = theory_mass_for_anomalous(rho, B)
    return float(mass[0]), branch, ""


def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = load_fit_rows()
    parabolic = 1.0 / (2.0 * THEORY_PARAMS.a)

    fig, (ax_n, ax_a) = plt.subplots(1, 2, figsize=(7.8, 3.75), sharex=True)

    summary_rows = []
    for rho in NORMAL_RHOS:
        color = COLORS[rho]
        m_ll = m_eff_ll_plus(normal_delta(rho), B_GRID)
        m_lltr = m_eff_lltr_plus(normal_delta(rho), B_GRID)
        ax_n.plot(B_GRID, m_ll, color=color, lw=1.45,
                  label=rf"$\rho={rho:+.3f}$")
        ax_n.plot(B_GRID, m_lltr, color=color, lw=1.15, ls="--",
                  label="_nolegend_")
        selected = [r for r in rows if abs(float(r["rho"]) - rho) < 1e-12]
        for item in selected:
            window = item["window"]
            Bbar = float(item["Bbar"])
            m_fit = float(item["m_eff_fit"])
            omit_from_plot = (round(rho, 3), window) in OMIT_UNRELIABLE_POINTS
            m_theory, branch, m_lltr_at_point = theory_mass_at_point(
                rho, Bbar, item["regime"]
            )
            if not omit_from_plot:
                ax_n.scatter(
                    Bbar, m_fit,
                    marker=WINDOW_MARKERS[window],
                    s=46,
                    color=color,
                    edgecolor="black",
                    linewidth=0.55,
                    zorder=5,
                )
                ax_n.text(Bbar, m_fit + 0.08, window, ha="center", va="bottom",
                          fontsize=7.0, color=color)
            summary_rows.append({
                **item,
                "branch_theory": branch,
                "m_eff_theory": m_theory,
                "m_eff_theory_LLTRplus_if_normal": m_lltr_at_point,
                "fit_over_theory": m_fit / m_theory if m_theory else np.nan,
                "plotted": not omit_from_plot,
            })

    for rho in ANOMALOUS_RHOS:
        color = COLORS[rho]
        m_curve, branch = theory_mass_for_anomalous(rho, B_GRID)
        ax_a.plot(B_GRID, m_curve, color=color, lw=1.45,
                  label=rf"$\rho={rho:+.3f}$")
        selected = [r for r in rows if abs(float(r["rho"]) - rho) < 1e-12]
        for item in selected:
            window = item["window"]
            Bbar = float(item["Bbar"])
            m_fit = float(item["m_eff_fit"])
            omit_from_plot = (round(rho, 3), window) in OMIT_UNRELIABLE_POINTS
            m_theory, branch_point, _ = theory_mass_at_point(
                rho, Bbar, item["regime"]
            )
            if not omit_from_plot:
                ax_a.scatter(
                    Bbar, m_fit,
                    marker=WINDOW_MARKERS[window],
                    s=46,
                    color=color,
                    edgecolor="black",
                    linewidth=0.55,
                    zorder=5,
                )
                ax_a.text(Bbar, m_fit + 0.85, window, ha="center", va="bottom",
                          fontsize=7.0, color=color)
            summary_rows.append({
                **item,
                "branch_theory": branch_point,
                "m_eff_theory": m_theory,
                "m_eff_theory_LLTRplus_if_normal": "",
                "fit_over_theory": m_fit / m_theory if m_theory else np.nan,
                "plotted": not omit_from_plot,
            })

    ax_n.axhline(parabolic, color="0.25", lw=0.95, ls=":")
    ax_a.axhline(parabolic, color="0.25", lw=0.95, ls=":")

    for ax, label in ((ax_n, "(a)"), (ax_a, "(b)")):
        ax.grid(True, alpha=0.24)
        ax.set_xlabel(r"$\bar B_w$")
        ax.set_ylabel(r"$m^*_{\mathrm{eff}}$")
        ax.text(
            0.025, 0.93, label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.5,
            fontweight="bold",
        )

    ax_n.set_ylim(0.0, 1.5)
    ax_a.set_ylim(0.0, 38.0)
    ax_n.set_xlim(0.012, 0.068)
    ax_a.set_xlim(0.012, 0.068)
    ax_n.legend(fontsize=6.4, ncol=1, frameon=False, loc="upper right")
    ax_a.legend(fontsize=6.4, ncol=1, frameon=False, loc="upper right")
    fig.subplots_adjust(left=0.095, right=0.985, bottom=0.15, top=0.96, wspace=0.30)

    png = OUTDIR / "appD5_lambda1_effective_mass_summary.png"
    pdf = OUTDIR / "appD5_lambda1_effective_mass_summary.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    plt.close(fig)

    csv_path = OUTDIR / "appD5_lambda1_effective_mass_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
