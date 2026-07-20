"""Generate Appendix Fig. D.4: windowed p=1 thermal damping.

The window ranges are read from Appendix Fig. D.3's CSV file, so this figure
uses exactly the same W0-W2 intervals shown in the manuscript.  For each
density and window, the zero-mode-excluded oscillatory magnetization is
projected onto the p=1 harmonic at the LL-counting frequency F.
"""
from __future__ import annotations

from pathlib import Path
import csv
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    from scipy.optimize import minimize_scalar
except Exception as exc:  # pragma: no cover
    raise RuntimeError("This script requires scipy.") from exc

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from make_appendix_figD1_lambda1_deltaOmega_M import (  # noqa: E402
    INV_B,
    OUTDIR,
    RHO_VALUES,
    T_VALUES,
    oscillatory_part,
    trace_for_density,
)
from step4_lk_thermal import THEORY_PARAMS  # noqa: E402


WINDOW_CSV = OUTDIR / "appD3_lambda1_zero_aligned_windows.csv"
LOG_T = np.log10(T_VALUES)
WINDOW_COLORS = {
    "W0": "#2364aa",
    "W1": "#2a9d55",
    "W2": "#b23a48",
}
OMIT_UNRELIABLE_SERIES = {
    (-0.020, "W2"),
    (-0.140, "W2"),
}


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


def label_temperature_axis(ax) -> None:
    tick_t = np.array([2.0e-4, 5.0e-4, 1.0e-3, 2.0e-3, 4.0e-3])
    ax.set_xticks(np.log10(tick_t))
    ax.set_xticklabels([
        r"$2$",
        r"$5$",
        r"$10$",
        r"$20$",
        r"$40$",
    ])


def rt_norm_from_mass(T: np.ndarray, m_eff: float, Bbar: float) -> np.ndarray:
    X = 2.0 * np.pi**2 * T * m_eff / Bbar
    rt = np.ones_like(X, dtype=float)
    small = np.abs(X) <= 1e-12
    safe = (~small) & (X < 700.0)
    rt[safe] = X[safe] / np.sinh(X[safe])
    rt[(~small) & (X >= 700.0)] = 0.0
    return rt / rt[0] if rt[0] != 0.0 else np.full_like(rt, np.nan)


def fit_lk_mass(T: np.ndarray, amp_norm: np.ndarray, Bbar: float) -> tuple[float, float]:
    mask = np.isfinite(T) & np.isfinite(amp_norm) & (amp_norm >= 0.0)
    T_fit = T[mask]
    y_fit = amp_norm[mask]
    if T_fit.size < 4:
        return float("nan"), float("nan")

    def loss(mass: float) -> float:
        pred = rt_norm_from_mass(T_fit, mass, Bbar)
        return float(np.mean((pred - y_fit) ** 2))

    res = minimize_scalar(loss, bounds=(1.0e-4, 500.0), method="bounded")
    return float(res.x), math.sqrt(float(res.fun))


def p1_harmonic_fit(x: np.ndarray, y: np.ndarray, F: float,
                    x0: float, x1: float) -> tuple[float, float, float, float]:
    mask = (x >= x0) & (x <= x1) & np.isfinite(y)
    xx = x[mask]
    yy = y[mask]
    if xx.size < 6:
        return float("nan"), float("nan"), float("nan"), float("nan")
    design = np.column_stack([
        np.cos(2.0 * np.pi * F * xx),
        np.sin(2.0 * np.pi * F * xx),
        np.ones_like(xx),
    ])
    coeff, *_ = np.linalg.lstsq(design, yy, rcond=None)
    y_model = design @ coeff
    amp = float(np.hypot(coeff[0], coeff[1]))
    rmse = float(np.sqrt(np.mean((yy - y_model) ** 2)))
    return amp, float(coeff[0]), float(coeff[1]), rmse


def load_windows() -> dict[tuple[float, str], dict[str, float | str]]:
    with WINDOW_CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    windows: dict[tuple[float, str], dict[str, float | str]] = {}
    for row in rows:
        rho = float(row["rho"])
        window = row["window"]
        windows[(rho, window)] = {
            "x_min": float(row["x_min"]),
            "x_max": float(row["x_max"]),
            "Bbar": float(row["Bbar"]),
            "F_expected": float(row["F_expected"]),
            "method": row["window_method"],
            "regime": row["regime_mid"],
            "active_density": float(row["active_density"]),
        }
    return windows


def main() -> None:
    setup_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    windows = load_windows()
    b_values = 1.0 / INV_B

    amplitude_rows = []
    fit_rows = []
    traces_by_key: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}

    for rho in RHO_VALUES:
        for temp in T_VALUES:
            data = trace_for_density(rho, b_values, float(temp))
            x_mag, mag_osc, _ = oscillatory_part(INV_B, data["magnetization"])
            traces_by_key[(rho, float(temp))] = (x_mag, mag_osc)

        for window in ("W0", "W1", "W2"):
            meta = windows[(rho, window)]
            x0 = float(meta["x_min"])
            x1 = float(meta["x_max"])
            Bbar = float(meta["Bbar"])
            F = float(meta["F_expected"])
            amps = []
            temp_rows = []
            for temp in T_VALUES:
                x_vals, y_vals = traces_by_key[(rho, float(temp))]
                amp, c_cos, c_sin, rmse = p1_harmonic_fit(x_vals, y_vals, F, x0, x1)
                amps.append(amp)
                temp_rows.append({
                    "rho": rho,
                    "Lambda": THEORY_PARAMS.Lambda,
                    "window": window,
                    "window_method": meta["method"],
                    "regime": meta["regime"],
                    "x_min": x0,
                    "x_max": x1,
                    "Bbar": Bbar,
                    "F_expected": F,
                    "active_density": float(meta["active_density"]),
                    "T": float(temp),
                    "p1_amplitude": amp,
                    "cos_coeff": c_cos,
                    "sin_coeff": c_sin,
                    "fit_rmse": rmse,
                })
            amp0 = amps[0]
            amp_norm = np.array([
                amp / amp0 if amp0 and np.isfinite(amp0) else np.nan
                for amp in amps
            ])
            m_fit, rmse_fit = fit_lk_mass(T_VALUES, amp_norm, Bbar)
            rt_fit = rt_norm_from_mass(T_VALUES, m_fit, Bbar)
            for idx, row in enumerate(temp_rows):
                row["p1_amplitude_norm"] = float(amp_norm[idx])
                row["lk_fit_norm"] = float(rt_fit[idx])
                row["m_eff_fit"] = m_fit
                amplitude_rows.append(row)
            fit_rows.append({
                "rho": rho,
                "Lambda": THEORY_PARAMS.Lambda,
                "window": window,
                "window_method": meta["method"],
                "regime": meta["regime"],
                "x_min": x0,
                "x_max": x1,
                "Bbar": Bbar,
                "F_expected": F,
                "active_density": float(meta["active_density"]),
                "m_eff_fit": m_fit,
                "fit_rmse": rmse_fit,
                "amp0": amp0,
            })

    fig, axes = plt.subplots(
        len(RHO_VALUES), 1,
        figsize=(7.8, 8.9),
        sharex=True,
        squeeze=False,
    )
    for row_idx, rho in enumerate(RHO_VALUES):
        ax = axes[row_idx, 0]
        for window in ("W0", "W1", "W2"):
            if (round(rho, 3), window) in OMIT_UNRELIABLE_SERIES:
                continue
            selected = [
                r for r in amplitude_rows
                if abs(float(r["rho"]) - rho) < 1e-12 and r["window"] == window
            ]
            selected.sort(key=lambda r: float(r["T"]))
            y = np.array([float(r["p1_amplitude_norm"]) for r in selected])
            fit = np.array([float(r["lk_fit_norm"]) for r in selected])
            Bbar = float(selected[0]["Bbar"])
            color = WINDOW_COLORS[window]
            ax.plot(LOG_T, y, "o-", color=color, ms=3.5, lw=1.15,
                    label=rf"{window}, $\bar B={Bbar:.3f}$")
            ax.plot(LOG_T, fit, "--", color=color, lw=1.0, alpha=0.78)

        ax.axhline(1.0, color="0.45", lw=0.55, alpha=0.55)
        ax.grid(True, alpha=0.22)
        ax.set_ylim(-0.06, 1.20)
        ax.set_ylabel(r"$\mathcal{R}^{\mathrm{num}}_{1,w}(T)$")
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
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.6},
        )
        label_temperature_axis(ax)
        if row_idx == 0:
            ax.legend(loc="lower left", fontsize=7.2, ncol=3, frameon=False)

    axes[-1, 0].set_xlabel(r"$k_B T\;(10^{-4}\,\mathrm{eV})$")
    fig.subplots_adjust(
        left=0.12,
        right=0.985,
        bottom=0.075,
        top=0.985,
        hspace=0.34,
    )

    png = OUTDIR / "appD4_lambda1_window_p1_damping.png"
    pdf = OUTDIR / "appD4_lambda1_window_p1_damping.pdf"
    png_rnum = OUTDIR / "appD4_lambda1_window_p1_damping_Rnum.png"
    pdf_rnum = OUTDIR / "appD4_lambda1_window_p1_damping_Rnum.pdf"
    fig.savefig(png, dpi=260)
    fig.savefig(pdf)
    fig.savefig(png_rnum, dpi=260)
    fig.savefig(pdf_rnum)
    plt.close(fig)

    amp_csv = OUTDIR / "appD4_lambda1_window_p1_damping.csv"
    with amp_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(amplitude_rows[0].keys()))
        writer.writeheader()
        writer.writerows(amplitude_rows)

    fit_csv = OUTDIR / "appD4_lambda1_window_p1_fit_summary.csv"
    with fit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fit_rows[0].keys()))
        writer.writeheader()
        writer.writerows(fit_rows)

    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    print(f"Wrote {png_rnum}")
    print(f"Wrote {pdf_rnum}")
    print(f"Wrote {amp_csv}")
    print(f"Wrote {fit_csv}")


if __name__ == "__main__":
    main()
