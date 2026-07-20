"""Build manuscript main figures from the current outline.

Outputs are written to ``figures/manuscript figures``.  The script keeps the
slide/source figures intact and makes manuscript-specific composites.
"""
from __future__ import annotations

from pathlib import Path
import sys
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from lambda1_deep_response import (  # noqa: E402
    response_for_rho,
    deep_lower_sector_at_B,
)
from step4_lambda1_stage3_thermal_damping_traces import (  # noqa: E402
    T_GRID,
    expected_frequency,
    fft_spectrum,
    oscillatory_part,
)
from step4_lambda1_stage3_first_three_window_traces import (  # noqa: E402
    zero_aligned_windows,
    local_extrema,
)
from ideal_flatband_model import (  # noqa: E402
    generate_path,
    get_eigenvalues,
    landau_level_energies_LL,
    landau_level_energies_LLTR,
    landau_level_energies_LLTRZero,
    landau_level_energy_LLL,
)
from step2_quantum_geometry import analytic_geometry_components  # noqa: E402
from step4_lk_thermal import (  # noqa: E402
    THEORY_PARAMS,
    carrier_sector_at_B,
    solve_mu_for_sector,
    sector_magnetization_finite_T,
)
from step4_lambda1_stage4_theory_effective_mass import (  # noqa: E402
    anomalous_delta_branch_vs_B,
    m_eff_ll_minus,
    m_eff_ll_plus,
    m_eff_lltr_minus,
    m_eff_lltr_plus,
    normal_delta,
)


OUTDIR = ROOT / "figures" / "manuscript figures"
FIG05 = ROOT / "figures" / "default_a1.115_c0.215_E0_0.0849_Lambda_0.5"
FIG10 = ROOT / "figures" / "default_a1.115_c0.215_E0_0.0849_Lambda_1.0"
APPENDIX_FRESH = ROOT / "figures" / "appendix_numerics_fresh"
APP_D4 = APPENDIX_FRESH / "appD4_lambda1_window_p1_damping.csv"
APP_D5 = APPENDIX_FRESH / "appD5_lambda1_effective_mass_summary.csv"

RHO_NORMAL = 0.010
RHO_ANOM = -0.150
RHO_PAIR = [RHO_NORMAL, RHO_ANOM]
RHO_LABEL = {
    RHO_NORMAL: r"normal, $\rho=+0.01$",
    RHO_ANOM: r"anomalous, $\rho=-0.15$",
}
COLORS = {
    RHO_NORMAL: "#2364aa",
    RHO_ANOM: "#b23a48",
}
WINDOW_COLORS = {
    "W0": "#2364aa",
    "W1": "#2a9d8f",
    "W2": "#d99022",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 8.5,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.0,
        "legend.fontsize": 7.2,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "figure.dpi": 120,
        "savefig.dpi": 320,
        "axes.linewidth": 0.7,
        "lines.linewidth": 1.15,
    })


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12, 1.05, label, transform=ax.transAxes, ha="left", va="bottom",
        fontsize=16, fontweight="bold"
    )


def read_img(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    return mpimg.imread(path)


def crop_right_panel(img: np.ndarray, trim: float = 0.02) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(w * 0.5)
    dx = int(w * trim)
    dy = int(h * trim)
    return img[dy:h - dy, x0 + dx:w - dx]


def crop_center(img: np.ndarray, trim_x: float = 0.03, trim_y: float = 0.03) -> np.ndarray:
    h, w = img.shape[:2]
    dx = int(w * trim_x)
    dy = int(h * trim_y)
    return img[dy:h - dy, dx:w - dx]


def save_both(fig: plt.Figure, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTDIR / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUTDIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        resolved = "\\\\?\\" + resolved
    with open(resolved, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value != "" else float("nan")


def standard_lk_rt_norm(t_vals: np.ndarray, m_eff: float, bbar: float,
                        t_ref: float | None = None) -> np.ndarray:
    x = 2.0 * np.pi**2 * t_vals * m_eff / bbar
    rt = np.ones_like(x, dtype=float)
    mask = np.abs(x) > 1e-12
    safe = mask & (x < 700.0)
    rt[safe] = x[safe] / np.sinh(x[safe])
    rt[mask & (x >= 700.0)] = 0.0
    if t_ref is None:
        ref = rt[0]
    else:
        x0 = 2.0 * np.pi**2 * t_ref * m_eff / bbar
        ref = x0 / np.sinh(x0) if abs(x0) > 1e-12 and x0 < 700.0 else 1.0
    return rt / ref if ref != 0.0 else np.full_like(rt, np.nan)


def make_fig1() -> None:
    a = 1.115
    c = 0.215
    e0 = 0.0
    e1_flat = e0 + c**2 / a
    label_size = 15
    tick_size = 13
    panel_size = 18

    fig = plt.figure(figsize=(7.2, 6.1), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.45])
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, :]),
    ]

    # Panel a: ideal flat-band dispersion along M-Gamma-K.
    p_gamma = np.array([0.0, 0.0])
    p_k = np.array([0.0, 0.5])
    p_m = np.array([0.25, np.sqrt(3) / 4.0])
    k_vecs, tick_indices = generate_path([p_m, p_gamma, p_k], n_points=80)
    evals = get_eigenvalues(k_vecs[:, 0], k_vecs[:, 1], e0, e1_flat, a, c)
    axes[0].plot(evals[:, 0], color="#b23a48", lw=2.4, label="flat band")
    axes[0].plot(evals[:, 1], color="#2364aa", lw=2.4, label="dispersive band")
    for x_tick in tick_indices:
        axes[0].axvline(x_tick, color="0.5", lw=0.7, ls="--")
    axes[0].set_xticks(tick_indices)
    axes[0].set_xticklabels(["M", r"$\Gamma$", "K"])
    axes[0].set_ylabel("Energy (eV)", fontsize=label_size)
    axes[0].set_ylim(e0 - 0.035, e0 + 0.165)

    # Panel b: Berry curvature over the Lambda=0.5 cutoff disk.
    lam = 0.5
    kv = np.linspace(-lam, lam, 181)
    kx, ky = np.meshgrid(kv, kv)
    omega = np.full_like(kx, np.nan, dtype=float)
    mask = kx**2 + ky**2 <= lam**2
    for i, j in np.argwhere(mask):
        omega[i, j] = analytic_geometry_components(kx[i, j], ky[i, j], a, c)[3]
    im = axes[1].pcolormesh(kx, ky, omega, shading="auto", cmap="RdBu_r")
    theta = np.linspace(0, 2 * np.pi, 300)
    axes[1].plot(lam * np.cos(theta), lam * np.sin(theta), color="0.1", lw=0.9)
    axes[1].set_aspect("equal")
    axes[1].set_xlabel(r"$k_x$ ($\mathrm{nm}^{-1}$)", fontsize=label_size)
    axes[1].set_ylabel(r"$k_y$ ($\mathrm{nm}^{-1}$)", fontsize=label_size)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
    cbar.set_label(r"$\Omega(k)$ ($\mathrm{nm}^{2}$)", fontsize=label_size)
    cbar.ax.tick_params(labelsize=tick_size)

    # Panel c: Landau-level spectrum.
    b_vals = np.linspace(0.001, 0.10, 260)
    ax = axes[2]
    ax.plot(b_vals, np.full_like(b_vals, landau_level_energies_LLTRZero(a, c, e0)),
            color="#2a9d5b", lw=2.0, ls="--")
    ax.plot(b_vals, landau_level_energy_LLL(b_vals, a, e0),
            color="#2a9d5b", lw=2.0)
    for n in range(1, 26):
        ep_ll, em_ll = landau_level_energies_LL(b_vals, n, a, c, e0)
        ep_tr, em_tr = landau_level_energies_LLTR(b_vals, n, a, c, e0)
        ax.plot(b_vals, ep_ll, color="#cc6677", lw=0.9, alpha=0.65)
        ax.plot(b_vals, em_ll, color="#2364aa", lw=0.9, alpha=0.78)
        ax.plot(b_vals, ep_tr, color="#b23a48", lw=0.9, ls="--", alpha=0.65)
        ax.plot(b_vals, em_tr, color="#264ee4", lw=0.9, ls="--", alpha=0.78)
    ax.set_xlim(0.0, 0.10)
    ax.set_ylim(e0 - 0.030, e0 + 0.155)
    ax.set_xlabel(r"Magnetic field $B$ ($\mathrm{nm}^{-2}$)", fontsize=label_size)
    ax.set_ylabel("Energy (eV)", fontsize=label_size)

    for label, ax in zip(["a", "b", "c"], axes):
        ax.text(
            -0.16, 1.08, label, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=panel_size, fontweight="bold",
        )
        ax.tick_params(axis="both", which="major", labelsize=tick_size, width=1.1, length=5)
        ax.grid(True, alpha=0.20)
    save_both(fig, "main_fig1_ideal_flatband_LL_spectrum")


def magnetization_data(rho: float, t: float = float(T_GRID[0])):
    inv_b = np.linspace(12.5, 125.0, 420)
    b_values = 1.0 / inv_b
    _, mag_raw, _, _ = response_for_rho(rho, b_values, t)
    x_mag, mag_osc = oscillatory_part(inv_b, mag_raw)
    freq, fft_vals = fft_spectrum(x_mag, mag_osc)
    f_exp, _, _ = expected_frequency(rho)
    return {
        "inv_b": inv_b,
        "b": b_values,
        "mag_raw": mag_raw,
        "x": x_mag,
        "mag_osc": mag_osc,
        "freq": freq,
        "fft": fft_vals,
        "f_exp": f_exp,
    }


def full_magnetization_fixed_deep(rho: float, b_values: np.ndarray,
                                  t: float = float(T_GRID[0])) -> np.ndarray:
    """Full fixed-density magnetization with zero modes included.

    This follows the corrected deep-hole target used in the Step-4 no-zero-LL
    workflow, but does not remove zero-mode terms from the magnetization sum.
    """
    values = np.empty_like(b_values, dtype=float)
    for i, b_val in enumerate(b_values):
        info = carrier_sector_at_B(rho, float(b_val), THEORY_PARAMS)
        if str(info["regime"]) == "deep_hole":
            energies, denergies_dB, target = deep_lower_sector_at_B(rho, float(b_val))
        else:
            energies = np.asarray(info["energies"], dtype=float)
            denergies_dB = np.asarray(info["denergies_dB"], dtype=float)
            target = float(info["target"])
        mu = solve_mu_for_sector(target, float(b_val), t, energies)
        values[i] = sector_magnetization_finite_T(
            mu, float(b_val), t, energies, denergies_dB
        )
    return values


def make_fig2() -> None:
    data = {rho: magnetization_data(rho) for rho in RHO_PAIR}
    label_size = 18
    tick_size = 15
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.2), constrained_layout=True)
    temp_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(T_GRID)))

    for col, rho in enumerate(RHO_PAIR):
        d = data[rho]
        color = COLORS[rho]
        order = np.argsort(d["inv_b"])
        mag_full = full_magnetization_fixed_deep(rho, d["b"], float(T_GRID[0]))
        axes[0, col].plot(d["inv_b"][order], mag_full[order], color=color)
        axes[0, col].set_xlabel(r"$1/B$ ($\mathrm{nm}^{2}$)", fontsize=label_size)
        axes[0, col].set_ylabel(r"$M$ (eV)", fontsize=label_size)

        for t_idx, temp in enumerate(T_GRID):
            td = magnetization_data(rho, float(temp))
            lw = 1.25 if t_idx == 0 else 0.75
            alpha = 0.95 if t_idx == 0 else 0.72
            axes[1, col].plot(
                td["x"], td["mag_osc"],
                color=temp_colors[t_idx], lw=lw, alpha=alpha
            )
        axes[1, col].axhline(0, color="0.45", lw=0.6)
        axes[1, col].set_xlabel(r"$1/B$ ($\mathrm{nm}^{2}$)", fontsize=label_size)
        axes[1, col].set_ylabel(r"$\widetilde M_{\rm no\,0LL}$ (eV)", fontsize=label_size)

        mask = d["freq"] <= 0.25
        amp = d["fft"] / np.nanmax(d["fft"][mask])
        axes[2, col].plot(d["freq"][mask], amp[mask], color=color)
        axes[2, col].axvline(d["f_exp"], color="0.15", ls="--", lw=0.9)
        axes[2, col].set_xlabel(r"Frequency $F$ ($\mathrm{nm}^{-2}$)", fontsize=label_size)
        axes[2, col].set_ylabel("FFT amplitude", fontsize=label_size)

    for label, ax in zip(["a", "b", "c", "d", "e", "f"], axes.ravel()):
        panel_label(ax, label)
        ax.tick_params(axis="both", which="major", labelsize=tick_size, width=1.0, length=4)
        ax.grid(True, alpha=0.22)
    save_both(fig, "main_fig2_normal_anomalous_magnetization_oscillations")


def get_low_t_window_traces(rho: float):
    inv_b = np.linspace(10.0, 125.0, 420)
    b_values = 1.0 / inv_b
    _, mag_raw, _, _ = response_for_rho(rho, b_values, float(T_GRID[0]))
    x, mag_osc = oscillatory_part(inv_b, mag_raw)
    f_exp, _, _ = expected_frequency(rho)
    windows = zero_aligned_windows(x, mag_osc, f_exp)
    return x, mag_osc, windows


def get_window_traces_all_t(rho: float):
    inv_b = np.linspace(10.0, 125.0, 420)
    b_values = 1.0 / inv_b
    traces = []
    for temp in T_GRID:
        _, mag_raw, _, _ = response_for_rho(rho, b_values, float(temp))
        x, mag_osc = oscillatory_part(inv_b, mag_raw)
        traces.append((float(temp), x, mag_osc))
    f_exp, _, _ = expected_frequency(rho)
    windows = zero_aligned_windows(traces[0][1], traces[0][2], f_exp)
    return traces, windows


def make_fig3() -> None:
    # Use the current Appendix D production evidence directly so the main
    # figures cannot drift from the paper's canonical damping/mass values.
    rows = read_csv_rows(APP_D4)
    amp = [
        r for r in rows
        if any(abs(as_float(r, "rho") - rho) < 1e-12 for rho in RHO_PAIR)
    ]
    fit_rows = read_csv_rows(APP_D5)
    fit_lookup = {
        (as_float(r, "rho"), r["window"]): r
        for r in fit_rows
        if any(abs(as_float(r, "rho") - rho) < 1e-12 for rho in RHO_PAIR)
    }

    label_size = 18
    tick_size = 15
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.2), constrained_layout=True)
    temp_colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(T_GRID)))
    for col, rho in enumerate(RHO_PAIR):
        traces, windows = get_window_traces_all_t(rho)
        x = traces[0][1]
        mag_osc = traces[0][2]
        ax = axes[0, col]
        x_right = windows[-1][1]
        mask = (x >= windows[0][0] - 1.5) & (x <= x_right + 1.5)
        for t_idx, (_, xt, yt) in enumerate(traces):
            tmask = (xt >= windows[0][0] - 1.5) & (xt <= x_right + 1.5)
            lw = 1.25 if t_idx == 0 else 0.75
            alpha = 0.95 if t_idx == 0 else 0.72
            ax.plot(xt[tmask], yt[tmask], color=temp_colors[t_idx], lw=lw, alpha=alpha)
        ax.axhline(0, color="0.45", lw=0.55)
        for idx, (x0, x1, _) in enumerate(windows):
            win = f"W{idx}"
            ax.axvspan(x0, x1, color=WINDOW_COLORS[win], alpha=0.16, lw=0)
            ax.text(0.5 * (x0 + x1), 0.93, win, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color=WINDOW_COLORS[win],
                    fontsize=14, fontweight="bold")
            xmax, ymax, xmin, ymin = local_extrema(x, mag_osc, x0, x1)
            ax.plot([xmax], [ymax], "o", color=WINDOW_COLORS[win], ms=4)
            ax.plot([xmin], [ymin], "v", color=WINDOW_COLORS[win], ms=4)
        ax.set_xlabel(r"$1/B$ ($\mathrm{nm}^{2}$)", fontsize=label_size)
        ax.set_ylabel(r"$\widetilde M_{\rm no\,0LL}$ (eV)", fontsize=label_size)

        ax2 = axes[1, col]
        windows = sorted({r["window"] for r in amp if abs(as_float(r, "rho") - rho) < 1e-12})
        for win in windows:
            group = [
                r for r in amp
                if r["window"] == win and abs(as_float(r, "rho") - rho) < 1e-12
            ]
            group = sorted(group, key=lambda r: as_float(r, "T"))
            c = WINDOW_COLORS.get(win, "0.3")
            t = np.array([as_float(r, "T") for r in group])
            a = np.array([as_float(r, "p1_amplitude_norm") for r in group])
            ax2.plot(t, a, "o-", color=c, label=win)
            fit = fit_lookup.get((rho, win))
            if fit is not None:
                t_dense = np.geomspace(float(np.min(t)), float(np.max(t)), 160)
                m_fit = as_float(fit, "m_eff_fit")
                bbar = as_float(fit, "Bbar")
                y_fit = standard_lk_rt_norm(t_dense, m_fit, bbar, t_ref=float(np.min(t)))
                ax2.plot(t_dense, y_fit, "--", color=c, alpha=0.75)
        ax2.set_xscale("log")
        ax2.set_xticks(
            [2.0e-4, 1.0e-3, 4.0e-3],
            labels=[
                r"$2{\times}10^{-4}$",
                r"$10^{-3}$",
                r"$4{\times}10^{-3}$",
            ],
        )
        ax2.set_ylim(bottom=-0.03, top=1.08)
        ax2.set_xlabel(r"$k_B T$ (eV)", fontsize=label_size)
        ax2.set_ylabel(r"$A_1(T)/A_1(T_{\min})$", fontsize=label_size)

    for label, ax in zip(["a", "b", "c", "d"], axes.ravel()):
        panel_label(ax, label)
        ax.tick_params(axis="both", which="major", labelsize=tick_size, width=1.0, length=4)
        ax.grid(True, alpha=0.22)
    for ax in axes[1, :]:
        ax.tick_params(axis="x", which="major", labelsize=11)
    save_both(fig, "main_fig3_thermal_damping_local_windows")


def make_fig4() -> None:
    rows = read_csv_rows(APP_D5)
    mass = [
        r for r in rows
        if any(abs(as_float(r, "rho") - rho) < 1e-12 for rho in RHO_PAIR)
    ]

    label_size = 18
    tick_size = 15
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 4.0), constrained_layout=True)
    for rho in RHO_PAIR:
        group = [r for r in mass if abs(as_float(r, "rho") - rho) < 1e-12]
        group = sorted(group, key=lambda r: as_float(r, "Bbar"))
        color = COLORS[rho]
        bbar_vals = np.array([as_float(r, "Bbar") for r in group])
        fit_vals = np.array([as_float(r, "m_eff_fit") for r in group])
        ax.plot(bbar_vals, fit_vals, "o-", color=color, lw=2.0, ms=7,
                label=RHO_LABEL[rho])

        branch = group[0]["branch_theory"]
        b_pad = 0.08 * (float(np.max(bbar_vals)) - float(np.min(bbar_vals)))
        b_min = max(1e-6, float(np.min(bbar_vals)) - b_pad)
        b_max = float(np.max(bbar_vals)) + b_pad
        b_dense = np.linspace(b_min, b_max, 240)
        if branch == "LL+":
            theory_dense = m_eff_ll_plus(normal_delta(rho), b_dense)
        elif branch == "LLTR+":
            theory_dense = m_eff_lltr_plus(normal_delta(rho), b_dense)
        else:
            delta_dense, _, _ = anomalous_delta_branch_vs_B(rho, b_dense)
            if branch == "LL-":
                theory_dense = m_eff_ll_minus(delta_dense, b_dense)
            elif branch == "LLTR-":
                theory_dense = m_eff_lltr_minus(delta_dense, b_dense)
            else:
                raise ValueError(f"Unknown LK branch for Fig. 4: {branch}")

        ax.plot(b_dense, theory_dense, "--", color=color, lw=1.8, alpha=0.55)
        for row in group:
            ax.text(as_float(row, "Bbar"), as_float(row, "m_eff_fit") + 0.35,
                    row["window"], ha="center", va="bottom", fontsize=14)
    ax.set_xlabel(r"Window-averaged field $\bar B$ ($\mathrm{nm}^{-2}$)", fontsize=label_size)
    ax.set_ylabel(r"$m_{\rm app}$ ($\mathrm{eV}^{-1}\mathrm{nm}^{-2}$)", fontsize=label_size)
    panel_label(ax, "a")
    ax.tick_params(axis="both", which="major", labelsize=tick_size, width=1.1, length=5)
    ax.grid(True, alpha=0.22)
    save_both(fig, "main_fig4_effective_mass_normal_anomalous")


def main() -> None:
    setup_style()
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    print(f"Wrote manuscript figures to: {OUTDIR}")


if __name__ == "__main__":
    main()
