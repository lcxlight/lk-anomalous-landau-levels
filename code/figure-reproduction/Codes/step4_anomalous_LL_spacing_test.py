"""
Numerical test of  <dE> = 4 pi a T(Lambda) B^2 / Lambda^2  for the anomalous
LL manifold of the THF flat band.

Strategy
--------
At each (Lambda, B):
  1. Enumerate the *anomalous* LL manifold (all LLs derived from the flat band
     and bounded by the momentum cutoff Lambda):
         - zero mode E_LLL = E0 + a B
         - LL-sector  lower branch  E_-(n) for n = 1 .. N_max
         - LLTR-sector lower branch E_-(n) for n = 1 .. N_max
     with N_max = floor(Lambda^2 / (2 B)).
  2. Sort the energies and take adjacent differences -> level spacings.
  3. Report  <dE>  = mean spacing  and  total spread = max - min.

Compare with the predicted formulas:
  spread_pred = a * 2*pi*T(Lambda) * B           (Step 2)
  <dE>_pred   = 4*pi * a * T(Lambda) * B^2 / Lambda^2
              = spread_pred / N_max

with the analytic integrated metric
  T(Lambda) = a Lambda^2 / [2 pi (a Lambda^2 + c^2/a)].

Diagnostics
-----------
  Plot 1: <dE> vs B for each Lambda, on log-log; expect slope 2 (B^2).
  Plot 2: <dE>*Lambda^2/B^2  vs B for each Lambda; expect a B-independent
          plateau equal to 4 pi a T(Lambda).
  Plot 3: spread vs B; expect linear with slope a*2*pi*T(Lambda).
  Plot 4: relative error  (<dE>_meas - <dE>_pred)/<dE>_pred  vs B for each Lambda.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ideal_flatband_model import (
    landau_level_energies_LL,
    landau_level_energies_LLTR,
    landau_level_energy_LLL,
)


def integrated_metric(a, c, Lambda):
    """Closed-form T = a Lambda^2 / [2 pi (a Lambda^2 + c^2/a)]."""
    return a * Lambda**2 / (2.0 * np.pi * (a * Lambda**2 + c**2 / a))


def manifold_LL_only(B, a, c, E0, Lambda):
    """Step-2 definition: LL-sector lower branch only, n = 0..N_max.
    n=0 is the zero mode E_LLL = E0 + aB (top of manifold);
    n>=1 uses landau_level_energies_LL lower branch.
    """
    Nmax = int(np.floor(Lambda**2 / (2.0 * B)))
    energies = [landau_level_energy_LLL(B, a, E0)]   # n = 0
    for n in range(1, Nmax + 1):
        _, E_LL_m = landau_level_energies_LL(B, n, a, c, E0)
        energies.append(E_LL_m)
    return np.sort(np.array(energies)), Nmax


def manifold_full(B, a, c, E0, Lambda):
    """Full physical anomalous manifold: LL ladder (n=0..N_max) + LLTR ladder (n=1..N_max)."""
    Nmax = int(np.floor(Lambda**2 / (2.0 * B)))
    energies = [landau_level_energy_LLL(B, a, E0)]
    for n in range(1, Nmax + 1):
        _, E_LL_m   = landau_level_energies_LL  (B, n, a, c, E0)
        _, E_LLTR_m = landau_level_energies_LLTR(B, n, a, c, E0)
        energies.append(E_LL_m)
        energies.append(E_LLTR_m)
    return np.sort(np.array(energies)), Nmax


def measure(manifold_fn, B, a, c, E0, Lambda):
    """Return (mean adjacent spacing, max-min spread, level count)."""
    energies, _ = manifold_fn(B, a, c, E0, Lambda)
    if energies.size < 2:
        return np.nan, np.nan, energies.size
    diffs = np.diff(energies)
    return diffs.mean(), energies[-1] - energies[0], energies.size


def main():
    a, c, E0 = 1.115, 0.215, 0.0849
    Lambdas  = [0.5, 1.0, 1.5, 2.0]
    # Use B values where N_max > 5 so the 'mean' is over enough levels.
    B_min, B_max, NB = 0.005, 0.10, 60
    B_values = np.linspace(B_min, B_max, NB)

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'figures',
        'default_a1.115_c0.215_E0_0.0849_Lambda_0.5',
        'step4_anomalous_LL_spacing_test',
    )
    os.makedirs(out_dir, exist_ok=True)

    results = {}
    for Lam in Lambdas:
        T = integrated_metric(a, c, Lam)
        rec = {
            'T'        : T,
            'spread_pred_slope': a * 2.0 * np.pi * T,
            'meandE_pred_pref' : 4.0 * np.pi * a * T / Lam**2,  # <dE> = pref * B^2
            'B'        : B_values,
            # Step-2 single-ladder definition
            'meandE_LL'   : np.zeros_like(B_values),
            'spread_LL'   : np.zeros_like(B_values),
            'Nlev_LL'     : np.zeros_like(B_values, dtype=int),
            # Full physical manifold (both ladders)
            'meandE_full' : np.zeros_like(B_values),
            'spread_full' : np.zeros_like(B_values),
            'Nlev_full'   : np.zeros_like(B_values, dtype=int),
        }
        for i, B in enumerate(B_values):
            md, sp, n = measure(manifold_LL_only, B, a, c, E0, Lam)
            rec['meandE_LL'][i] = md
            rec['spread_LL'][i] = sp
            rec['Nlev_LL'  ][i] = n
            md, sp, n = measure(manifold_full,    B, a, c, E0, Lam)
            rec['meandE_full'][i] = md
            rec['spread_full'][i] = sp
            rec['Nlev_full'  ][i] = n
        results[Lam] = rec
        print(f"Lambda={Lam:.2f}: T={T:.5f}, "
              f"spread slope predicted = a*2piT = {rec['spread_pred_slope']:.4f}, "
              f"<dE> prefactor predicted = 4piaT/Lambda^2 = "
              f"{rec['meandE_pred_pref']:.4f}")

    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(Lambdas)))

    # ===== Plots for the Step-2 single-ladder definition =====================
    # (The formula <dE> = 4 pi a T B^2 / Lambda^2 was DERIVED from this
    #  definition; this is the apples-to-apples test.)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ax = axes[0, 0]
    for col, Lam in zip(colors, Lambdas):
        rec = results[Lam]
        ax.loglog(rec['B'], rec['meandE_LL'], 'o', ms=4, color=col,
                  label=f'$\\Lambda$={Lam}')
        ax.loglog(rec['B'], rec['meandE_pred_pref'] * rec['B']**2,
                  '-', color=col, alpha=0.6)
    ax.set_xlabel('B'); ax.set_ylabel(r'$\langle\Delta E\rangle$ [LL ladder only]')
    ax.set_title(r'(a) Step-2 manifold: $\langle\Delta E\rangle$ vs $B$, log-log')
    ax.legend(); ax.grid(True, which='both', alpha=0.3)

    ax = axes[0, 1]
    for col, Lam in zip(colors, Lambdas):
        rec = results[Lam]
        ratio = rec['meandE_LL'] * Lam**2 / rec['B']**2
        ax.plot(rec['B'], ratio, 'o-', ms=4, color=col,
                label=f'$\\Lambda$={Lam}')
        ax.axhline(rec['meandE_pred_pref'] * Lam**2,
                   linestyle='--', color=col, alpha=0.5)
    ax.set_xlabel('B')
    ax.set_ylabel(r'$\langle\Delta E\rangle\,\Lambda^{2}/B^{2}$')
    ax.set_title(r'(b) Plateau test: should equal $4\pi a T(\Lambda)$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for col, Lam in zip(colors, Lambdas):
        rec = results[Lam]
        ax.plot(rec['B'], rec['spread_LL'], 'o', ms=4, color=col,
                label=f'$\\Lambda$={Lam} meas')
        ax.plot(rec['B'], rec['spread_pred_slope'] * rec['B'],
                '-', color=col, alpha=0.6, label=None)
    ax.set_xlabel('B'); ax.set_ylabel(r'$\Delta E_{\rm spread}$ [LL ladder]')
    ax.set_title(r'(c) Spread vs $B$ -- expect slope $a\cdot 2\pi T$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for col, Lam in zip(colors, Lambdas):
        rec = results[Lam]
        pred = rec['meandE_pred_pref'] * rec['B']**2
        rel  = (rec['meandE_LL'] - pred) / pred * 100.0
        ax.plot(rec['B'], rel, 'o-', ms=4, color=col,
                label=f'$\\Lambda$={Lam}')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel('B'); ax.set_ylabel(r'$(\langle\Delta E\rangle_{\rm meas}-\langle\Delta E\rangle_{\rm pred})/\langle\Delta E\rangle_{\rm pred}$ [%]')
    ax.set_title('(d) Relative error in mean-spacing prediction')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle('Step-2 single-ladder anomalous manifold (LL sector lower branch only)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'singleladder_test.png'), dpi=200)
    plt.close(fig)

    # ===== Plots for the FULL physical manifold (both ladders) ===============
    # Reference question: does the SAME formula apply to the full manifold?

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    for col, Lam in zip(colors, Lambdas):
        rec = results[Lam]
        ax.loglog(rec['B'], rec['meandE_full'], 'o', ms=4, color=col,
                  label=f'$\\Lambda$={Lam}')
        # Compare to the Step-2 prediction (same formula)
        ax.loglog(rec['B'], rec['meandE_pred_pref'] * rec['B']**2,
                  '-', color=col, alpha=0.5)
    ax.set_xlabel('B'); ax.set_ylabel(r'$\langle\Delta E\rangle$ [full manifold]')
    ax.set_title(r'(a) Full manifold $\langle\Delta E\rangle$ vs $B$ vs Step-2 prediction')
    ax.legend(fontsize=8); ax.grid(True, which='both', alpha=0.3)

    ax = axes[1]
    for col, Lam in zip(colors, Lambdas):
        rec = results[Lam]
        ratio = rec['meandE_full'] * Lam**2 / rec['B']**2
        ax.plot(rec['B'], ratio, 'o-', ms=4, color=col,
                label=f'$\\Lambda$={Lam}')
        ax.axhline(rec['meandE_pred_pref'] * Lam**2,
                   linestyle='--', color=col, alpha=0.4)
    ax.set_xlabel('B')
    ax.set_ylabel(r'$\langle\Delta E\rangle\,\Lambda^{2}/B^{2}$ [full]')
    ax.set_title(r'(b) Plateau test on full manifold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for col, Lam in zip(colors, Lambdas):
        rec = results[Lam]
        ax.plot(rec['B'], rec['spread_full'], 'o-', ms=4, color=col,
                label=f'$\\Lambda$={Lam}')
        ax.plot(rec['B'], rec['spread_pred_slope'] * rec['B'],
                '--', color=col, alpha=0.4)
    ax.set_xlabel('B'); ax.set_ylabel(r'spread [full]')
    ax.set_title(r'(c) Full-manifold spread vs $B$')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    fig.suptitle('Full physical manifold (LL + LLTR ladders, both lower branches)',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'fullmanifold_test.png'), dpi=200)
    plt.close(fig)

    # ===== Numerical summary tables ==========================================
    print('\n=== Step-2 single-ladder manifold at B = 0.05 ===')
    print(f"{'Lambda':>8} {'Nlev':>6} {'<dE>_meas':>12} {'<dE>_pred':>12}"
          f"{'rel_err':>10} {'spread_meas':>14} {'spread_pred':>12}{'rel_err':>10}")
    for Lam in Lambdas:
        rec = results[Lam]
        idx = int(np.argmin(np.abs(rec['B'] - 0.05)))
        B   = rec['B'][idx]
        md  = rec['meandE_LL'][idx]
        sp  = rec['spread_LL'][idx]
        nl  = rec['Nlev_LL'][idx]
        mdp = rec['meandE_pred_pref'] * B**2
        spp = rec['spread_pred_slope'] * B
        print(f"{Lam:>8.2f} {nl:>6d} {md:>12.5e} {mdp:>12.5e}"
              f" {(md-mdp)/mdp*100:>9.2f}% {sp:>14.5e} {spp:>12.5e}"
              f" {(sp-spp)/spp*100:>9.2f}%")

    print('\n=== Plateau <dE>*Lambda^2/B^2 (mean over B>0.02), single ladder ===')
    print(f"{'Lambda':>8} {'plateau_meas':>14} {'plateau_pred=4piaT':>20} {'rel_err':>10}")
    for Lam in Lambdas:
        rec = results[Lam]
        mask = rec['B'] > 0.02
        plateau = (rec['meandE_LL'][mask] * Lam**2 / rec['B'][mask]**2).mean()
        pred = rec['meandE_pred_pref'] * Lam**2
        print(f"{Lam:>8.2f} {plateau:>14.5f} {pred:>20.5f} {(plateau-pred)/pred*100:>9.2f}%")

    print('\n=== Full manifold (LL + LLTR) at B = 0.05  -- for comparison ===')
    print(f"{'Lambda':>8} {'Nlev':>6} {'<dE>_meas':>12} {'<dE>_pred(Step2)':>18}"
          f"{'rel_err':>10} {'spread_meas':>14}")
    for Lam in Lambdas:
        rec = results[Lam]
        idx = int(np.argmin(np.abs(rec['B'] - 0.05)))
        B   = rec['B'][idx]
        md  = rec['meandE_full'][idx]
        sp  = rec['spread_full'][idx]
        nl  = rec['Nlev_full'][idx]
        mdp = rec['meandE_pred_pref'] * B**2
        print(f"{Lam:>8.2f} {nl:>6d} {md:>12.5e} {mdp:>18.5e}"
              f" {(md-mdp)/mdp*100:>9.2f}% {sp:>14.5e}")

    print(f"\nFigures saved to {out_dir}")


if __name__ == '__main__':
    main()
