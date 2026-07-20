"""
Analytical asymptotes for the anomalous-LL gap scaling, overlaid on the
exact numerics.

Supports the qualitative scaling claims in
   figures/.../step4_anomalous_LL_spacing_test/figure caption.md
   (sections "single-ladder" and "full-manifold")
by deriving:

(I)  Single ladder, low-i gaps  (E_{i+1}-E_i with i << N_max)
        G_i^low(B)  ~  2 c^2 B^2 / [ a Lambda^4 ]            ~ B^2

(II) Single ladder, top gap  (i = N_max, between LLL zero mode and
     the highest lower-branch state E^LL_-(n=1))
        small B (a^2 B << c^2):  G_top(B) ~ 2 a^3 B^2 / c^2  ~ B^2
        large B (a^2 B >> c^2):  G_top(B) ~ aB - c^2/(3a)    ~ B

(III) Full manifold, lowest gap = E^LLTR_-(2) - E^LLTR_-(1).  Exact:
        G_0^full(B) = aB - (W_2 - W_1)/2,   W_n = sqrt(V_n^2 + 4c^2 B),
        V_n = c^2/a + (2n-1) a B.
      Asymptotes:
        small B (a^2 B << c^2):  G_0^full ~ 2 a^3 B^2 / c^2   ~ B^2
        large B (a^2 B >> c^2):  G_0^full ~ 2 c^2 / (3 a)      ~ const.
      Crossover B* ~ c^2/a^2 (same scale as the top gap).

Derivations live in Notes/analytical_LL_gap_scaling.md.

Outputs one figure with three panels (one per claim) at Lambda=2.0:
    figures/.../step4_anomalous_LL_spacing_test/analytical_gap_scaling.png
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
from step4_anomalous_LL_spacing_test import (
    manifold_LL_only,
    manifold_full,
)


# --- exact gap helpers --------------------------------------------------------

def gap_at(manifold_fn, B, a, c, E0, Lambda, i):
    """Exact gap G_i(B) = E_{i+1} - E_i in the chosen manifold; NaN if i too large."""
    E, _ = manifold_fn(B, a, c, E0, Lambda)
    if E.size < i + 2:
        return np.nan
    return E[i + 1] - E[i]


def top_gap(B, a, c, E0, Lambda):
    """Exact top gap G_{N_max} = E_LLL - E^LL_-(1) in the single ladder."""
    _, E_LL_m = landau_level_energies_LL(B, 1, a, c, E0)
    E_LLL = landau_level_energy_LLL(B, a, E0)
    return E_LLL - E_LL_m


# --- analytical asymptotes ----------------------------------------------------

def asym_low_singleladder(B, a, c, Lambda):
    """G_i^low ~ 2 c^2 B^2 / (a Lambda^4)  -- i-independent leading order."""
    return 2.0 * c**2 * B**2 / (a * Lambda**4)


def asym_top_smallB(B, a, c):
    """Top gap small-B asymptote: 2 a^3 B^2 / c^2."""
    return 2.0 * a**3 * B**2 / c**2


def asym_top_largeB(B, a, c):
    """Top gap large-B asymptote: aB - c^2/(3a)."""
    return a * B - c**2 / (3.0 * a)


def asym_low_fullmanifold_smallB(B, a, c):
    """Small-B asymptote of E^LLTR_-(2)-E^LLTR_-(1):  2 a^3 B^2 / c^2."""
    return 2.0 * a**3 * B**2 / c**2


def asym_low_fullmanifold_largeB(B, a, c):
    """Large-B asymptote of E^LLTR_-(2)-E^LLTR_-(1):  2 c^2 / (3 a) (B-indep)."""
    return np.full_like(np.atleast_1d(B), 2.0 * c**2 / (3.0 * a))


def exact_G0_full_closed(B, a, c):
    """Exact closed form G_0^full = aB - (W_2-W_1)/2 (uses LLTR formulas only)."""
    V1 = c**2 / a + 1 * a * B
    V2 = c**2 / a + 3 * a * B
    W1 = np.sqrt(V1**2 + 4 * c**2 * B)
    W2 = np.sqrt(V2**2 + 4 * c**2 * B)
    return a * B - 0.5 * (W2 - W1)


# --- main figure --------------------------------------------------------------

def main():
    a, c, E0 = 1.115, 0.215, 0.0849
    Lambda   = 2.0    # most levels => widest range of i
    B_values = np.geomspace(0.005, 0.10, 80)

    out_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), '..', 'figures',
        'default_a1.115_c0.215_E0_0.0849_Lambda_0.5',
        'step4_anomalous_LL_spacing_test',
    )
    os.makedirs(out_dir, exist_ok=True)

    # Crossover scale for the top gap.
    B_star = c**2 / a**2

    # ------ exact gaps -------------------------------------------------------
    # Single ladder: low-i for i = 0,1,2,5,10
    is_low = [0, 1, 2, 5, 10]
    G_low_single = {i: np.array([gap_at(manifold_LL_only, B, a, c, E0, Lambda, i)
                                 for B in B_values])
                    for i in is_low}
    # Top gap: i = N_max in single ladder = E_LLL - E_-(1)
    G_top_arr = np.array([top_gap(B, a, c, E0, Lambda) for B in B_values])

    # Full manifold: lowest gap
    G0_full_arr = np.array([gap_at(manifold_full, B, a, c, E0, Lambda, 0)
                            for B in B_values])

    # ------ figure -----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # Panel A: low-i single ladder + analytical 2 c^2 B^2 / (a Lambda^4)
    ax = axes[0]
    cmap = plt.cm.viridis(np.linspace(0.05, 0.85, len(is_low)))
    for col, i in zip(cmap, is_low):
        ax.loglog(B_values, G_low_single[i], 'o', ms=4, color=col,
                  label=f'numerical $G_{{{i}}}$')
    ax.loglog(B_values, asym_low_singleladder(B_values, a, c, Lambda),
              'k--', lw=1.5,
              label=r'$2c^{2}B^{2}/(a\Lambda^{4})$')
    ax.set_xlabel('B'); ax.set_ylabel(r'gap')
    ax.set_title(r'(A) Single ladder, low-$i$ gaps  ($\Lambda=2$)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)

    # Panel B: top gap + small-B + large-B asymptotes
    ax = axes[1]
    ax.loglog(B_values, G_top_arr, 'o', ms=4, color='C3',
              label=r'numerical  $G_{N_{\max}}=E_{\rm LLL}-E^{LL}_{-}(1)$')
    ax.loglog(B_values, asym_top_smallB(B_values, a, c),
              'k--', lw=1.5,
              label=r'small-$B$:  $2a^{3}B^{2}/c^{2}$')
    # large-B asymptote can go negative for very small B; clip for the log plot
    largeB = asym_top_largeB(B_values, a, c)
    largeB_pos = np.where(largeB > 0, largeB, np.nan)
    ax.loglog(B_values, largeB_pos, 'k:', lw=1.5,
              label=r'large-$B$:  $aB - c^{2}/(3a)$')
    ax.axvline(B_star, color='gray', lw=0.8, alpha=0.7)
    ax.text(B_star, ax.get_ylim()[1] * 0.5,
            r'  $B^{*}=c^{2}/a^{2}$', color='gray', fontsize=8)
    ax.set_xlabel('B'); ax.set_ylabel('gap')
    ax.set_title(r'(B) Single ladder, top gap  ($i=N_{\max}$)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)

    # Panel C: full manifold lowest gap + both asymptotes
    ax = axes[2]
    ax.loglog(B_values, G0_full_arr, 'o', ms=4, color='C2',
              label=r'numerical  $G_{0}^{\rm full}=E^{LLTR}_{-}(2)-E^{LLTR}_{-}(1)$')
    ax.loglog(B_values, asym_low_fullmanifold_smallB(B_values, a, c),
              'k--', lw=1.5,
              label=r'small-$B$:  $2a^{3}B^{2}/c^{2}$')
    ax.loglog(B_values, asym_low_fullmanifold_largeB(B_values, a, c),
              'k:', lw=1.5,
              label=r'large-$B$:  $2c^{2}/(3a)$ (const.)')
    ax.axvline(B_star, color='gray', lw=0.8, alpha=0.7)
    ax.text(B_star, ax.get_ylim()[1] * 0.5,
            r'  $B^{*}=c^{2}/a^{2}$', color='gray', fontsize=8)
    ax.set_xlabel('B'); ax.set_ylabel('gap')
    ax.set_title(r'(C) Full manifold, lowest gap  ($i=0$)')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(True, which='both', alpha=0.3)

    fig.suptitle(
        r'Analytical asymptotes vs exact numerics, $\Lambda=2$  ($a=1.115$, $c=0.215$, $E_0=0.0849$)',
        fontsize=12)
    fig.tight_layout()
    out = os.path.join(out_dir, 'analytical_gap_scaling.png')
    fig.savefig(out, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out}')

    # ------ short numerical agreement table ---------------------------------
    print('\n=== Numerical vs analytical at Lambda = 2.0 ===')
    print('(A) Low-i single ladder, prediction = 2 c^2 B^2 / (a Lambda^4):')
    print(f'{"B":>8} {"G_0_meas":>12} {"G_0_pred":>12} {"rel_err":>9}')
    for B in [0.005, 0.01, 0.02, 0.05, 0.10]:
        meas = gap_at(manifold_LL_only, B, a, c, E0, Lambda, 0)
        pred = asym_low_singleladder(B, a, c, Lambda)
        print(f'{B:>8.4f} {meas:>12.4e} {pred:>12.4e} {(meas-pred)/pred*100:>+8.2f}%')

    print('\n(B) Top gap, both asymptotes:')
    print(f'{"B":>8} {"G_top_meas":>12} {"smallB_pred":>12} {"largeB_pred":>12}')
    for B in [0.005, 0.01, 0.02, 0.05, 0.10]:
        meas = top_gap(B, a, c, E0, Lambda)
        sml  = asym_top_smallB(B, a, c)
        lrg  = asym_top_largeB(B, a, c)
        print(f'{B:>8.4f} {meas:>12.4e} {sml:>12.4e} {lrg:>12.4e}')
    print(f'\nCrossover scale B* = c^2/a^2 = {B_star:.5f}')

    print('\n(C) Full-manifold lowest gap, both asymptotes:')
    print(f'{"B":>8} {"G_0_meas":>12} {"smallB_pred":>12} {"largeB_pred":>12}')
    for B in [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 1.0]:
        meas_closed = exact_G0_full_closed(B, a, c)
        sml  = float(asym_low_fullmanifold_smallB(B, a, c))
        lrg  = float(asym_low_fullmanifold_largeB(B, a, c))
        print(f'{B:>8.4f} {meas_closed:>12.4e} {sml:>12.4e} {lrg:>12.4e}')


if __name__ == '__main__':
    main()
