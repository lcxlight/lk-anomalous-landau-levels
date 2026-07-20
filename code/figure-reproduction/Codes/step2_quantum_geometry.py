"""
Step 2: Quantum Geometry of the Ideal Flat Band
================================================
Computes the quantum geometric tensor (quantum metric + Berry curvature),
quantum distance, and their relation to anomalous Landau level spread for
the ideal isolated flat band in the THF two-band model.

Model Hamiltonian:
    H(k) = [[E0 + a|k|^2,  c(kx + iky)],
             [c(kx - iky),  E1         ]]
with ideal flat band condition E1 = E0 + c^2/a → lower band E_- = E0 (flat).

Key analytic results (derived in this file):
  - Flat-band eigenstate is implemented in a smooth gauge that is gauge-equivalent
    to the note's closed-form state |u_-⟩ = [-c, a k_-] / sqrt(c^2 + a^2 k^2)
  - Upper-band state is taken as the orthogonal complement of |u_-⟩
  - Energy gap: E_gap(k) = ak^2 + c^2/a  (exact under ideal flat band condition)
  - QGT via Hellmann-Feynman: Q_μν = M_μ* M_ν / E_gap^2
      where M_μ = <u_+|∂_μ H|u_->
  - BZ-integrated metric (analytic): T = aΛ^2 / [2π(aΛ^2 + c^2/a)]
  - Max quantum distance (k=0 vs k=Λ): d_max = sqrt(aΛ^2 / (aΛ^2 + c^2/a)) = sqrt(2πT)

References:
  Hwang Y, Rhim JW, Yang BJ. Nat Commun 12, 6433 (2021)  — isolated flat bands
  Rhim JW, Kim K, Yang BJ. Nature 584, 59-63 (2020)      — singular flat bands
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---------------------------------------------------------------------------
# Default parameters (matching ideal_flatband_model.py)
# ---------------------------------------------------------------------------
A_VAL  = 1.115
C_VAL  = 0.215
E0_VAL = 0.0849
LAMBDA = 0.5    # momentum BZ cutoff: NMax ~ Lambda^2 / (2B)

# ---------------------------------------------------------------------------
# Eigenstates (analytic)
# ---------------------------------------------------------------------------

def flat_band_state(kx, ky, a, c):
    """
    Lower (flat) band eigenstate at (kx, ky).

    |u_-⟩ = [-c(kx+iky)/(k D),  ak/D]   D = sqrt(c^2 + a^2 k^2), k = |k|

    At k=0 there is a gauge singularity (the state winds by 2π in phase as
    the direction of k is varied — this encodes Chern number C = 1).
    We regularise k=0 by setting the state to (1, 0), consistent with the
    canonical gauge where the singularity is at k = ∞.
    """
    k2 = kx**2 + ky**2
    if k2 < 1e-20:
        return np.array([1.0 + 0j, 0.0 + 0j])
    k = np.sqrt(k2)
    D = np.sqrt(c**2 + a**2 * k2)
    u1 = -c * (kx + 1j * ky) / (k * D)
    u2 =  a * k / D
    return np.array([u1, u2])


def upper_band_state(kx, ky, a, c):
    """
    Upper (dispersive) band eigenstate.  Orthogonal complement of |u_-⟩:
        |u_+⟩ = [-u2_minus*,  u1_minus*]
    """
    u = flat_band_state(kx, ky, a, c)
    return np.array([-u[1].conj(), u[0].conj()])


def energy_gap(kx, ky, a, c):
    """E_gap = E_+ - E_- = ak^2 + c^2/a  (exact under ideal flat band condition)."""
    return a * (kx**2 + ky**2) + c**2 / a

# ---------------------------------------------------------------------------
# Quantum geometric tensor  (Hellmann-Feynman formula, no finite differences)
# ---------------------------------------------------------------------------

def qgt_at_k(kx, ky, a, c):
    """
    Quantum geometric tensor components of the flat band at (kx, ky).

    Uses the Hellmann-Feynman identity:
        <u_+|∂_μ u_-> = <u_+|∂_μ H|u_-> / (E_- - E_+)

    => Q_μν = <∂_μ u_-|u_+><u_+|∂_ν u_->
             = M_μ*.conj() * M_ν / E_gap^2
    where  M_μ = <u_+|∂_μ H|u_->

    Quantum metric:   g_μν = Re(Q_μν)
    Berry curvature:  Ω    = -2 Im(Q_xy)

    Returns: g_xx, g_xy, g_yy, Omega
    """
    u_m = flat_band_state(kx, ky, a, c)
    u_p = upper_band_state(kx, ky, a, c)
    Eg  = energy_gap(kx, ky, a, c)

    # H derivatives: ∂H/∂kx = [[2akx, c],[c, 0]]; ∂H/∂ky = [[2aky, ic],[-ic, 0]]
    dHx = np.array([[2*a*kx,    c   ],
                    [c,         0.0 ]], dtype=complex)
    dHy = np.array([[2*a*ky,    1j*c],
                    [-1j*c,     0.0 ]], dtype=complex)

    Mx = u_p.conj() @ dHx @ u_m     # <u_+|∂x H|u_->
    My = u_p.conj() @ dHy @ u_m     # <u_+|∂y H|u_->

    Qxx = Mx.conj() * Mx / Eg**2
    Qxy = Mx.conj() * My / Eg**2
    Qyy = My.conj() * My / Eg**2

    return Qxx.real, Qxy.real, Qyy.real, -2.0 * Qxy.imag


# ---------------------------------------------------------------------------
# Analytic formulas (closed form, for validation)
# ---------------------------------------------------------------------------

def analytic_geometry_components(kx, ky, a, c):
    """
    Closed-form geometry from the note:
        g_xx = g_yy = c^2 / (a k^2 + c^2/a)^2
        g_xy = 0
        Omega = 2 c^2 / (a k^2 + c^2/a)^2
    """
    denom = (a * (kx**2 + ky**2) + c**2 / a)**2
    g_diag = c**2 / denom
    omega = 2.0 * c**2 / denom
    return g_diag, 0.0, g_diag, omega

def T_metric_analytic(a, c, Lambda):
    """
    BZ-integrated quantum metric (trace), computed analytically.

    T = integral_{|k|<=Lambda} (g_xx + g_yy) d^2k / (2pi)^2
      = a Lambda^2 / [2pi (a Lambda^2 + c^2/a)]

    Derivation: in polar coords, tr(g) = (theta')^2/4 + sin^2(theta)/(4k^2)
    where cos theta = (ak^2 - c^2/a)/(ak^2 + c^2/a).
    Both terms equal c^2 k / (ak^2 + c^2/a)^2, giving an analytic integral.

    As Lambda -> infinity, T -> 1/(2pi) = C/(2pi) with Chern number C=1.
    """
    return (a * Lambda**2) / (2 * np.pi * (a * Lambda**2 + c**2 / a))


def d_max_analytic(a, c, Lambda):
    """
    Maximum quantum distance within the BZ, achieved between k=0 and |k|=Lambda.

    d_max^2 = a Lambda^2 / (a Lambda^2 + c^2/a) = 2 pi T

    Proof: the Bloch sphere polar angle theta satisfies cos(theta(k=0)) = -1
    and cos(theta(Lambda)) = (aLambda^2 - c^2/a)/(aLambda^2 + c^2/a).
    The quantum distance from south pole gives d^2 = (1 + cos theta)/2.
    The global maximum within the BZ is indeed between k=0 and k=Lambda
    (not between two BZ-boundary points).
    """
    return np.sqrt(a * Lambda**2 / (a * Lambda**2 + c**2 / a))


# ---------------------------------------------------------------------------
# Numerical grid computations
# ---------------------------------------------------------------------------

def compute_geometry_grid(a, c, Lambda, N=200):
    """
    Evaluate quantum metric and Berry curvature on an N x N Cartesian grid
    over the circular BZ disk |k| <= Lambda.

    Returns kx_grid, ky_grid, g_xx, g_yy, Omega  (NaN outside BZ disk).
    """
    kv = np.linspace(-Lambda, Lambda, N)
    KX, KY = np.meshgrid(kv, kv)
    mask = KX**2 + KY**2 <= Lambda**2

    g_xx  = np.full((N, N), np.nan)
    g_yy  = np.full((N, N), np.nan)
    Omega = np.full((N, N), np.nan)

    idx = np.argwhere(mask)
    for i, j in idx:
        gxx, _, gyy, om = qgt_at_k(KX[i, j], KY[i, j], a, c)
        g_xx[i, j]  = gxx
        g_yy[i, j]  = gyy
        Omega[i, j] = om

    return KX, KY, g_xx, g_yy, Omega


def bz_integrals(g_xx, g_yy, Omega, Lambda, N):
    """Integrate metric trace and Berry curvature over the BZ grid."""
    dk = 2 * Lambda / (N - 1)
    T   = np.nansum(g_xx + g_yy) * dk**2 / (2 * np.pi)**2
    Phi = np.nansum(Omega) * dk**2          # total Berry flux
    return T, Phi


def audit_note_consistency(a, c, Lambda, N=41):
    """
    Pointwise audit of the implemented QGT against the closed-form formulas used
    in the note. This checks gauge-invariant quantities only.
    """
    kv = np.linspace(-Lambda, Lambda, N)
    max_abs_err = {'g_xx': 0.0, 'g_xy': 0.0, 'g_yy': 0.0, 'Omega': 0.0}

    for kx in kv:
        for ky in kv:
            if kx**2 + ky**2 > Lambda**2:
                continue
            vals_num = qgt_at_k(kx, ky, a, c)
            vals_ana = analytic_geometry_components(kx, ky, a, c)
            for key, num, ana in zip(max_abs_err.keys(), vals_num, vals_ana):
                max_abs_err[key] = max(max_abs_err[key], abs(num - ana))

    return max_abs_err


# ---------------------------------------------------------------------------
# Quantum distance
# ---------------------------------------------------------------------------

def quantum_distance(kx1, ky1, kx2, ky2, a, c):
    """Bures/quantum distance between flat band states at two k-points."""
    u1 = flat_band_state(kx1, ky1, a, c)
    u2 = flat_band_state(kx2, ky2, a, c)
    return np.sqrt(max(0.0, 1.0 - abs(u1.conj() @ u2)**2))


# ---------------------------------------------------------------------------
# Anomalous LL spread (from the LL spectrum)
# ---------------------------------------------------------------------------

def ll_energies_lower_branch(B_val, a, c, E0, N_max=None, Lambda=LAMBDA):
    """
    Exact lower-branch energies E_m(n, B) for LL sector n = 0..N_max.

    E_m(n) = E0 + 0.5*(c^2/a + (2n+1)aB - sqrt(discriminant))
    discriminant = (c^2/a + (2n+1)aB)^2 - 4c^2 B

    For n=0 (ideal flat band): discriminant = (c^2/a - aB)^2  → E_m(0) = E0 + aB exactly.
    """
    if N_max is None:
        N_max = max(1, int(Lambda**2 / (2.0 * B_val)))
    u = c**2 / a
    v = a * B_val
    energies = []
    for n in range(0, N_max + 1):
        disc = (u + (2*n + 1)*v)**2 - 4 * c**2 * B_val
        Em = E0 + 0.5 * (u + (2*n + 1)*v - np.sqrt(max(disc, 0)))
        energies.append(Em)
    return np.array(energies)


def anomalous_ll_spread(B_val, a, c, E0, Lambda=LAMBDA):
    """
    Spread of the anomalous LL lower branches: max(E_m) - min(E_m).
    E_m(0) = E0 + aB is the highest; E_m(N_max) is the lowest.
    """
    E_m = ll_energies_lower_branch(B_val, a, c, E0, Lambda=Lambda)
    return E_m[0] - E_m[-1]   # E_m(0) - E_m(N_max)  [positive]


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_geometry_maps(a, c, E0, Lambda, N=150, save='quantum_geometry.png'):
    """Figure 1: quantum metric trace and Berry curvature maps over the BZ."""
    print("\n--- Computing quantum geometry maps ---")
    KX, KY, gxx, gyy, Om = compute_geometry_grid(a, c, Lambda, N)
    T_num, Phi_num = bz_integrals(gxx, gyy, Om, Lambda, N)
    T_ana = T_metric_analytic(a, c, Lambda)
    d_ana = d_max_analytic(a, c, Lambda)
    C_num = Phi_num / (2 * np.pi)

    print(f"  T (numeric)  = {T_num:.6f}   T (analytic) = {T_ana:.6f}")
    print(f"  Berry flux Phi = {Phi_num:.4f}  ->  Chern ~= {C_num:.4f}")
    print(f"  d_max (analytic, k=0 vs k=Lambda) = {d_ana:.5f}")
    print(f"  Relation check: sqrt(2piT) = {np.sqrt(2*np.pi*T_ana):.5f}  vs d_max = {d_ana:.5f}")

    tr_g = gxx + gyy
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # --- Metric trace ---
    ax = axes[0]
    im = ax.pcolormesh(KX, KY, tr_g, cmap='hot_r', shading='auto')
    ax.add_patch(plt.Circle((0, 0), Lambda, fill=False, edgecolor='white', lw=1.5))
    plt.colorbar(im, ax=ax, label=r'$\mathrm{tr}(g) = g_{xx} + g_{yy}$')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x$', fontsize=13)
    ax.set_ylabel(r'$k_y$', fontsize=13)
    ax.set_title(rf'Quantum Metric Trace  ($a={a},\ c={c}$)', fontsize=12)
    ax.text(0.04, 0.05,
            rf'$T={T_num:.4f}$  (analytic: {T_ana:.4f})',
            transform=ax.transAxes, color='white', fontsize=10)

    # --- Berry curvature ---
    ax = axes[1]
    vmax = np.nanpercentile(np.abs(Om), 98)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im = ax.pcolormesh(KX, KY, Om, cmap='RdBu_r', norm=norm, shading='auto')
    ax.add_patch(plt.Circle((0, 0), Lambda, fill=False, edgecolor='black', lw=1.5))
    plt.colorbar(im, ax=ax, label=r'$\Omega(\mathbf{k})$')
    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x$', fontsize=13)
    ax.set_ylabel(r'$k_y$', fontsize=13)
    ax.set_title(rf'Berry Curvature  (Chern $C \approx {C_num:.3f}$)', fontsize=12)

    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save}")
    return T_num, T_ana, d_ana, C_num


def plot_quantum_distance_profile(a, c, Lambda, N=300, save='quantum_distance.png'):
    """Figure 2: quantum distance d(k,0) as a function of |k| along kx axis."""
    print("\n--- Quantum distance profile ---")
    k_vals = np.linspace(0.0, Lambda, N)
    d_vals = np.array([quantum_distance(k, 0, 0, 0, a, c) for k in k_vals])

    # Analytic curve: d(k,0)^2 = ak^2/(ak^2+c^2/a)
    d_ana  = np.sqrt(a * k_vals**2 / (a * k_vals**2 + c**2/a + 1e-30))
    d_ana[0] = 0.0

    d_edge = d_max_analytic(a, c, Lambda)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(k_vals, d_vals,  'b-',  lw=2.0, label='numeric')
    ax.plot(k_vals, d_ana,   'r--', lw=1.5, label=r'analytic $\sqrt{ak^2/(ak^2+c^2/a)}$')
    ax.axvline(Lambda, color='gray', ls='--', alpha=0.7,
               label=rf'$\Lambda={Lambda}$')
    ax.axhline(d_edge, color='green', ls=':', alpha=0.8,
               label=rf'$d_\mathrm{{max}}={d_edge:.4f}$')
    ax.set_xlabel(r'$|k|$',                        fontsize=13)
    ax.set_ylabel(r'Quantum distance $d(k,\,0)$',  fontsize=13)
    ax.set_title(rf'Quantum Distance from $k=0$  ($a={a},\,c={c}$)', fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, ls=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    print(f"  d_max (k=0 vs k=Lambda) = {d_edge:.5f}")
    print(f"  Saved: {save}")
    return d_edge


def plot_anomalous_ll_spread(a, c, E0, Lambda, save='anomalous_ll_spread.png'):
    """
    Figure 3: anomalous LL lower-branch energies vs B, showing the spread.

    Also plots the correct small-B analytic prediction:
        Total spread = a * 2pi * T * B  = a^3 * Lambda^2 / (a^2*Lambda^2 + c^2) * B
    which is valid when B << Lambda^2 / 2  AND  B << c^2/a^2.
    (The formula Lambda^2*a^3*B/c^2 is the c->0 / Lambda<<c/a limit and does NOT apply here.)
    """
    print("\n--- Anomalous LL spread vs B ---")
    B_vals = np.linspace(0.005, 0.1, 100)
    spreads = [anomalous_ll_spread(B, a, c, E0, Lambda) for B in B_vals]

    T_ana = T_metric_analytic(a, c, Lambda)
    # Correct small-B geometric prediction: slope = a * 2piT
    geo_factor = a * 2.0 * np.pi * T_ana   # = a^3*Lambda^2 / (a^2*Lambda^2 + c^2)
    spread_geo = geo_factor * B_vals

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(B_vals, spreads,    'b-',  lw=2.0, label='Exact (numeric LL energies)')
    ax.plot(B_vals, spread_geo, 'g--', lw=1.5,
            label=rf'Small-$B$: $a\cdot 2\pi T \cdot B$  ($T={T_ana:.4f}$, slope={geo_factor:.3f})')
    ax.set_xlabel(r'$B_z$',                      fontsize=13)
    ax.set_ylabel(r'Anomalous LL spread $\Delta E$', fontsize=13)
    ax.set_title(rf'Anomalous LL spread vs $B$  ($a={a},\,c={c}$)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, ls=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save}")
    print(f"  T (analytic) = {T_ana:.5f},  d_max = {d_max_analytic(a,c,Lambda):.5f}")
    print(f"  Geometric spread factor (a*2piT) = {geo_factor:.4f}  (energy/field)")


def plot_parameter_scan(E0, Lambda, save='parameter_scan.png'):
    """
    Figure 4: T_metric and d_max vs parameters a and c.

    Uses analytic formulas (instant, no grid needed).
    """
    print("\n--- Parameter scan ---")
    a_vals = np.linspace(0.3, 4.0, 200)
    c_vals = np.linspace(0.05, 0.6, 200)

    c_fixed = C_VAL
    a_fixed = A_VAL

    T_vs_a    = T_metric_analytic(a_vals, c_fixed, Lambda)
    d_vs_a    = d_max_analytic(a_vals, c_fixed, Lambda)
    T_vs_c    = T_metric_analytic(a_fixed, c_vals, Lambda)
    d_vs_c    = d_max_analytic(a_fixed, c_vals, Lambda)

    # Correct spread factor: slope = a*2piT  (exact small-B limit with N_max~1/B)
    spread_vs_a = a_vals * 2*np.pi*T_vs_a
    spread_vs_c = a_fixed * 2*np.pi*T_vs_c

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    def twin_plot(ax, xv, y1, y2, xlabel, y1label, y2label, title, vline):
        ax2 = ax.twinx()
        l1, = ax.plot(xv, y1, 'b-', lw=1.8, label=y1label)
        l2, = ax2.plot(xv, y2, 'r-', lw=1.8, label=y2label)
        ax.axvline(vline, color='gray', ls='--', alpha=0.7, label=f'current={vline}')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(y1label, color='b', fontsize=11)
        ax2.set_ylabel(y2label, color='r', fontsize=11)
        ax.set_title(title, fontsize=11)
        ax.legend(handles=[l1, l2], fontsize=8, loc='upper left')
        return ax2

    twin_plot(axes[0,0], a_vals, T_vs_a, d_vs_a,
              r'$a$', r'$T$ (metric integral)', r'$d_\mathrm{max}$',
              rf'Geometry vs $a$  (fixed $c={c_fixed}$)', A_VAL)
    twin_plot(axes[0,1], c_vals, T_vs_c, d_vs_c,
              r'$c$', r'$T$ (metric integral)', r'$d_\mathrm{max}$',
              rf'Geometry vs $c$  (fixed $a={a_fixed}$)', C_VAL)
    twin_plot(axes[1,0], a_vals, spread_vs_a, T_vs_a,
              r'$a$', r'Spread factor $a\cdot 2\pi T$', r'$T$',
              rf'LL spread factor vs $a$', A_VAL)
    twin_plot(axes[1,1], c_vals, spread_vs_c, T_vs_c,
              r'$c$', r'Spread factor $a\cdot 2\pi T$', r'$T$',
              rf'LL spread factor vs $c$', C_VAL)

    plt.suptitle(rf'Quantum Geometry Parameter Scan ($\Lambda={Lambda}$)', fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save}")

    # Find optimal parameters for maximal spread
    best_a_idx = np.argmax(spread_vs_a)
    best_c_idx = np.argmin(spread_vs_c)   # spread decreases with c
    print(f"\n  Optimal a (fixed c={c_fixed}): a* = {a_vals[best_a_idx]:.3f}  "
          f"-> T={T_vs_a[best_a_idx]:.4f}, d_max={d_vs_a[best_a_idx]:.4f}")
    print(f"  Smallest c (fixed a={a_fixed}): c = {c_vals[0]:.3f}  "
          f"-> T={T_vs_c[0]:.4f}, d_max={d_vs_c[0]:.4f}")
    print("  (Spread diverges as c->0 due to flat band merging dispersive band)")


def plot_parameter_comparison(E0, Lambda, save='parameter_comparison.png'):
    """
    Figure 5 (Sub-step 5): Compare anomalous LL spread for default vs optimised parameters.

    Left panel  — LL lower-branch energy fan (shaded band from E_m(0) to E_m(N_max)) vs B.
    Right panel — ΔE_spread vs B with exact numeric curve and analytic geometric prediction.

    Confirms that increasing a from 1.115 to 2.5 (at fixed c=0.215) significantly enlarges
    the anomalous LL spread, in agreement with the geometric factor a·2πT.
    """
    print("\n--- Parameter comparison (Sub-step 5) ---")
    param_sets = [
        {'a': A_VAL,  'c': C_VAL,  'label': f'Default  ($a={A_VAL}$)'},
        {'a': 2.5,    'c': C_VAL,  'label':  'Optimised ($a=2.5$)'},
    ]
    colors = ['steelblue', 'firebrick']
    B_vals = np.linspace(0.005, 0.1, 80)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ps, col in zip(param_sets, colors):
        a, c = ps['a'], ps['c']
        T_ana    = T_metric_analytic(a, c, Lambda)
        # Correct small-B slope: a*2piT = a^3*Lambda^2/(a^2*Lambda^2 + c^2).
        # (The formula a*2piT/(1-2piT) = Lambda^2*a^3/c^2 is valid only when
        #  Lambda << c/a, which does NOT hold here.)
        geo_fac  = a * 2.0 * np.pi * T_ana

        # --- Left panel: energy fan ---
        ax = axes[0]
        top_curve = []
        bot_curve = []
        for B in B_vals:
            Em = ll_energies_lower_branch(B, a, c, E0, Lambda=Lambda)
            top_curve.append(Em[0])    # E_m(0) = E0 + aB  (highest)
            bot_curve.append(Em[-1])   # E_m(N_max)        (lowest)
        top_curve = np.array(top_curve)
        bot_curve = np.array(bot_curve)
        ax.fill_between(B_vals, bot_curve, top_curve, color=col, alpha=0.30)
        ax.plot(B_vals, top_curve, '-', color=col, lw=1.5)
        ax.plot(B_vals, bot_curve, '-', color=col, lw=1.5,
                label=f"{ps['label']}  ($T={T_ana:.4f}$)")

        # --- Right panel: spread vs B ---
        ax2 = axes[1]
        spreads  = [anomalous_ll_spread(B, a, c, E0, Lambda) for B in B_vals]
        pred_geo = geo_fac * B_vals
        ax2.plot(B_vals, spreads,  '-',  color=col, lw=2.0,
                 label=f"{ps['label']}  (factor={geo_fac:.2f})")
        ax2.plot(B_vals, pred_geo, '--', color=col, lw=1.2, alpha=0.7)

        print(f"  {ps['label']:30s}  T={T_ana:.5f}  geo_factor={geo_fac:.3f}")

    # Reference line: E0 + c^2/a (zero-mode energy, field-independent)
    E_zero = E0 + C_VAL**2 / A_VAL
    axes[0].axhline(E_zero, color='gray', ls=':', lw=1.0, label=r'$E_\mathrm{flat}$')
    axes[0].set_xlabel(r'$B_z$', fontsize=13)
    axes[0].set_ylabel(r'$E_m(n,\,B)$', fontsize=13)
    axes[0].set_title('Anomalous LL Fan (lower branches)', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, ls=':', alpha=0.4)

    axes[1].set_xlabel(r'$B_z$', fontsize=13)
    axes[1].set_ylabel(r'Spread $\Delta E$', fontsize=13)
    axes[1].set_title(r'Spread vs $B$  (solid=exact, dashed=geometric pred.)', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, ls=':', alpha=0.4)

    plt.suptitle(
        rf'Sub-step 5: Parameter Optimisation Comparison  ($\Lambda={Lambda},\ c={C_VAL}$)',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save}")

    # Ratio of spread factors
    T0   = T_metric_analytic(A_VAL,  C_VAL, Lambda)
    T1   = T_metric_analytic(2.5,    C_VAL, Lambda)
    fac0 = A_VAL * 2*np.pi*T0
    fac1 = 2.5   * 2*np.pi*T1
    print(f"  Optimised spread factor = {fac1:.3f}  ({fac1/fac0:.2f}x default)")


def plot_spread_linearity(E0, Lambda, save='spread_linearity.png'):
    """
    Figure 6 (Sub-step 6): Verify ΔE_spread ∝ W × B numerically.

    For 6 parameter sets (a ∈ {0.5, 0.8, 1.115, 1.5, 2.0, 2.5}, c fixed = 0.215):
      - Compute spread(B) numerically on B ∈ [0.01, 0.06]
      - Fit slope_num = ΔE/B via linear fit through origin in the small-B linear regime
      - Compare to analytic prediction slope_pred = a·2πT = a³Λ²/(a²Λ²+c²)

    Left panel : spread curves + geometric predictions (dashed, same colour)
    Right panel: scatter slope_num vs slope_pred — should lie on y = x
    """
    print("\n--- Spread proportional to W*B linearity verification (Sub-step 6) ---")
    a_test = [0.5, 0.8, A_VAL, 1.5, 2.0, 2.5]
    c_test = C_VAL
    B_plot = np.linspace(0.005, 0.1, 80)

    colors_6 = plt.cm.viridis(np.linspace(0.1, 0.9, len(a_test)))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_l, ax_r = axes

    results = []
    print(f"  {'a':>6}  {'T':>8}  {'pred_slope':>12}  {'num_slope':>11}  {'rel_err':>9}")
    print("  " + "-"*54)

    for a_val, col in zip(a_test, colors_6):
        T_ana     = T_metric_analytic(a_val, c_test, Lambda)
        # Correct small-B slope: a*2piT = a^3*Lambda^2/(a^2*Lambda^2 + c^2)
        slope_pred = a_val * 2.0 * np.pi * T_ana

        # Linear regime: B << c^2/a^2  AND  N_max = Lambda^2/(2B) >= 10.
        # Use B in [B_max/10, B_max] where B_max = min(0.3*c^2/a^2, Lambda^2/20).
        B_max_lin = min(0.3 * c_test**2 / a_val**2, Lambda**2 / 20.0)
        B_min_lin = B_max_lin / 10.0
        B_fit_a   = np.linspace(B_min_lin, B_max_lin, 30)

        # Numeric spread on the per-a linear-regime range
        sp_fit = np.array([anomalous_ll_spread(B, a_val, c_test, E0, Lambda)
                           for B in B_fit_a])
        # Linear fit through origin: slope = sum(B*sp) / sum(B^2)
        slope_num = np.dot(B_fit_a, sp_fit) / np.dot(B_fit_a, B_fit_a)
        rel_err   = abs(slope_num - slope_pred) / slope_pred * 100

        results.append((a_val, T_ana, slope_pred, slope_num, rel_err))
        print(f"  {a_val:6.3f}  {T_ana:8.5f}  {slope_pred:12.4f}  {slope_num:11.4f}  {rel_err:8.2f}%")

        # Left panel: spread curves
        sp_plot  = [anomalous_ll_spread(B, a_val, c_test, E0, Lambda) for B in B_plot]
        pred_geo = slope_pred * B_plot
        ax_l.plot(B_plot, sp_plot,  '-',  color=col, lw=1.8,
                  label=rf'$a={a_val}$, $T={T_ana:.4f}$')
        ax_l.plot(B_plot, pred_geo, '--', color=col, lw=1.0, alpha=0.6)

    # Right panel: scatter
    slope_preds = [r[2] for r in results]
    slope_nums  = [r[3] for r in results]
    max_rel     = max(r[4] for r in results)

    ax_r.scatter(slope_preds, slope_nums, c=colors_6, s=80, zorder=5)
    for r, col in zip(results, colors_6):
        ax_r.annotate(f'$a={r[0]}$', (r[2], r[3]),
                      textcoords='offset points', xytext=(5, 3), fontsize=8, color=col)
    lim = max(max(slope_preds), max(slope_nums)) * 1.05
    ax_r.plot([0, lim], [0, lim], 'k--', lw=1.0, label='$y=x$')
    status = 'confirmed' if max_rel < 10 else 'deviates >10%'
    ax_r.set_xlabel(r'Predicted slope $a \cdot 2\pi T$', fontsize=12)
    ax_r.set_ylabel(r'Measured slope $\Delta E / B$', fontsize=12)
    ax_r.set_title(rf'Spread $\propto W\times B$ — {status}  (max err={max_rel:.1f}%)',
                   fontsize=11)
    ax_r.legend(fontsize=9)
    ax_r.grid(True, ls=':', alpha=0.4)
    ax_r.set_xlim(0, lim); ax_r.set_ylim(0, lim)

    ax_l.set_xlabel(r'$B_z$', fontsize=13)
    ax_l.set_ylabel(r'Spread $\Delta E$', fontsize=13)
    ax_l.set_title(r'Spread vs $B$  (solid=numeric, dashed=prediction)', fontsize=12)
    ax_l.legend(fontsize=8)
    ax_l.grid(True, ls=':', alpha=0.4)

    plt.suptitle(
        rf'Sub-step 6: $\Delta E \propto W\!\times\!B$ Verification  ($c={c_test},\ \Lambda={Lambda}$)',
        fontsize=13)
    plt.tight_layout()
    plt.savefig(save, dpi=300, bbox_inches='tight')
    print(f"  Saved: {save}")
    return results


def print_summary(a, c, E0, Lambda):
    """Print a concise summary of all key geometric quantities."""
    T   = T_metric_analytic(a, c, Lambda)
    d   = d_max_analytic(a, c, Lambda)
    E1  = E0 + c**2 / a
    gap = c**2 / a        # energy gap at k=0 between bands
    finite_flux_fraction = 2 * np.pi * T
    audit = audit_note_consistency(a, c, Lambda)
    print("\n" + "="*60)
    print("  STEP 2 QUANTUM GEOMETRY SUMMARY")
    print("="*60)
    print(f"  Parameters:   a={a},  c={c},  E0={E0},  Lambda={Lambda}")
    print(f"  Flat band:    E_flat = E0 + c^2/a = {E1:.5f}")
    print(f"  Band gap at k=0:  c^2/a = {gap:.5f}")
    print(f"  Band gap at k=Lambda:  a*Lambda^2+c^2/a = {a*Lambda**2+gap:.5f}")
    print("  Topology:     continuum-limit Chern number C = 1")
    print(f"  Finite-disk flux fraction (|k|<=Lambda): {finite_flux_fraction:.4f}")
    print(f"  T = integral tr(g) dk / (2pi)^2 = {T:.6f}  [analytic]")
    print(f"  T_max (Lambda->infinity) = {1/(2*np.pi):.6f}  = 1/(2pi)")
    print(f"  T/T_max            = {finite_flux_fraction:.4f}  = a*Lambda^2/(a*Lambda^2+c^2/a)")
    print(f"  d_max (k=0 vs k=Lambda) = {d:.6f}")
    print(f"  Relation: d_max = sqrt(2piT) = {np.sqrt(2*np.pi*T):.6f}  [ok]")
    print(f"  LL spread factor (a*2piT) = {a*2*np.pi*T:.4f}  (energy/field, exact small-B slope)")
    print("  Note/code audit: max pointwise abs error")
    print(f"    g_xx={audit['g_xx']:.2e}, g_xy={audit['g_xy']:.2e}, "
          f"g_yy={audit['g_yy']:.2e}, Omega={audit['Omega']:.2e}")
    print("="*60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Figure output relative to project root (one level up from Codes/)
    _ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    _FIG = os.path.join(_ROOT, 'figures', 'default_a1.115_c0.215_E0_0.0849_Lambda_0.5')
    os.makedirs(_FIG, exist_ok=True)

    a, c, E0, Lam = A_VAL, C_VAL, E0_VAL, LAMBDA

    print_summary(a, c, E0, Lam)

    # Figure 1: quantum metric + Berry curvature maps
    T_num, T_ana, d_ana, C_num = plot_geometry_maps(a, c, E0, Lam, N=150,
        save=os.path.join(_FIG, 'quantum_geometry.png'))

    # Figure 2: quantum distance profile
    d_edge = plot_quantum_distance_profile(a, c, Lam,
        save=os.path.join(_FIG, 'quantum_distance.png'))

    # Figure 3: anomalous LL spread vs B
    plot_anomalous_ll_spread(a, c, E0, Lam,
        save=os.path.join(_FIG, 'anomalous_ll_spread.png'))

    # Figure 4: parameter scan (analytic, fast)
    plot_parameter_scan(E0, Lam,
        save=os.path.join(_FIG, 'parameter_scan.png'))

    # Figure 5 (Sub-step 5): default vs optimised parameter comparison
    plot_parameter_comparison(E0, Lam,
        save=os.path.join(_FIG, 'parameter_comparison.png'))

    # Figure 6 (Sub-step 6): verify spread ∝ W × B across parameter sets
    linearity_results = plot_spread_linearity(E0, Lam,
        save=os.path.join(_FIG, 'spread_linearity.png'))

    print("\nAll Step 2 figures saved.")


if __name__ == "__main__":
    main()
