import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.special import erf

# --- Band Structure Part ---

def ham_chern_thf(k, E0, E1, a, c):
    """
    Hamiltonian for the Topological Heavy Fermion (THF) model.
    """
    kx, ky = k
    
    h11 = E0 + a * (kx**2 + ky**2)
    h12 = c * (kx + 1j * ky)
    h21 = c * (kx - 1j * ky)
    h22 = E1
    
    return np.array([[h11, h12], [h21, h22]])

def get_eigenvalues(kx, ky, E0, E1, a, c):
    """
    Calculate eigenvalues for a grid of k points.
    """
    evals = np.zeros((len(kx), 2))
    
    for i in range(len(kx)):
        H = ham_chern_thf((kx[i], ky[i]), E0, E1, a, c)
        e = np.linalg.eigvalsh(H)
        evals[i] = np.sort(e)
        
    return evals

def generate_path(points, n_points=50):
    k_path_flat = []
    tick_indices = [0]
    current_idx = 0
    
    for i in range(len(points) - 1):
        start = np.array(points[i])
        end = np.array(points[i+1])
        ts = np.linspace(0, 1, n_points)
        for t in ts:
            p = (1-t)*start + t*end
            k_path_flat.append(p)
            current_idx += 1
        tick_indices.append(current_idx)
        
    return np.array(k_path_flat), tick_indices

def plot_band_structure(a_val=1.115, c_val=0.215, E0_val=0.0849, E1_val=0.1123,
                        filename='ideal_flatband_dispersion.png'):
    print("\n--- Calculating Band Structure ---")
    E1_flat = E0_val + (c_val**2) / a_val
    
    print(f"Parameters: a={a_val}, c={c_val}, E0={E0_val}")
    print(f"E1 (original) = {E1_val}")
    print(f"E1 (flat band condition) = {E1_flat}")
    
    pG = np.array([0.0, 0.0])
    pK = np.array([0.0, 0.5])
    pM = np.array([0.25, np.sqrt(3)/4])
    
    path_points = [pM, pG, pK]
    labels = ['M', '$\Gamma$', 'K']
    
    n_points = 50
    k_vecs, tick_indices = generate_path(path_points, n_points)
    
    evals_original = get_eigenvalues(k_vecs[:, 0], k_vecs[:, 1], E0_val, E1_val, a_val, c_val)
    evals_flat = get_eigenvalues(k_vecs[:, 0], k_vecs[:, 1], E0_val, E1_flat, a_val, c_val)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax = axes[0]
    ax.plot(evals_original[:, 0], 'r-', label='Lower Band')
    ax.plot(evals_original[:, 1], 'b-', label='Upper Band')
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Energy')
    ax.set_title(f'Band Structure (E1={E1_val})')
    ax.grid(True, alpha=0.3)
    for x in tick_indices:
        ax.axvline(x, color='k', linestyle='--', alpha=0.5)
        
    ax = axes[1]
    ax.plot(evals_flat[:, 0], 'r-', label='Lower Band')
    ax.plot(evals_flat[:, 1], 'b-', label='Upper Band')
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(labels)
    ax.set_title(f'Ideal Flat Band (E1={E1_flat:.4f})')
    ax.grid(True, alpha=0.3)
    for x in tick_indices:
        ax.axvline(x, color='k', linestyle='--', alpha=0.5)
    
    ax.set_ylim(0.05, 0.25)
        
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Band structure plot saved to {filename}")

# --- Landau Level Part ---

def landau_level_energies_LL(B_val, n, a, c, E0):
    """
    Calculates Landau Level energies for the 'LL' sector (LLp, LLm) based on Mathematica notebook.
    Corresponds to LLp and LLm in the notebook.
    Formula: E = E0 + 1/2 * ( (c^2/a + (1+2n)aB) ± sqrt((c^2/a + (1+2n)aB)^2 - 4 c^2 B) )
    
    Indexing note:
    - For the full spectrum and DOS, use this branch formula with n >= 1.
    - The n = 0 case belongs to the two zeroth LLs and should be handled
      separately.
    """
    # Ensure inputs are valid for vectorization if needed, but here simple scalars/arrays work
    term1 = (c**2) / a
    term2 = (1 + 2 * n) * a * B_val
    
    # Discriminant inside sqrt
    # Note: Notebook has -4 beta^2 Bz. In Python c is beta.
    discriminant = (term1 + term2)**2 - 4 * (c**2) * B_val
    
    # Ensure non-negative discriminant for sqrt
    sqrt_val = np.sqrt(np.maximum(discriminant, 0))
    
    E_p = E0 + 0.5 * (term1 + term2 + sqrt_val)
    E_m = E0 + 0.5 * (term1 + term2 - sqrt_val)
    
    return E_p, E_m

def landau_level_energies_LLTR(B_val, n, a, c, E0):
    """
    Calculates Landau Level energies for the 'LLTR' sector (LLTRp, LLTRm) based on Mathematica notebook.
    Corresponds to LLTRp and LLTRm in the notebook.
    Formula: E = E0 + 1/2 * ( (c^2/a + (2n-1)aB) ± sqrt((c^2/a + (2n-1)aB)^2 + 4 c^2 B) )
    
    Indexing note:
    - For the full spectrum and DOS, use this branch formula with n >= 1.
    - The LLTR zeroth level is handled separately by
      landau_level_energies_LLTRZero().
    """
    term1 = (c**2) / a
    term2 = a * B_val * (2 * n - 1)
    
    # Discriminant inside sqrt
    # Note: Notebook has +4 beta^2 Bz.
    discriminant = (term1 + term2)**2 + 4 * (c**2) * B_val
    
    sqrt_val = np.sqrt(np.maximum(discriminant, 0))
    
    E_p = E0 + 0.5 * (term1 + term2 + sqrt_val)
    E_m = E0 + 0.5 * (term1 + term2 - sqrt_val)
    
    return E_p, E_m

def landau_level_energies_LLTRZero(a, c, E0):
    """
    Calculates the zero mode energy for the 'LLTR' sector.
    """
    return E0 + (c**2) / a

def landau_level_energy_LLL(B_val, a, E0):
    """
    Zeroth LL in the LL sector.
    """
    return E0 + a * B_val


def landau_level_energy_LLL(B_val, a, E0):
    """
    Zeroth LL in the LL sector:
        E_LLL = E0 + aB
    """
    return E0 + a * B_val

def plot_landau_levels(a, c, E0, B_max=0.3, num_points=200, N_max=5,
                       filename='landau_levels_spectrum.png'):
    """
    Plots the Landau level spectrum matching the Mathematica notebook's style.
    Combines LL and LLTR sectors.
    """
    print("\n--- Calculating Landau Levels ---")
    B_values = np.linspace(0.001, B_max, num_points) # Start from small B to avoid potential singularities if any
    
    plt.figure(figsize=(8, 6))
    
    # --- Plot the two zeroth LLs separately ---
    # The LL-sector n=0 lower branch gives E0 + aB, while the LLTR zero mode
    # gives E0 + c^2/a. The LL-sector n=0 upper branch coincides with the LLTR
    # zero mode and must not be plotted as an additional independent level.
    E_zero = landau_level_energies_LLTRZero(a, c, E0)
    E_lll = landau_level_energy_LLL(B_values, a, E0)
    plt.plot(B_values, np.full_like(B_values, E_zero), 'k--', linewidth=2,
             label='Zero mode: $E_0+c^2/a$')
    plt.plot(B_values, E_lll, color='darkblue', linestyle='-', linewidth=2,
             label='Zero mode: $E_0+aB$')
    
    # Plot LLTRp (Red Dashed) and LLTRm (Blue Dashed) for n >= 1
    for n in range(1, N_max + 1):
        E_p_list = []
        E_m_list = []
        for B in B_values:
            Ep, Em = landau_level_energies_LLTR(B, n, a, c, E0)
            E_p_list.append(Ep)
            E_m_list.append(Em)
        
        # Only label the first ones to avoid cluttered legend
        l_p = 'LLTR p (n>=1)' if n==1 else None
        l_m = 'LLTR m (n>=1)' if n==1 else None
        plt.plot(B_values, E_p_list, 'r--', linewidth=2, label=l_p)
        plt.plot(B_values, E_m_list, 'b--', linewidth=2, label=l_m)

    # --- LL Sector (excited branches only: n >= 1) ---
    for n in range(1, N_max + 1):
        E_p_list = []
        E_m_list = []
        for B in B_values:
            Ep, Em = landau_level_energies_LL(B, n, a, c, E0)
            E_p_list.append(Ep)
            E_m_list.append(Em)
            
        l_p = 'LL p (n>=1)' if n == 1 else None
        l_m = 'LL m (n>=1)' if n == 1 else None
        
        plt.plot(B_values, E_p_list, color='darkred', linestyle='-', linewidth=1.5, alpha=0.6, label=l_p)
        plt.plot(B_values, E_m_list, color='darkblue', linestyle='-', linewidth=1.5, alpha=0.6, label=l_m)

    plt.xlabel(r'$B_z$ (Magnetic Field)', fontsize=14)
    plt.ylabel(r'Energy ($E$)', fontsize=14)
    plt.title(f'Landau Levels vs Magnetic Field\n$a={a}, c={c}, E_0={E0}$', fontsize=16)
    plt.xlim(0, B_max)
    plt.ylim(0.05, 0.3) # Matches notebook range approximately
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='best')
    
    # Save the plot
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Landau level plot saved to {filename}")

# --- DOS Oscillation Part ---

def gaussian_broadened(E, E_level, Gamma):
    """
    Gaussian broadening kernel G(E, E_level, Gamma).

    Formula (from Mathematica notebook):
        G = (1 / sqrt(4 * pi * Gamma)) * exp(-(E - E_level)^2 / (4 * Gamma))

    A cutoff is applied when (E - E_level)^2 / (4 * Gamma) > 700 to avoid
    numerical underflow in exp(), matching the Mathematica notebook exactly.
    """
    x = (E - E_level) ** 2 / (4.0 * Gamma)
    if x > 700.0:
        return 0.0
    return (1.0 / np.sqrt(4.0 * np.pi * Gamma)) * np.exp(-x)


def dos_at_energy(E, B_val, a, c, E0, Gamma, N_max=20):
    """
    Total density of states at energy E and magnetic field B_val.

    Degeneracy per Landau level: Deg = B_val / (2 * pi)

    Three contributions (matching Mathematica LLDOSFun):
      DOSLL0   : Two zero modes —
                   EngLL0up  = E0 + a*B  (LL sector n=0, c-orbital LLL, field-linear)
                   EngLL0down = E0 + c^2/a (LLTR zero mode, f-orbital, field-independent)
                 The n=0 upper branch of the LL sector coincides exactly with EngLL0down
                 and is NOT counted separately (it is the same state).
      DOSLLup  : LL sector, n = 1..N_max, upper (+) and lower (-) branches
      DOSLLdown: LLTR sector, n = 1..N_max, upper (+) and lower (-) branches

    Total DOS = Deg * (DOSLL0 + DOSLLup + DOSLLdown)
    """
    Deg = B_val / (2.0 * np.pi)

    # DOSLL0: two zero modes (one per sector at n=0)
    E_LLL_up   = E0 + a * B_val                      # LL sector LLL (c-orbital, field-linear)
    E_zero_down = landau_level_energies_LLTRZero(a, c, E0)  # LLTR zero mode (f-orbital, fixed)
    DOSLL0 = gaussian_broadened(E, E_LLL_up, Gamma) + gaussian_broadened(E, E_zero_down, Gamma)

    # DOSLLup: LL sector, n = 1..N_max (both branches); n=0 already in DOSLL0
    DOSLLup = 0.0
    for n in range(1, N_max + 1):
        E_p, E_m = landau_level_energies_LL(B_val, n, a, c, E0)
        DOSLLup += gaussian_broadened(E, E_p, Gamma)
        DOSLLup += gaussian_broadened(E, E_m, Gamma)

    # DOSLLdown: LLTR sector, n = 1..N_max (both branches)
    DOSLLdown = 0.0
    for n in range(1, N_max + 1):
        E_p, E_m = landau_level_energies_LLTR(B_val, n, a, c, E0)
        DOSLLdown += gaussian_broadened(E, E_p, Gamma)
        DOSLLdown += gaussian_broadened(E, E_m, Gamma)

    return Deg * (DOSLL0 + DOSLLup + DOSLLdown)


def integrated_dos(E_low, E_high, B_val, a, c, E0, Gamma, N_max=20):
    """
    Analytically integrate the DOS from E_low to E_high at fixed B_val.

    Since each Landau level contributes a Gaussian G(E, E_level, Gamma), and
        integral G(E) dE = (1/2)[erf((E_high-E_level)/(2*sqrt(Gamma)))
                                - erf((E_low -E_level)/(2*sqrt(Gamma)))]
    the total integral is a closed-form sum — no numerical quadrature needed.
    This is exact and handles arbitrarily narrow peaks (small Gamma) efficiently.
    """
    Deg = B_val / (2.0 * np.pi)
    sig = 2.0 * np.sqrt(Gamma)   # denominator: 2*sqrt(Gamma)

    def gauss_integral(E_level):
        return 0.5 * (erf((E_high - E_level) / sig) - erf((E_low - E_level) / sig))

    total = 0.0

    # Zero modes
    total += gauss_integral(E0 + a * B_val)
    total += gauss_integral(landau_level_energies_LLTRZero(a, c, E0))

    # LL sector n = 1..N_max
    for n in range(1, N_max + 1):
        Ep, Em = landau_level_energies_LL(B_val, n, a, c, E0)
        total += gauss_integral(Ep) + gauss_integral(Em)

    # LLTR sector n = 1..N_max
    for n in range(1, N_max + 1):
        Ep, Em = landau_level_energies_LLTR(B_val, n, a, c, E0)
        total += gauss_integral(Ep) + gauss_integral(Em)

    return Deg * total


def find_fermi_energy(rho, B_val, a, c, E0, Gamma, Lambda):
    """
    Find the Fermi energy EF such that the filling equals rho.

    Key quantities:
        E_LLL_up    = E0 + a*B_val     (LL sector LLL: c-orbital, field-linear)
        E_zero_down = E0 + c^2/a       (LLTR zero mode: f-orbital, field-independent)
        E_ref       = (E_LLL_up + E_zero_down) / 2   (B-dependent midpoint; Mathematica E0Mean)
        N_max_eff   = max(1, floor(Lambda^2 / (2*Bz)))  (B-dependent, matches Mathematica)
        rhoUpMax    = integral of DOS from E0 to E_ref  (Mathematica HoleDensityUpFun[0,...])

    Three cases (matching Mathematica SolveFermiEnergyFun):
        rho == 0              : EF = E_ref
        rho > 0               : electrons; EF above E_ref
                                  solve: integral(E_ref -> EF) DOS = rho
        abs(rho) <= rhoUpMax  : small hole doping; E0 <= EF < E_ref
                                  solve: integral(EF -> E_ref) DOS = abs(rho)
        abs(rho) > rhoUpMax   : large hole doping; EF < E0
                                  solve: integral(EF -> E0) DOS = abs(rho) - rhoUpMax

    Root-finding via scipy.optimize.brentq (Brent's method).
    Returns np.nan if the bracket cannot be found.
    """
    E_LLL_up    = E0 + a * B_val
    E_zero_down = landau_level_energies_LLTRZero(a, c, E0)
    E_ref = 0.5 * (E_LLL_up + E_zero_down)   # B-dependent midpoint (Mathematica E0Mean)

    # B-dependent NMax — matches Mathematica: NMax = Max[1, Floor[Lambda^2 / (2*Bz)]]
    N_max_eff = max(1, int(Lambda**2 / (2.0 * B_val)))

    if rho == 0.0:
        return E_ref

    # rhoUpMax: DOS integrated from E0 to E_ref (Mathematica HoleDensityUpFun[0,...])
    rhoUpMax = integrated_dos(E0, E_ref, B_val, a, c, E0, Gamma, N_max_eff)

    if rho > 0:
        # Case 1: electron doping — EF above E_ref
        search_width = max(0.15, 5.0 * a * B_val * N_max_eff)
        E_lo = E_ref + 1e-10
        E_hi = E_ref + search_width

        def equation(EF):
            return integrated_dos(E_ref, EF, B_val, a, c, E0, Gamma, N_max_eff) - rho

        # Expand bracket if needed
        for _ in range(5):
            if equation(E_hi) > 0:
                break
            E_hi *= 2.0

        try:
            return brentq(equation, E_lo, E_hi, xtol=1e-10, rtol=1e-8, maxiter=200)
        except ValueError:
            return np.nan


    elif abs(rho) <= rhoUpMax:
        # Case 2: small hole doping — E0 <= EF < E_ref
        # Matches Mathematica: HoleDensityUpFun integrates [EF, E_ref], bracket {ef, 0, E0Mean}
        E_lo = E0
        E_hi = E_ref - 1e-10

        def equation(EF):
            return integrated_dos(EF, E_ref, B_val, a, c, E0, Gamma, N_max_eff) - abs(rho)

        try:
            return brentq(equation, E_lo, E_hi, xtol=1e-10, rtol=1e-8, maxiter=200)
        except ValueError:
            return np.nan

    else:
        # Case 3: large hole doping — EF < E0
        # Matches Mathematica: HoleDensityDownFun[ef] == abs(rho) - rhoUpMax
        # HoleDensityDownFun integrates from EF to 0 (= E0 in Python), for EF < E0
        search_width = max(0.15, 5.0 * a * B_val * N_max_eff)
        E_lo = E0 - 5.0 * search_width
        E_hi = E0 - 1e-10
        target = abs(rho) - rhoUpMax

        def equation(EF):
            return integrated_dos(EF, E0, B_val, a, c, E0, Gamma, N_max_eff) - target

        try:
            return brentq(equation, E_lo, E_hi, xtol=1e-10, rtol=1e-8, maxiter=200)
        except ValueError:
            return np.nan


def carrier_density_from_fermi_energy(EF, B_val, a, c, E0, Gamma, Lambda):
    """
    Compute the carrier density rho corresponding to a chosen Fermi energy EF.

    Sign convention matches find_fermi_energy(...):
      rho > 0 : EF above E_ref
      rho < 0 : EF below E_ref
    """
    E_LLL_up = E0 + a * B_val
    E_zero_down = landau_level_energies_LLTRZero(a, c, E0)
    E_ref = 0.5 * (E_LLL_up + E_zero_down)
    N_max_eff = max(1, int(Lambda**2 / (2.0 * B_val)))

    if EF >= E_ref:
        return integrated_dos(E_ref, EF, B_val, a, c, E0, Gamma, N_max_eff)
    return -integrated_dos(EF, E_ref, B_val, a, c, E0, Gamma, N_max_eff)


def plot_density_vs_fermi_energy(a, c, E0, Gamma, Lambda, B_fixed=0.02,
                                 filename='density_vs_fermi_energy.png'):
    """
    Plot carrier density rho as a function of Fermi energy at fixed magnetic field.
    """
    E_LLL_up = E0 + a * B_fixed
    E_zero_down = landau_level_energies_LLTRZero(a, c, E0)
    E_ref = 0.5 * (E_LLL_up + E_zero_down)
    N_max_eff = max(1, int(Lambda**2 / (2.0 * B_fixed)))
    rho_max = Lambda**2 / (4.0 * np.pi)

    half_range = max(0.08, 3.0 * a * B_fixed * N_max_eff)
    E_grid = np.linspace(E_ref - half_range, E_ref + half_range, 1500)
    rho_grid = np.array([
        carrier_density_from_fermi_energy(E, B_fixed, a, c, E0, Gamma, Lambda)
        for E in E_grid
    ])

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(E_grid, rho_grid, color='darkgreen', linewidth=1.8)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=1.0)
    ax.axvline(E_ref, color='gray', linestyle='--', linewidth=1.0,
               label=rf'$E_\mathrm{{ref}}={E_ref:.4f}$')
    ax.axvline(E_LLL_up, color='darkblue', linestyle=':', linewidth=1.0,
               label=rf'$E_0+aB={E_LLL_up:.4f}$')
    ax.axvline(E_zero_down, color='black', linestyle=':', linewidth=1.0,
               label=rf'$E_0+c^2/a={E_zero_down:.4f}$')
    ax.axhline(rho_max, color='firebrick', linestyle=':', linewidth=1.0,
               label=rf'$\rho_\max={rho_max:.4f}$')
    ax.axhline(-rho_max, color='firebrick', linestyle=':', linewidth=1.0)
    ax.set_xlabel(r'Fermi energy $E_F$', fontsize=13)
    ax.set_ylabel(r'Carrier density $\rho$', fontsize=13)
    ax.set_title(rf'Carrier Density vs Fermi Energy at $B_z={B_fixed}$'
                 rf'  ($a={a},\ c={c}$)', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def plot_dos_vs_density(a, c, E0, Gamma, Lambda, B_fixed=0.02,
                        filename='dos_vs_density.png'):
    """
    Plot DOS as a function of carrier density rho at fixed magnetic field.

    The curve is constructed parametrically from an energy grid:
      E -> (rho(E), DOS(E))
    using the same full-model density convention as the rest of the code.
    """
    E_LLL_up = E0 + a * B_fixed
    E_zero_down = landau_level_energies_LLTRZero(a, c, E0)
    E_ref = 0.5 * (E_LLL_up + E_zero_down)
    N_max_eff = max(1, int(Lambda**2 / (2.0 * B_fixed)))
    rho_max_full = Lambda**2 / (2.0 * np.pi)

    half_range = max(0.08, 3.0 * a * B_fixed * N_max_eff)
    E_grid = np.linspace(E_ref - half_range, E_ref + half_range, 2000)
    rho_grid = np.array([
        carrier_density_from_fermi_energy(E, B_fixed, a, c, E0, Gamma, Lambda)
        for E in E_grid
    ])
    dos_grid = np.array([
        dos_at_energy(E, B_fixed, a, c, E0, Gamma, N_max_eff)
        for E in E_grid
    ])

    order = np.argsort(rho_grid)
    rho_sorted = rho_grid[order]
    dos_sorted = dos_grid[order]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(rho_sorted, dos_sorted, color='purple', linewidth=1.8)
    ax.axvline(0.0, color='gray', linestyle='--', linewidth=1.0)
    ax.axvline(rho_max_full, color='firebrick', linestyle=':', linewidth=1.0,
               label=rf'$\rho_{{\max,\ full}}={rho_max_full:.4f}$')
    ax.axvline(-rho_max_full, color='firebrick', linestyle=':', linewidth=1.0)
    ax.set_xlabel(r'Carrier density $\rho$', fontsize=13)
    ax.set_ylabel(r'DOS$(E_F(\rho), B_z)$', fontsize=13)
    ax.set_title(rf'DOS vs Carrier Density at $B_z={B_fixed}$'
                 rf'  ($a={a},\ c={c}$)', fontsize=13)
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.legend(fontsize=9, loc='best')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def plot_dos_vs_energy(a, c, E0, Gamma, Lambda, B_fixed=0.02,
                       rho_list=None, filename='dos_vs_energy.png'):
    """
    Figure 1: DOS(E) at a fixed magnetic field, with Fermi levels marked.

    Shows all Landau level peaks as a smooth broadened curve. Vertical dashed
    lines mark EF for each specified filling fraction in rho_list.

    Saves: dos_vs_energy.png
    """
    if rho_list is None:
        rho_list = [0.005, 0.01, 0.015, -0.005, -0.01]

    E_LLL_up    = E0 + a * B_fixed
    E_zero_down = landau_level_energies_LLTRZero(a, c, E0)
    E_ref = 0.5 * (E_LLL_up + E_zero_down)   # midpoint of the two zero modes

    # B-dependent NMax (matches Mathematica: NMax = Max[1, Floor[Lambda^2/(2*Bz)]])
    N_max_eff = max(1, int(Lambda**2 / (2.0 * B_fixed)))

    # Energy grid centred on E_ref, spanning several LL spacings
    half_range = max(0.08, 3.0 * a * B_fixed * N_max_eff)
    E_grid = np.linspace(E_ref - half_range, E_ref + half_range, 2000)
    DOS_grid = np.array([dos_at_energy(E, B_fixed, a, c, E0, Gamma, N_max_eff)
                         for E in E_grid])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(E_grid, DOS_grid, 'k-', linewidth=1.2, label='DOS')
    ax.axvline(E_LLL_up,    color='orange', linestyle=':', linewidth=1.0,
               label=rf'$E_\mathrm{{LLL}}={E_LLL_up:.4f}$')
    ax.axvline(E_zero_down, color='steelblue', linestyle=':', linewidth=1.0,
               label=rf'$E_\mathrm{{flat}}={E_zero_down:.4f}$')
    ax.axvline(E_ref,       color='gray',   linestyle='--', linewidth=1.0,
               label=rf'$E_\mathrm{{ref}}={E_ref:.4f}$')

    colors = plt.cm.tab10(np.linspace(0, 1, len(rho_list)))
    for rho, col in zip(rho_list, colors):
        EF = find_fermi_energy(rho, B_fixed, a, c, E0, Gamma, Lambda)
        if not np.isnan(EF):
            label = rf'$\rho={rho:+.3f}$,  $E_F={EF:.4f}$'
            ax.axvline(EF, color=col, linestyle='--', linewidth=1.2, label=label)

    ax.set_xlabel('Energy $E$', fontsize=13)
    ax.set_ylabel(r'DOS($E$)', fontsize=13)
    ax.set_title(rf'Density of States at $B_z={B_fixed}$   '
                 rf'($a={a},\ c={c},\ \Gamma={Gamma}$)', fontsize=13)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")

    # --- Zoomed-in version for the anomalous LL region ---
    E_lo, E_hi = 0.05, 0.15
    mask = (E_grid >= E_lo) & (E_grid <= E_hi)
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    ax2.plot(E_grid[mask], DOS_grid[mask], 'k-', linewidth=1.2, label='DOS')
    ax2.axvline(E_LLL_up,    color='orange',    linestyle=':', linewidth=1.0,
                label=rf'$E_\mathrm{{LLL}}={E_LLL_up:.4f}$')
    ax2.axvline(E_zero_down, color='steelblue', linestyle=':', linewidth=1.0,
                label=rf'$E_\mathrm{{flat}}={E_zero_down:.4f}$')
    ax2.axvline(E_ref,       color='gray',      linestyle='--', linewidth=1.0,
                label=rf'$E_\mathrm{{ref}}={E_ref:.4f}$')
    for rho, col in zip(rho_list, colors):
        EF = find_fermi_energy(rho, B_fixed, a, c, E0, Gamma, Lambda)
        if not np.isnan(EF) and E_lo <= EF <= E_hi:
            label = rf'$\rho={rho:+.3f}$,  $E_F={EF:.4f}$'
            ax2.axvline(EF, color=col, linestyle='--', linewidth=1.2, label=label)
    ax2.set_xlim(E_lo, E_hi)
    ax2.set_xlabel('Energy $E$', fontsize=13)
    ax2.set_ylabel(r'DOS($E$)', fontsize=13)
    ax2.set_title(rf'DOS (zoomed $E\in[{E_lo},{E_hi}]$) at $B_z={B_fixed}$   '
                  rf'($a={a},\ c={c},\ \Gamma={Gamma}$)', fontsize=13)
    ax2.legend(fontsize=9, loc='upper right')
    ax2.grid(True, linestyle=':', alpha=0.4)
    plt.tight_layout()
    zoom_filename = filename.replace('.png', '_zoom.png')
    plt.savefig(zoom_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {zoom_filename}")


def plot_fermi_energy_vs_B(a, c, E0, Gamma, Lambda,
                           B_range=(0.001, 0.1), num_B=200,
                           rho_list=None, filename='fermi_energy_vs_B.png'):
    """
    Figure 2: Fermi energy EF(Bz) vs magnetic field for several fixed fillings.

    Shows quantum oscillations: as Bz increases, successive Landau levels cross
    EF, causing step-like oscillations. Electron fillings (rho > 0) sit above
    E_flat; hole fillings (rho < 0) sit below.

    Saves: fermi_energy_vs_B.png
    """
    if rho_list is None:
        rho_list = [0.005, 0.01, 0.015, -0.005, -0.01]

    B_values = np.linspace(B_range[0], B_range[1], num_B)
    E_zero_down = landau_level_energies_LLTRZero(a, c, E0)
    E_ref_curve = 0.5 * (E0 + a * B_values + E_zero_down)   # E_ref(B) for all B

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(rho_list)))

    for rho, col in zip(rho_list, colors):
        EF_list = []
        for B in B_values:
            EF_list.append(find_fermi_energy(rho, B, a, c, E0, Gamma, Lambda))
        ax.plot(B_values, EF_list, color=col, linewidth=1.4,
                label=rf'$\rho={rho:+.3f}$')

    ax.axhline(E_zero_down, color='steelblue', linestyle=':', linewidth=1.0,
               label=r'$E_\mathrm{flat}$')
    ax.plot(B_values, E_ref_curve, color='gray', linestyle='--', linewidth=1.0,
            label=r'$E_\mathrm{ref}(B_z)$')
    ax.set_xlabel(r'$B_z$', fontsize=13)
    ax.set_ylabel(r'$E_F(B_z)$', fontsize=13)
    ax.set_title(rf'Fermi Energy vs Magnetic Field   '
                 rf'($a={a},\ c={c},\ \Gamma={Gamma}$)', fontsize=13)
    ax.legend(fontsize=10, loc='best')
    ax.grid(True, linestyle=':', alpha=0.4)
    ax.set_xlim(B_range)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def plot_dos_oscillations(a, c, E0, Gamma, Lambda,
                          B_range=(0.005, 0.1), num_B=500,
                          rho_val=0.02, filename='dos_oscillations.png'):
    """
    Figure 3: DOS evaluated at the Fermi energy vs 1/Bz — the dHvA-like signal.

    As 1/Bz increases (Bz decreases), LLs become more closely spaced and the
    Fermi level oscillates in and out of high-DOS peaks. Plotting vs 1/Bz makes
    the quantum oscillations periodic, matching standard dHvA presentation.

    Saves: dos_oscillations.png
    """
    B_values = np.linspace(B_range[0], B_range[1], num_B)
    inv_B = 1.0 / B_values

    DOS_at_EF = []
    EF_values = []
    for B in B_values:
        try:
            N_max_B = max(1, int(Lambda**2 / (2.0 * B)))
            EF = find_fermi_energy(rho_val, B, a, c, E0, Gamma, Lambda)
            dos = dos_at_energy(EF, B, a, c, E0, Gamma, N_max_B) if not np.isnan(EF) else np.nan
        except Exception:
            EF = np.nan
            dos = np.nan
        DOS_at_EF.append(dos)
        EF_values.append(EF)

    E_flat = E0 + c**2 / a

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10),
                                         gridspec_kw={'height_ratios': [1, 1, 1]})

    # --- Top panel: Fermi energy + LL spectrum vs 1/B ---
    EF_arr = np.array(EF_values)
    EF_valid = EF_arr[~np.isnan(EF_arr)]
    if len(EF_valid) > 0:
        y_lo = min(EF_valid.min(), E0) - 0.005
        y_hi = max(EF_valid.max(), E_flat) + 0.01
    else:
        y_lo, y_hi = E0 - 0.01, E_flat + 0.05

    N_max_plot = max(1, int(Lambda**2 / (2.0 * B_range[0])))
    ax1.plot(inv_B, [landau_level_energy_LLL(B, a, E0) for B in B_values],
             color='olive', linewidth=0.6, alpha=0.5)
    ax1.axhline(E_flat, color='steelblue', linestyle=':', linewidth=1.0,
                label=rf'$E_\mathrm{{flat}}={E_flat:.4f}$')
    for n in range(1, N_max_plot + 1):
        Ep_LL,  Em_LL  = landau_level_energies_LL(B_values, n, a, c, E0)
        Ep_TR, Em_TR = landau_level_energies_LLTR(B_values, n, a, c, E0)
        for Earr in [Ep_LL, Em_LL, Ep_TR, Em_TR]:
            if np.any((Earr >= y_lo) & (Earr <= y_hi)):
                ax1.plot(inv_B, Earr, color='olive', linewidth=0.5, alpha=0.4)

    ax1.plot(inv_B, EF_values, 'r-', linewidth=1.2, label=rf'$E_F$  ($\rho={rho_val}$)')
    ax1.axhline(E0, color='gray', linestyle=':', linewidth=1.0,
                label=rf'$E_0={E0}$')
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_ylabel(r'Energy', fontsize=13)
    ax1.set_title(rf'DOS Oscillations at $\rho={rho_val}$   '
                  rf'($a={a},\ c={c},\ \Gamma={Gamma}$)', fontsize=13)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, linestyle=':', alpha=0.4)

    # --- Middle panel: DOS at E_F vs 1/B ---
    ax2.plot(inv_B, DOS_at_EF, 'b-', linewidth=1.0)
    ax2.set_xlabel(r'$1/B_z$', fontsize=13)
    ax2.set_ylabel(r'DOS$(E_F,\,B_z)$', fontsize=13)
    ax2.grid(True, linestyle=':', alpha=0.4)

    # --- Bottom panel: FFT of DOS oscillations ---
    # Resample onto uniform 1/B grid for clean FFT
    DOS_arr = np.array(DOS_at_EF)
    valid = ~np.isnan(DOS_arr)
    if np.sum(valid) > 10:
        inv_B_uniform = np.linspace(inv_B[valid].min(), inv_B[valid].max(),
                                    len(inv_B[valid]))
        dos_uniform = np.interp(inv_B_uniform, np.sort(inv_B[valid]),
                                DOS_arr[valid][np.argsort(inv_B[valid])])
        # Subtract mean to remove DC component
        dos_uniform -= np.mean(dos_uniform)
        # Apply Hanning window to reduce spectral leakage
        window = np.hanning(len(dos_uniform))
        dos_windowed = dos_uniform * window
        # FFT
        fft_vals = np.abs(np.fft.rfft(dos_windowed))
        d_inv_B = inv_B_uniform[1] - inv_B_uniform[0]
        freqs = np.fft.rfftfreq(len(dos_windowed), d=d_inv_B)
        # Skip DC (freq=0)
        ax3.plot(freqs[1:], fft_vals[1:], 'darkgreen', linewidth=1.0)
        ax3.set_xlabel(r'Frequency $F$ (in $1/B_z$ units)', fontsize=13)
        ax3.set_ylabel(r'FFT amplitude', fontsize=13)
        ax3.set_xlim(0, freqs[-1] * 0.5)  # show lower half for clarity
        ax3.grid(True, linestyle=':', alpha=0.4)
    else:
        ax3.text(0.5, 0.5, 'Insufficient data for FFT', transform=ax3.transAxes,
                 ha='center', va='center', fontsize=12)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def plot_dos_oscillations_vs_B(a, c, E0, Gamma, Lambda,
                               B_range=(0.005, 0.1), num_B=500,
                               rho_val=0.02, filename='dos_oscillations_vs_B.png'):
    """
    Same as plot_dos_oscillations but with B_z on the x-axis instead of 1/B_z.
    Top panel: E_F(B_z) with LL fan.  Bottom panel: DOS(E_F, B_z).
    """
    B_values = np.linspace(B_range[0], B_range[1], num_B)

    DOS_at_EF = []
    EF_values = []
    for B in B_values:
        try:
            N_max_B = max(1, int(Lambda**2 / (2.0 * B)))
            EF = find_fermi_energy(rho_val, B, a, c, E0, Gamma, Lambda)
            dos = dos_at_energy(EF, B, a, c, E0, Gamma, N_max_B) if not np.isnan(EF) else np.nan
        except Exception:
            EF = np.nan
            dos = np.nan
        DOS_at_EF.append(dos)
        EF_values.append(EF)

    E_flat = E0 + c**2 / a

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True,
                                    gridspec_kw={'height_ratios': [1, 1]})

    # Top panel: Fermi energy + LL spectrum vs B
    EF_arr = np.array(EF_values)
    EF_valid = EF_arr[~np.isnan(EF_arr)]
    if len(EF_valid) > 0:
        y_lo = min(EF_valid.min(), E0) - 0.005
        y_hi = max(EF_valid.max(), E_flat) + 0.01
    else:
        y_lo, y_hi = E0 - 0.01, E_flat + 0.05

    N_max_plot = max(1, int(Lambda**2 / (2.0 * B_range[0])))
    ax1.plot(B_values, [landau_level_energy_LLL(B, a, E0) for B in B_values],
             color='olive', linewidth=0.6, alpha=0.5)
    ax1.axhline(E_flat, color='steelblue', linestyle=':', linewidth=1.0,
                label=rf'$E_\mathrm{{flat}}={E_flat:.4f}$')
    for n in range(1, N_max_plot + 1):
        Ep_LL,  Em_LL  = landau_level_energies_LL(B_values, n, a, c, E0)
        Ep_TR, Em_TR = landau_level_energies_LLTR(B_values, n, a, c, E0)
        for Earr in [Ep_LL, Em_LL, Ep_TR, Em_TR]:
            if np.any((Earr >= y_lo) & (Earr <= y_hi)):
                ax1.plot(B_values, Earr, color='olive', linewidth=0.5, alpha=0.4)

    ax1.plot(B_values, EF_values, 'r-', linewidth=1.2, label=rf'$E_F$  ($\rho={rho_val}$)')
    ax1.axhline(E0, color='gray', linestyle=':', linewidth=1.0,
                label=rf'$E_0={E0}$')
    ax1.set_ylim(y_lo, y_hi)
    ax1.set_ylabel(r'Energy', fontsize=13)
    ax1.set_title(rf'DOS Oscillations vs $B_z$ at $\rho={rho_val}$   '
                  rf'($a={a},\ c={c},\ \Gamma={Gamma}$)', fontsize=13)
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, linestyle=':', alpha=0.4)

    # Bottom panel: DOS at E_F vs B
    ax2.plot(B_values, DOS_at_EF, 'b-', linewidth=1.0)
    ax2.set_xlabel(r'$B_z$', fontsize=13)
    ax2.set_ylabel(r'DOS$(E_F,\,B_z)$', fontsize=13)
    ax2.grid(True, linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {filename}")


def main():
    # Figure output relative to project root (one level up from Codes/)
    _ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    _FIG = os.path.join(_ROOT, 'figures', 'default_a1.115_c0.215_E0_0.0849_Lambda_0.5')
    os.makedirs(_FIG, exist_ok=True)

    # Shared model parameters (matching the Mathematica notebook)
    a_val      = 1.115
    c_val      = 0.215
    E0_val     = 0.0849
    Gamma_val  = 0.001**2  # Landau level broadening (reduced from 0.002^2 to resolve anomalous LLs)
    Lambda_val = 0.5     # momentum cutoff; NMax = Max[1,Floor[Lambda^2/(2*Bz)]] per B value

    # 1. Band Structure
    plot_band_structure(filename=os.path.join(_FIG, 'ideal_flatband_dispersion.png'))

    # 2. Landau Level spectrum
    plot_landau_levels(
        a=a_val, c=c_val, E0=E0_val, B_max=0.1, N_max=20,
        filename=os.path.join(_FIG, 'landau_levels_spectrum.png')
    )

    # 2b. Landau Level spectrum for the optimized Step 2 parameter set
    _FIG_SB = os.path.join(_ROOT, 'figures', 'smallB_opt_a2.5_c0.215_E0_0.0849_Lambda_0.5')
    os.makedirs(_FIG_SB, exist_ok=True)
    plot_landau_levels(
        a=2.5, c=c_val, E0=E0_val, B_max=0.1, N_max=20,
        filename=os.path.join(_FIG_SB, 'landau_levels_spectrum_a2p5.png')
    )

    # 3. DOS vs Energy at fixed B (Figure 1)
    print("\n--- Figure 1: DOS vs Energy at fixed Bz ---")
    plot_dos_vs_energy(
        a=a_val, c=c_val, E0=E0_val,
        Gamma=Gamma_val, Lambda=Lambda_val,
        B_fixed=0.02,
        rho_list=[0.005, 0.01, 0.015, -0.005, -0.01],
        filename=os.path.join(_FIG, 'dos_vs_energy.png')
    )

    # 4. Fermi Energy vs B for several fillings (Figure 2)
    print("\n--- Figure 2: Fermi Energy vs Bz ---")
    plot_fermi_energy_vs_B(
        a=a_val, c=c_val, E0=E0_val,
        Gamma=Gamma_val, Lambda=Lambda_val,
        B_range=(0.001, 0.1), num_B=200,
        rho_list=[0.005, 0.01, 0.015, -0.005, -0.01],
        filename=os.path.join(_FIG, 'fermi_energy_vs_B.png')
    )

    # 5. DOS at Fermi level vs 1/Bz (Figure 3 — dHvA-like oscillations)
    print("\n--- Figure 3: DOS Oscillations vs 1/Bz ---")
    plot_dos_oscillations(
        a=a_val, c=c_val, E0=E0_val,
        Gamma=Gamma_val, Lambda=Lambda_val,
        B_range=(0.005, 0.1), num_B=500,
        rho_val=0.015,
        filename=os.path.join(_FIG, 'dos_oscillations.png')
    )

if __name__ == "__main__":
    main()
