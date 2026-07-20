"""
Step 4.1 — Numerical test of the LK theory for anomalous Landau levels.

Builds finite-temperature fixed-density diagnostics on top of the analytical
LL spectrum of the ideal THF flat band and tests:

    X_p = pi^2 p k_B T [c^2 B - (mu - E_0)^2] / [a B (mu - E_0)^2]

in the bulk regime |mu - E_0| << c sqrt(B).

Important terminology correction:
    The density of states is a spectral quantity and is not temperature
    dependent. The finite-T kernel computed below is d rho / d mu, i.e. a
    compressibility-like response. It is useful as a diagnostic, but it should
    not be used as the final LK thermal-damping observable. The primary Step 4
    observable should be the oscillatory grand potential, fixed-density
    Helmholtz free energy, or magnetization.

Protocol for the current diagnostic path: fix carrier density rho, solve for
mu(B, T) via Brent on the finite-T occupation sum, collect the
compressibility-like response, FFT in 1/B, and compare with analytical thermal
scales.

Entry points (CLI):
    python step4_lk_thermal.py --mode v1        # sum-rule + zero-T check
    python step4_lk_thermal.py --mode main      # diagnostic rho x T x B sweep + figures
    python step4_lk_thermal.py --mode v2        # Onsager + dispersive control
    python step4_lk_thermal.py --mode v3        # robustness (B-window, T-grid, sign)

Plan reference: Implementation and Verification Plan for Step 4.1.md
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, curve_fit
from scipy.signal import savgol_filter

# Make ideal_flatband_model importable when running from this file's directory
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from ideal_flatband_model import (
    landau_level_energies_LL,
    landau_level_energies_LLTR,
    landau_level_energies_LLTRZero,
    landau_level_energy_LLL,
)


# ------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------

@dataclass(frozen=True)
class Params:
    a: float = 1.115
    c: float = 0.215
    E0: float = 0.0849
    Lambda: float = 2.0

    @property
    def E_flat(self) -> float:
        return self.E0 + self.c ** 2 / self.a

    @property
    def B_star(self) -> float:
        return self.c ** 2 / self.a ** 2


DEFAULT = Params()
FIG_ROOT = os.path.abspath(
    os.path.join(_HERE, "..", "figures",
                 "default_a1.115_c0.215_E0_0.0849_Lambda_2.0",
                 "step4_lk_thermodynamics_fixed_density")
)
THEORY_PARAMS = Params(Lambda=1.0)
THEORY_FIG_ROOT = os.path.abspath(
    os.path.join(_HERE, "..", "figures",
                 "default_a1.115_c0.215_E0_0.0849_Lambda_1.0",
                 "step4_lk_theory_tests")
)


# ------------------------------------------------------------------
# 1. Analytical LL spectrum
# ------------------------------------------------------------------

def analytical_LL_spectrum(B: float, p: Params = DEFAULT) -> np.ndarray:
    """Flat array of LL energies.  Includes:
      - LL-sector n=0 (LLL) : E0 + a B
      - LLTR zero mode      : E0 + c^2/a
      - LL+ / LL-  branches for n = 1..N_max
      - LLTR+ / LLTR- branches for n = 1..N_max
    with N_max = floor(Lambda^2 / (2 B)).

    Returns energies sorted ascending.  No band-index metadata is attached
    since the FD sum only cares about energies and total counts.
    """
    N_max = max(1, int(np.floor(p.Lambda ** 2 / (2.0 * B))))
    n_arr = np.arange(1, N_max + 1)

    Ep_LL,   Em_LL   = landau_level_energies_LL  (B, n_arr, p.a, p.c, p.E0)
    Ep_LLTR, Em_LLTR = landau_level_energies_LLTR(B, n_arr, p.a, p.c, p.E0)

    extras = np.array([
        landau_level_energy_LLL(B, p.a, p.E0),
        landau_level_energies_LLTRZero(p.a, p.c, p.E0),
    ])

    energies = np.concatenate([extras, Em_LL, Ep_LL, Em_LLTR, Ep_LLTR])
    return np.sort(energies)


def analytical_LL_spectrum_with_dB(B: float, p: Params = DEFAULT
                                   ) -> tuple[np.ndarray, np.ndarray]:
    """Sorted LL energies and analytical dE/dB for every level."""
    N_max = max(1, int(np.floor(p.Lambda ** 2 / (2.0 * B))))
    n_arr = np.arange(1, N_max + 1, dtype=float)

    Ep_LL, Em_LL = landau_level_energies_LL(B, n_arr, p.a, p.c, p.E0)
    Ep_LLTR, Em_LLTR = landau_level_energies_LLTR(B, n_arr, p.a, p.c, p.E0)

    c2 = p.c ** 2
    U = c2 / p.a + (2.0 * n_arr + 1.0) * p.a * B
    Up = (2.0 * n_arr + 1.0) * p.a
    Delta = np.sqrt(np.maximum(U ** 2 - 4.0 * c2 * B, 1e-300))
    dEp_LL = 0.5 * (Up + (U * Up - 2.0 * c2) / Delta)
    dEm_LL = 0.5 * (Up - (U * Up - 2.0 * c2) / Delta)

    V = c2 / p.a + (2.0 * n_arr - 1.0) * p.a * B
    Vp = (2.0 * n_arr - 1.0) * p.a
    W = np.sqrt(V ** 2 + 4.0 * c2 * B)
    dEp_LLTR = 0.5 * (Vp + (V * Vp + 2.0 * c2) / W)
    dEm_LLTR = 0.5 * (Vp - (V * Vp + 2.0 * c2) / W)

    extras = np.array([
        landau_level_energy_LLL(B, p.a, p.E0),
        landau_level_energies_LLTRZero(p.a, p.c, p.E0),
    ])
    dextras = np.array([p.a, 0.0])

    energies = np.concatenate([extras, Em_LL, Ep_LL, Em_LLTR, Ep_LLTR])
    denergies = np.concatenate([dextras, dEm_LL, dEp_LL, dEm_LLTR, dEp_LLTR])
    order = np.argsort(energies)
    return energies[order], denergies[order]


def E_ref_B(B: float, p: Params = DEFAULT) -> float:
    """Mid-gap reference energy matching the existing codebase convention."""
    return p.E0 + 0.5 * (p.a * B + p.c ** 2 / p.a)


def rho_max(p: Params = DEFAULT) -> float:
    """Half-band carrier density (half-filling = full lower band)."""
    return p.Lambda ** 2 / (4.0 * np.pi)


# ------------------------------------------------------------------
# 2. Finite-T occupation, DOS, and Brent solver
# ------------------------------------------------------------------

def _fermi(x: np.ndarray) -> np.ndarray:
    """Numerically stable Fermi-Dirac.  x = beta (E - mu)."""
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos]  = np.exp(-x[pos]) / (1.0 + np.exp(-x[pos]))
    out[~pos] = 1.0 / (1.0 + np.exp(x[~pos]))
    return out


def total_filling_finite_T(mu: float, B: float, T: float,
                           ll: np.ndarray) -> float:
    """Total filling rho_tot = (B/2pi) sum_n f(E_n - mu; T).

    All LL energies are counted (both bands).  Sign convention relative to
    E_ref is handled by subtracting rho_ref in solve_mu_from_rho.
    """
    if T <= 0.0:
        return (B / (2.0 * np.pi)) * float(np.sum(ll <= mu))
    beta = 1.0 / T
    x = beta * (ll - mu)
    return (B / (2.0 * np.pi)) * float(np.sum(_fermi(x)))


def grand_potential_finite_T(mu: float, B: float, T: float,
                             ll: np.ndarray) -> float:
    """Grand potential per unit area for the discrete LL spectrum.

    omega = -T * D(B) * sum_j log(1 + exp(-(E_j - mu)/T))

    Units use k_B = 1.  At T=0 this becomes
    D(B) * sum_{E_j < mu} (E_j - mu).
    """
    degeneracy = B / (2.0 * np.pi)
    if T <= 0.0:
        occupied = ll < mu
        return degeneracy * float(np.sum(ll[occupied] - mu))
    z = -(ll - mu) / T
    return -T * degeneracy * float(np.sum(np.logaddexp(0.0, z)))


def grand_potential_density_check(mu: float, B: float, T: float,
                                  ll: np.ndarray,
                                  dmu: float = 1e-6) -> tuple[float, float, float]:
    """Finite-difference check of rho = -d omega / d mu."""
    omega_p = grand_potential_finite_T(mu + dmu, B, T, ll)
    omega_m = grand_potential_finite_T(mu - dmu, B, T, ll)
    rho_from_omega = -(omega_p - omega_m) / (2.0 * dmu)
    rho_direct = total_filling_finite_T(mu, B, T, ll)
    return rho_from_omega, rho_direct, abs(rho_from_omega - rho_direct)


def sector_filling_finite_T(mu: float, B: float, T: float,
                            energies: np.ndarray) -> float:
    """Carrier density in one active sector."""
    degeneracy = B / (2.0 * np.pi)
    if T <= 0.0:
        return degeneracy * float(np.sum(energies <= mu))
    x = (energies - mu) / T
    return degeneracy * float(np.sum(_fermi(x)))


def sector_grand_potential_finite_T(mu: float, B: float, T: float,
                                    energies: np.ndarray) -> float:
    """Grand potential for one active carrier sector."""
    degeneracy = B / (2.0 * np.pi)
    if T <= 0.0:
        occupied = energies < mu
        return degeneracy * float(np.sum(energies[occupied] - mu))
    z = -(energies - mu) / T
    return -T * degeneracy * float(np.sum(np.logaddexp(0.0, z)))


def sector_magnetization_finite_T(mu: float, B: float, T: float,
                                  energies: np.ndarray,
                                  denergies_dB: np.ndarray) -> float:
    """Exact discrete carrier-sector magnetization at fixed sector chemical potential."""
    degeneracy = B / (2.0 * np.pi)
    ddegeneracy = 1.0 / (2.0 * np.pi)
    if T <= 0.0:
        occupied = energies < mu
        omega_sum = float(np.sum(energies[occupied] - mu))
        denergy_sum = float(np.sum(denergies_dB[occupied]))
        return -ddegeneracy * omega_sum - degeneracy * denergy_sum
    z = -(energies - mu) / T
    L = np.logaddexp(0.0, z)
    occ = _fermi((energies - mu) / T)
    return T * ddegeneracy * float(np.sum(L)) - degeneracy * float(np.sum(occ * denergies_dB))


def sector_internal_energy_finite_T(mu: float, B: float, T: float,
                                    energies: np.ndarray) -> float:
    """Direct carrier-sector energy U = D sum_j eps_j f(eps_j-mu).

    The energies passed here must be the same sector energies used in the
    sector grand potential: absolute electron energies for electron sectors,
    and positive hole energies E_ref-E_j for the shallow-hole sector.
    """
    degeneracy = B / (2.0 * np.pi)
    if T <= 0.0:
        occupied = energies <= mu
        return degeneracy * float(np.sum(energies[occupied]))
    x = (energies - mu) / T
    return degeneracy * float(np.sum(energies * _fermi(x)))


def sector_density_check(mu: float, B: float, T: float, energies: np.ndarray,
                         dmu: float = 1e-6) -> tuple[float, float, float]:
    """Finite-difference check of n_sector = -d omega_sector / d mu."""
    omega_p = sector_grand_potential_finite_T(mu + dmu, B, T, energies)
    omega_m = sector_grand_potential_finite_T(mu - dmu, B, T, energies)
    rho_from_omega = -(omega_p - omega_m) / (2.0 * dmu)
    rho_direct = sector_filling_finite_T(mu, B, T, energies)
    return rho_from_omega, rho_direct, abs(rho_from_omega - rho_direct)


def solve_mu_for_sector(target: float, B: float, T: float,
                        energies: np.ndarray) -> float:
    """Solve D sum f(E-mu)=target for a finite active sector."""
    degeneracy = B / (2.0 * np.pi)
    capacity = degeneracy * energies.size
    tol = max(1e-12, 1e-10 * max(1.0, abs(target)))
    if target < -tol or target > capacity + tol:
        raise ValueError(
            f"sector target {target} outside [0,{capacity}] at B={B}"
        )
    if energies.size == 0:
        raise ValueError(f"empty active sector at B={B}")
    if target <= tol:
        return float(np.min(energies) - max(50.0 * T, 0.05))
    if target >= capacity - tol:
        return float(np.max(energies) + max(50.0 * T, 0.05))

    pad = max(50.0 * T, 0.05)
    mu_lo = float(np.min(energies) - pad)
    mu_hi = float(np.max(energies) + pad)

    def eq(mu: float) -> float:
        return sector_filling_finite_T(mu, B, T, energies) - target

    for _ in range(8):
        if eq(mu_lo) <= 0 <= eq(mu_hi):
            break
        mu_lo -= pad
        mu_hi += pad
    else:
        raise RuntimeError(
            f"solve_mu_for_sector: bracket failed for target={target}, B={B}, T={T}"
        )
    return brentq(eq, mu_lo, mu_hi, xtol=1e-12, rtol=1e-10, maxiter=200)


def carrier_sector_at_B(rho_signed: float, B: float, p: Params = DEFAULT) -> dict:
    """Return active carrier-sector energies and density target.

    Regimes:
      rho > 0: electrons above E_ref.
      -rho_up <= rho < 0: holes between E0 and E_ref, with energies E_ref-E.
      rho < -rho_up: remaining electrons below E0, target rho'_e.
    """
    ll, dll_dB = analytical_LL_spectrum_with_dB(B, p)
    E_ref = E_ref_B(B, p)
    degeneracy = B / (2.0 * np.pi)

    up_mask = (ll >= p.E0) & (ll < E_ref)
    down_mask = ll < p.E0
    upper_count = float(np.sum(up_mask))
    down_count = float(np.sum(down_mask))
    rho_up = degeneracy * float(np.sum(up_mask))
    rho_down = degeneracy * float(np.sum(down_mask))

    if rho_signed > 0.0:
        energies = ll[ll > E_ref]
        return {
            "regime": "electron",
            "sector": "electron",
            "energies": energies,
            "denergies_dB": dll_dB[ll > E_ref],
            "target": rho_signed,
            "rho_up": rho_up,
            "rho_down": rho_down,
            "drho_prime_e_dB": np.nan,
            "rho_prime_e": np.nan,
            "E_ref": E_ref,
        }

    abs_rho = abs(rho_signed)
    if abs_rho <= rho_up + 1e-12:
        electron_energies = ll[up_mask]
        hole_energies = E_ref - electron_energies
        hole_denergies = -dll_dB[up_mask]
        order = np.argsort(hole_energies)
        return {
            "regime": "shallow_hole",
            "sector": "upper_hole",
            "energies": hole_energies[order],
            "denergies_dB": hole_denergies[order],
            "target": abs_rho,
            "rho_up": rho_up,
            "rho_down": rho_down,
            "drho_prime_e_dB": np.nan,
            "rho_prime_e": np.nan,
            "E_ref": E_ref,
        }

    rho_prime_e = rho_up + rho_down - abs_rho
    drho_prime_e_dB = (upper_count + down_count) / (2.0 * np.pi)
    return {
        "regime": "deep_hole",
        "sector": "lower_electron",
        "energies": ll[down_mask],
        "denergies_dB": dll_dB[down_mask],
        "target": rho_prime_e,
        "rho_up": rho_up,
        "rho_down": rho_down,
        "drho_prime_e_dB": drho_prime_e_dB,
        "rho_prime_e": rho_prime_e,
        "E_ref": E_ref,
    }


def zero_temperature_sector_thermo_vs_B(rho_signed: float, B_grid: np.ndarray,
                                        p: Params = DEFAULT
                                        ) -> dict[str, np.ndarray]:
    """Zero-temperature sector filling with possible partial LL occupation.

    This is the direct T=0 counting analogue of thermodynamics_sector_vs_B.
    It fills the active carrier-sector levels from low to high energy.  When
    the requested density falls inside a degenerate LL, the last LL is counted
    with fractional occupation, so the fixed density is matched without using
    artificial thermal smearing.
    """
    fermi_energy = np.empty(B_grid.size)
    mu_sector = np.empty(B_grid.size)
    internal_energy = np.empty(B_grid.size)
    rho_target = np.empty(B_grid.size)
    regimes: list[str] = []
    sectors: list[str] = []

    for i, B in enumerate(B_grid):
        info = carrier_sector_at_B(rho_signed, float(B), p)
        energies = np.sort(np.asarray(info["energies"], dtype=float))
        target = float(info["target"])
        degeneracy = float(B) / (2.0 * np.pi)
        capacity = degeneracy * energies.size
        tol = max(1e-12, 1e-10 * max(1.0, abs(target)))

        if target < -tol or target > capacity + tol:
            raise ValueError(
                f"zero-T sector target {target} outside [0,{capacity}] at B={B}"
            )

        if target <= tol:
            filled_count = 0
            frac = 0.0
            mu = float(energies[0])
        elif target >= capacity - tol:
            filled_count = energies.size
            frac = 0.0
            mu = float(energies[-1])
        else:
            q = target / degeneracy
            filled_count = int(np.floor(q))
            frac = float(q - filled_count)
            if frac <= 1e-10 and filled_count > 0:
                frac = 0.0
                mu = float(energies[filled_count - 1])
            else:
                mu = float(energies[min(filled_count, energies.size - 1)])

        energy_sum = float(np.sum(energies[:filled_count]))
        if frac > 0.0 and filled_count < energies.size:
            energy_sum += frac * float(energies[filled_count])

        mu_sector[i] = mu
        if info["regime"] == "shallow_hole":
            fermi_energy[i] = float(info["E_ref"] - mu)
        else:
            fermi_energy[i] = mu
        internal_energy[i] = degeneracy * energy_sum
        rho_target[i] = target
        regimes.append(str(info["regime"]))
        sectors.append(str(info["sector"]))

    return {
        "fermi_energy": fermi_energy,
        "mu": mu_sector,
        "internal_energy": internal_energy,
        "rho_target": rho_target,
        "regime": np.asarray(regimes, dtype=object),
        "carrier_sector": np.asarray(sectors, dtype=object),
    }


def thermodynamics_sector_vs_B(rho_signed: float, B_grid: np.ndarray, T: float,
                               p: Params = DEFAULT,
                               mu_fixed_sector: float | None = None
                               ) -> dict[str, np.ndarray]:
    """Carrier-resolved thermodynamic sweep using the three-regime convention."""
    n = B_grid.size
    mu_sector = np.empty(n)
    fermi_energy = np.empty(n)
    chi = np.empty(n)
    omega = np.empty(n)
    free_energy = np.empty(n)
    internal_energy = np.empty(n)
    magnetization = np.empty(n)
    solver_err = np.empty(n)
    omega_identity_err = np.empty(n)
    rho_target = np.empty(n)
    rho_up = np.empty(n)
    rho_down = np.empty(n)
    rho_prime_e = np.empty(n)
    drho_prime_e_dB = np.empty(n)
    E_ref_arr = np.empty(n)
    omega_fixed = np.empty(n) if mu_fixed_sector is not None else None
    regimes: list[str] = []
    sectors: list[str] = []

    for i, B in enumerate(B_grid):
        info = carrier_sector_at_B(rho_signed, float(B), p)
        energies = info["energies"]
        denergies_dB = info["denergies_dB"]
        target = float(info["target"])
        if target < -1e-10:
            raise ValueError(
                f"rho={rho_signed} exceeds lower-sector capacity at B={B}: "
                f"rho_up={info['rho_up']}, rho_down={info['rho_down']}"
            )

        mu = solve_mu_for_sector(target, float(B), T, energies)
        rho_check = sector_filling_finite_T(mu, float(B), T, energies)
        omega_i = sector_grand_potential_finite_T(mu, float(B), T, energies)
        energy_i = sector_internal_energy_finite_T(mu, float(B), T, energies)
        magnetization_i = sector_magnetization_finite_T(
            mu, float(B), T, energies, denergies_dB
        )
        if info["regime"] == "deep_hole":
            magnetization_i -= mu * float(info["drho_prime_e_dB"])
        _, _, identity_err = sector_density_check(mu, float(B), T, energies)

        mu_sector[i] = mu
        if info["regime"] == "shallow_hole":
            fermi_energy[i] = float(info["E_ref"] - mu)
        else:
            fermi_energy[i] = mu
        chi[i] = dos_at_mu_finite_T(mu, float(B), T, energies)
        omega[i] = omega_i
        free_energy[i] = omega_i + mu * target
        internal_energy[i] = energy_i
        magnetization[i] = magnetization_i
        solver_err[i] = abs(rho_check - target)
        omega_identity_err[i] = identity_err
        rho_target[i] = target
        rho_up[i] = float(info["rho_up"])
        rho_down[i] = float(info["rho_down"])
        rho_prime_e[i] = float(info["rho_prime_e"])
        drho_prime_e_dB[i] = float(info["drho_prime_e_dB"])
        E_ref_arr[i] = float(info["E_ref"])
        regimes.append(str(info["regime"]))
        sectors.append(str(info["sector"]))
        if omega_fixed is not None:
            omega_fixed[i] = sector_grand_potential_finite_T(
                mu_fixed_sector, float(B), T, energies
            )

    out = {
        "mu": mu_sector,
        "fermi_energy": fermi_energy,
        "chi": chi,
        "omega": omega,
        "free_energy": free_energy,
        "internal_energy": internal_energy,
        "magnetization": magnetization,
        "rho_target": rho_target,
        "rho_up": rho_up,
        "rho_down": rho_down,
        "rho_prime_e": rho_prime_e,
        "drho_prime_e_dB": drho_prime_e_dB,
        "E_ref": E_ref_arr,
        "solver_err": solver_err,
        "omega_identity_err": omega_identity_err,
        "regime": np.asarray(regimes, dtype=object),
        "carrier_sector": np.asarray(sectors, dtype=object),
    }
    if omega_fixed is not None:
        out["omega_fixed_mu"] = omega_fixed
    return out


def rho_ref_finite_T(B: float, T: float, p: Params = DEFAULT) -> float:
    """Filling at mu = E_ref — analogous to the zero-T rho_ref but
    computed with the same finite-T kernel used for the solver."""
    ll = analytical_LL_spectrum(B, p)
    return total_filling_finite_T(E_ref_B(B, p), B, T, ll)


def dos_at_mu_finite_T(mu: float, B: float, T: float,
                       ll: np.ndarray) -> float:
    """Compressibility-like response d rho / d mu.

    chi_T(mu, B) = (B/2pi) sum_n -df/dE|_{E_n - mu, T}
    with -df/dE = beta / [4 cosh^2(beta (E - mu) / 2)].

    This is not a temperature-dependent density of states.  The spectral DOS
    is set by the LL energies and a separate broadening convention.
    """
    if T <= 0.0:
        raise ValueError("Finite-T DOS requires T > 0. Use a small positive T.")
    beta = 1.0 / T
    x = 0.5 * beta * (ll - mu)
    # clip before cosh to avoid overflow; exp(350) ~ 1e152 is safe but cosh^2 hits inf at ~710
    xc = np.clip(x, -350.0, 350.0)
    kernel = beta / (4.0 * np.cosh(xc) ** 2)
    return (B / (2.0 * np.pi)) * float(np.sum(kernel))


def solve_mu_from_rho(rho_signed: float, B: float, T: float,
                      p: Params = DEFAULT,
                      ll: np.ndarray | None = None,
                      rho_ref: float | None = None) -> float:
    """Brent root-find for mu such that
        total_filling_finite_T(mu, B, T) - rho_ref(B, T) = rho_signed.

    rho_signed follows the existing codebase's convention:
        rho_signed > 0  ->  mu above E_ref (electron doping)
        rho_signed < 0  ->  mu below E_ref (hole / anomalous doping)
    """
    if ll is None:
        ll = analytical_LL_spectrum(B, p)
    if rho_ref is None:
        rho_ref = rho_ref_finite_T(B, T, p)
    target = rho_ref + rho_signed

    # Bracket.  At mu = ll.min() - pad, filling ~ 0; at mu = ll.max() + pad, filling ~ full.
    pad = max(20.0 * T, 0.05)
    mu_lo = float(ll.min()) - pad
    mu_hi = float(ll.max()) + pad

    def eq(mu):
        return total_filling_finite_T(mu, B, T, ll) - target

    # Sanity check on bracket; expand if needed
    for _ in range(8):
        if eq(mu_lo) < 0 < eq(mu_hi):
            break
        mu_lo -= pad
        mu_hi += pad
    else:
        raise RuntimeError(
            f"solve_mu_from_rho: bracket failed for rho={rho_signed}, B={B}, T={T}"
        )
    return brentq(eq, mu_lo, mu_hi, xtol=1e-12, rtol=1e-10, maxiter=200)


# ------------------------------------------------------------------
# 3. DOS sweep over B-grid and FFT amplitude extraction
# ------------------------------------------------------------------

def dos_vs_B(rho_signed: float, B_grid: np.ndarray, T: float,
             p: Params = DEFAULT) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mu_array, chi_T_array) of length len(B_grid).

    chi_T is d rho / d mu from the Fermi derivative kernel.  It is kept for
    diagnostics and legacy plots, but should not be called a temperature-
    dependent DOS in the Step 4 LK interpretation.
    """
    mus = np.empty(B_grid.size)
    doses = np.empty(B_grid.size)
    for i, B in enumerate(B_grid):
        ll = analytical_LL_spectrum(B, p)
        rho_ref = total_filling_finite_T(E_ref_B(B, p), B, T, ll)
        mu = solve_mu_from_rho(rho_signed, B, T, p, ll=ll, rho_ref=rho_ref)
        mus[i] = mu
        doses[i] = dos_at_mu_finite_T(mu, B, T, ll)
    return mus, doses


def thermodynamics_vs_B(rho_signed: float, B_grid: np.ndarray, T: float,
                        p: Params = DEFAULT,
                        mu_fixed: float | None = None
                        ) -> dict[str, np.ndarray]:
    """Fixed-density thermodynamic sweep over B.

    The codebase's input rho is a signed carrier density relative to the
    finite-T reference filling at E_ref(B).  For the Legendre transform we use
    the absolute target filling used by the solver:

        n_target(B,T) = rho_ref(B,T) + rho_signed.

    This convention is saved explicitly because the reference filling depends
    on B through E_ref(B).
    """
    mus = np.empty(B_grid.size)
    chi = np.empty(B_grid.size)
    omega = np.empty(B_grid.size)
    free_energy = np.empty(B_grid.size)
    rho_ref_arr = np.empty(B_grid.size)
    n_target = np.empty(B_grid.size)
    solver_err = np.empty(B_grid.size)
    omega_identity_err = np.empty(B_grid.size)
    omega_fixed = np.empty(B_grid.size) if mu_fixed is not None else None

    for i, B in enumerate(B_grid):
        ll = analytical_LL_spectrum(B, p)
        rho_ref = total_filling_finite_T(E_ref_B(B, p), B, T, ll)
        target = rho_ref + rho_signed
        mu = solve_mu_from_rho(rho_signed, B, T, p, ll=ll, rho_ref=rho_ref)
        rho_check = total_filling_finite_T(mu, B, T, ll)
        omega_i = grand_potential_finite_T(mu, B, T, ll)
        _, _, identity_err = grand_potential_density_check(mu, B, T, ll)

        mus[i] = mu
        chi[i] = dos_at_mu_finite_T(mu, B, T, ll)
        omega[i] = omega_i
        free_energy[i] = omega_i + mu * target
        rho_ref_arr[i] = rho_ref
        n_target[i] = target
        solver_err[i] = abs(rho_check - target)
        omega_identity_err[i] = identity_err
        if omega_fixed is not None:
            omega_fixed[i] = grand_potential_finite_T(mu_fixed, B, T, ll)

    out = {
        "mu": mus,
        "chi": chi,
        "omega": omega,
        "free_energy": free_energy,
        "rho_ref": rho_ref_arr,
        "n_target": n_target,
        "solver_err": solver_err,
        "omega_identity_err": omega_identity_err,
    }
    if omega_fixed is not None:
        out["omega_fixed_mu"] = omega_fixed
    return out


def fft_amplitude(B_grid: np.ndarray, signal: np.ndarray,
                  F_target: float, detrend_deg: int = 3,
                  zero_pad_factor: int = 8
                  ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Detrend signal(1/B), resample to uniform 1/B, Hann-window, FFT.

    Returns (amp_at_F_target, freqs, amps_full_spectrum).
    Frequencies are in units where B has dimension 1 (i.e., conjugate to 1/B).
    """
    inv_B = 1.0 / B_grid
    order = np.argsort(inv_B)
    inv_B, sig = inv_B[order], signal[order]

    coeffs = np.polyfit(inv_B, sig, detrend_deg)
    detrended = sig - np.polyval(coeffs, inv_B)

    inv_B_u = np.linspace(inv_B.min(), inv_B.max(), inv_B.size)
    sig_u = np.interp(inv_B_u, inv_B, detrended)

    window = np.hanning(sig_u.size)
    n_fft = int(sig_u.size * max(1, zero_pad_factor))
    spec = np.fft.rfft(sig_u * window, n=n_fft)
    d = inv_B_u[1] - inv_B_u[0]
    freqs = np.fft.rfftfreq(n_fft, d=d)
    amps = np.abs(spec) * 2.0 / sig_u.size

    if F_target < freqs[0] or F_target > freqs[-1]:
        amp = 0.0
    else:
        amp = float(np.interp(F_target, freqs, amps))
    return amp, freqs, amps


# ------------------------------------------------------------------
# 4. LK thermal damping and m*_eff fit
# ------------------------------------------------------------------

def lk_damping(X: np.ndarray | float) -> np.ndarray | float:
    """X / sinh(X), robust at small X (-> 1 - X^2/6)."""
    X = np.asarray(X, dtype=float)
    out = np.ones_like(X)
    small = np.abs(X) < 1e-6
    out[small] = 1.0 - X[small] ** 2 / 6.0
    big = ~small
    out[big] = X[big] / np.sinh(X[big])
    return out if out.ndim else float(out)


def fit_meff_lk(T_values: np.ndarray, amp_ratio: np.ndarray,
                B_center: float) -> Tuple[float, float]:
    """Fit A(T)/A(T_min) = X/sinh X with X = 2 pi^2 T m_eff / B_center.
    Returns (m_eff, fit_rmse).  k_B = hbar = 1; m_e = 1.
    """
    def model(T, m_eff):
        X = 2.0 * np.pi ** 2 * T * m_eff / B_center
        return lk_damping(X)

    p0 = [0.5]
    try:
        popt, _ = curve_fit(model, T_values, amp_ratio, p0=p0, maxfev=10000)
        m_eff = float(popt[0])
    except Exception:
        m_eff = float("nan")
    resid = amp_ratio - model(T_values, m_eff)
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    return m_eff, rmse


def predicted_Xp(T: float, B: float, delta_mu: float, p: Params = DEFAULT,
                 p_harmonic: int = 1) -> float:
    """Plan's Eq.: X_p = pi^2 p k_B T [c^2 B - delta_mu^2] / [a B delta_mu^2]."""
    num = (p.c ** 2) * B - delta_mu ** 2
    den = p.a * B * delta_mu ** 2
    return np.pi ** 2 * p_harmonic * T * num / den


def predicted_meff_slope(delta_mu: float, p: Params = DEFAULT) -> float:
    """Slope of apparent m*_eff(B) in the bulk regime."""
    return p.c ** 2 / (2.0 * p.a * delta_mu ** 2)


def smooth_background(y: np.ndarray, window_frac: float = 0.19,
                      polyorder: int = 3) -> np.ndarray:
    """Smooth y on the sampled grid with a Savitzky-Golay background."""
    y = np.asarray(y, dtype=float)
    if y.size < 7:
        return y.copy()
    win = max(7, int(np.ceil(window_frac * y.size)))
    if win % 2 == 0:
        win += 1
    win = min(win, y.size if y.size % 2 else y.size - 1)
    if win <= polyorder + 2:
        return y.copy()
    return savgol_filter(y, win, polyorder, mode="interp")


def classify_lower_branch(delta: np.ndarray | float) -> np.ndarray | str:
    """Classify anomalous lower branch from delta = mu_bar - E0."""
    arr = np.asarray(delta)
    labels = np.where(arr > 0.0, "LL-", np.where(arr < 0.0, "LLTR-", "cross"))
    return str(labels.item()) if labels.ndim == 0 else labels


def anomalous_Xp(T: float, B: np.ndarray, mu_bar: np.ndarray,
                 p: Params = DEFAULT, harmonic: int = 1) -> np.ndarray:
    """Branch-aware anomalous lower-branch thermal argument."""
    B = np.asarray(B, dtype=float)
    delta = np.asarray(mu_bar, dtype=float) - p.E0
    den = p.a * B * delta ** 2
    X = np.full_like(B, np.nan, dtype=float)
    ll = delta > 0.0
    lltr = delta < 0.0
    X[ll] = (np.pi ** 2 * harmonic * T *
             (p.c ** 2 * B[ll] - delta[ll] ** 2) / den[ll])
    X[lltr] = (np.pi ** 2 * harmonic * T *
               (p.c ** 2 * B[lltr] + delta[lltr] ** 2) / den[lltr])
    return X


def dispersive_Xp(T: float, B: np.ndarray, mu_bar: np.ndarray,
                  branch: str = "LL+", p: Params = DEFAULT,
                  harmonic: int = 1) -> np.ndarray:
    """Exact upper-branch dispersive thermal argument."""
    B = np.asarray(B, dtype=float)
    delta = np.asarray(mu_bar, dtype=float) - p.E0
    den = p.a * B * delta ** 2
    if branch == "LL+":
        num = delta ** 2 - p.c ** 2 * B
    elif branch == "LLTR+":
        num = delta ** 2 + p.c ** 2 * B
    else:
        raise ValueError(f"Unknown dispersive branch {branch}")
    return np.pi ** 2 * harmonic * T * num / den


def standard_Xp(T: float, B: np.ndarray, p: Params = DEFAULT,
                harmonic: int = 1) -> np.ndarray:
    """Parabolic upper-branch limit with m* = 1/(2a)."""
    B = np.asarray(B, dtype=float)
    return np.pi ** 2 * harmonic * T / (p.a * B)


def apparent_mass_theory(B: np.ndarray | float, mu_bar: np.ndarray | float,
                         branch: str, p: Params = DEFAULT) -> np.ndarray | float:
    """Branch-resolved apparent mass implied by the local LK X argument."""
    B_arr = np.asarray(B, dtype=float)
    delta = np.asarray(mu_bar, dtype=float) - p.E0
    den = 2.0 * p.a * delta ** 2
    if branch == "LL+":
        val = (delta ** 2 - p.c ** 2 * B_arr) / den
    elif branch == "LLTR+":
        val = (delta ** 2 + p.c ** 2 * B_arr) / den
    elif branch == "LL-":
        val = (p.c ** 2 * B_arr - delta ** 2) / den
    elif branch == "LLTR-":
        val = (p.c ** 2 * B_arr + delta ** 2) / den
    elif branch == "standard":
        val = np.full_like(B_arr, 1.0 / (2.0 * p.a), dtype=float)
    else:
        raise ValueError(f"Unknown branch {branch}")
    return float(val) if np.ndim(val) == 0 else val


def peak_near_target(freqs: np.ndarray, amps: np.ndarray, F_target: float,
                     half_width_bins: int = 1) -> tuple[float, float, float]:
    """Return (F_peak, amp_peak, delta_F) from a local FFT window."""
    if freqs.size < 2:
        return float("nan"), float("nan"), float("nan")
    delta_F = float(freqs[1] - freqs[0])
    lo = max(0.0, F_target - half_width_bins * delta_F)
    hi = F_target + half_width_bins * delta_F
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        idx = int(np.argmin(np.abs(freqs - F_target)))
    else:
        local = np.where(mask)[0]
        idx = int(local[np.argmax(amps[local])])
    return float(freqs[idx]), float(amps[idx]), delta_F


def normalized_prediction(T_grid: np.ndarray, B_grid: np.ndarray,
                          mu_bar: np.ndarray, sector: str,
                          p: Params = DEFAULT) -> np.ndarray:
    """Window-averaged LK prediction normalized to T_grid[0]."""
    vals = []
    for T in T_grid:
        if sector == "anomalous":
            X = anomalous_Xp(T, B_grid, mu_bar, p)
        elif sector == "dispersive":
            X = dispersive_Xp(T, B_grid, mu_bar, "LL+", p)
            bad = ~np.isfinite(X) | (X <= 0)
            if np.any(bad):
                X = np.where(bad, standard_Xp(T, B_grid, p), X)
        elif sector == "standard":
            X = standard_Xp(T, B_grid, p)
        else:
            raise ValueError(sector)
        vals.append(float(np.nanmedian(lk_damping(np.abs(X)))))
    vals = np.asarray(vals)
    return vals / vals[0] if vals[0] != 0 else vals


def branch_mode(labels: np.ndarray) -> str:
    labels = np.asarray(labels)
    vals, counts = np.unique(labels, return_counts=True)
    return str(vals[np.argmax(counts)])


# ==================================================================
# V1 correctness checks
# ==================================================================

def v1_sum_rule(B: float = 0.03, T: float = 1e-3,
                p: Params = DEFAULT) -> dict:
    """Integrate DOS_T(mu) analytically over all mu: should equal total
    number of LL states times B/(2pi) (each FD kernel integrates to 1).
    """
    ll = analytical_LL_spectrum(B, p)
    # integral of -df/dE dE = 1 per level, summed over levels
    # dos_T = (B/2pi) sum_n (-df/dE)_n, so integral dos_T dmu = (B/2pi) * N_states
    # Verify numerically: sample mu over wide range and trapezoid-integrate.
    mu_grid = np.linspace(ll.min() - 30 * T, ll.max() + 30 * T, 20000)
    dos_vals = np.array([dos_at_mu_finite_T(mu, B, T, ll) for mu in mu_grid])
    integral = np.trapz(dos_vals, mu_grid)
    expected = (B / (2.0 * np.pi)) * ll.size
    return {
        "B": B, "T": T, "N_states": ll.size,
        "integral": integral, "expected": expected,
        "rel_err": abs(integral - expected) / expected,
    }


def v1_zero_T_limit(B: float = 0.03, rho_signed: float = -0.002,
                    p: Params = DEFAULT) -> dict:
    """At very small T, solve_mu_from_rho should reproduce the step-function
    filling: mu sits between the last fully-occupied LL and the first empty
    one (modulo partial filling of one level = tiny fractional offset).
    Cross-check the inverse: rho computed from the solved mu matches target.
    """
    T = 1e-7
    ll = analytical_LL_spectrum(B, p)
    rho_ref = total_filling_finite_T(E_ref_B(B, p), B, T, ll)
    mu = solve_mu_from_rho(rho_signed, B, T, p, ll=ll, rho_ref=rho_ref)
    rho_check = total_filling_finite_T(mu, B, T, ll) - rho_ref
    return {
        "B": B, "rho_signed": rho_signed, "T": T,
        "mu": mu, "rho_check": rho_check,
        "abs_err": abs(rho_check - rho_signed),
    }


def v1_self_consistency(rho_signed: float = -0.002,
                        T: float = 1e-3, B_grid: np.ndarray | None = None,
                        p: Params = DEFAULT) -> dict:
    """After solving for mu(B, T), rho at that mu equals the input rho_signed."""
    if B_grid is None:
        B_grid = np.array([0.01, 0.03, 0.05, 0.07, 0.1])
    errs = []
    for B in B_grid:
        ll = analytical_LL_spectrum(B, p)
        rho_ref = total_filling_finite_T(E_ref_B(B, p), B, T, ll)
        mu = solve_mu_from_rho(rho_signed, B, T, p, ll=ll, rho_ref=rho_ref)
        rho_check = total_filling_finite_T(mu, B, T, ll) - rho_ref
        errs.append(abs(rho_check - rho_signed))
    return {
        "rho_signed": rho_signed, "T": T,
        "max_abs_err": float(max(errs)),
        "errs": errs,
    }


def run_v1(p: Params = DEFAULT) -> int:
    print("=" * 72)
    print("V1. Code-correctness checks")
    print("=" * 72)

    # 1. Sum rule
    print("\n[1] DOS_T sum rule (integral over mu vs total LL state count):")
    for B, T in [(0.02, 1e-3), (0.05, 5e-4), (0.1, 2e-3)]:
        r = v1_sum_rule(B=B, T=T, p=p)
        ok = "OK" if r["rel_err"] < 1e-3 else "FAIL"
        print(f"   B={B:.3f}, T={T:.1e}: N_states={r['N_states']:>4d}, "
              f"integral={r['integral']:.6e}, expected={r['expected']:.6e}, "
              f"rel_err={r['rel_err']:.2e}  [{ok}]")

    # 2. Zero-T limit
    print("\n[2] Zero-T Brent solver recovers target filling:")
    for B in [0.01, 0.03, 0.05, 0.1]:
        for rho in [-0.0005, -0.002, -0.004, 0.05]:
            r = v1_zero_T_limit(B=B, rho_signed=rho, p=p)
            ok = "OK" if r["abs_err"] < 1e-8 else "FAIL"
            print(f"   B={B:.3f}, rho={rho:+.4f}: mu={r['mu']:.5f}, "
                  f"rho_check={r['rho_check']:+.6e}, abs_err={r['abs_err']:.1e}  [{ok}]")

    # 3. Self-consistency at finite T
    print("\n[3] Finite-T self-consistency (rho round-trip):")
    for rho in [-0.0005, -0.002, -0.004, 0.05]:
        for T in [5e-4, 2e-3]:
            r = v1_self_consistency(rho_signed=rho, T=T, p=p)
            ok = "OK" if r["max_abs_err"] < 1e-8 else "FAIL"
            print(f"   rho={rho:+.4f}, T={T:.1e}: max_abs_err={r['max_abs_err']:.1e}  [{ok}]")

    return 0


# ==================================================================
# Main sweep
# ==================================================================

RHO_LIST = [-0.0005, -0.002, -0.003, -0.004, 0.05]

def luttinger_mu_bar(rho_signed: float, p: Params = DEFAULT) -> float:
    """Approximate mu bar from Luttinger relation: mu_bar = E_flat - 4 pi a |rho|
    (anomalous side) or above E_flat for electron doping."""
    if rho_signed < 0:
        return p.E_flat - 4.0 * np.pi * p.a * abs(rho_signed)
    # dispersive band: F = 2 pi rho => E_F = E_flat + 2 a rho  (using F = (E_F-E_flat)/(2 a))
    # Wait — for dispersive band with E = E_flat + a k^2, Onsager gives F = A_FS/(2 pi)
    # Since A_FS = pi k_F^2 and rho = k_F^2/(4 pi) (per 2D band, Chern 0),
    # F = k_F^2/2 = 2 pi rho. And E_F - E_flat = a k_F^2 = 4 pi a rho.
    return p.E_flat + 4.0 * np.pi * p.a * rho_signed


def build_T_grid(rho_signed: float = -0.002, B_center: float = 0.03,
                 n_T: int = 8, Xp_range: Tuple[float, float] = (0.5, 5.0),
                 p: Params = DEFAULT) -> np.ndarray:
    """Log-spaced T such that predicted X_p spans [X_lo, X_hi] at central
    (rho, B)."""
    delta_mu = luttinger_mu_bar(rho_signed, p) - p.E0
    # X_p / T = pi^2 (c^2 B - delta_mu^2) / (a B delta_mu^2)
    slope = np.pi ** 2 * (p.c ** 2 * B_center - delta_mu ** 2) / (p.a * B_center * delta_mu ** 2)
    if slope <= 0:
        # edge regime — fall back to a sensible default
        slope = np.pi ** 2 * p.c ** 2 / (p.a * delta_mu ** 2)
    T_lo = Xp_range[0] / abs(slope)
    T_hi = Xp_range[1] / abs(slope)
    return np.logspace(np.log10(T_lo), np.log10(T_hi), n_T)


def _ensure_figdir():
    os.makedirs(FIG_ROOT, exist_ok=True)


def _B_grid_uniform_in_invB(B_lo=0.01, B_hi=0.1, n=400) -> np.ndarray:
    invB = np.linspace(1.0 / B_hi, 1.0 / B_lo, n)
    return 1.0 / invB[::-1]


def run_main(p: Params = DEFAULT) -> int:
    _ensure_figdir()
    print("=" * 72)
    print("Main rho x T x B sweep")
    print("=" * 72)

    B_grid = _B_grid_uniform_in_invB(0.01, 0.1, 400)
    print(f"B grid: {B_grid.size} points uniform in 1/B, "
          f"B in [{B_grid.min():.4f}, {B_grid.max():.4f}]")

    # Build T grid from central anomalous rho
    T_grid_anom = build_T_grid(-0.002, B_center=0.03, n_T=8, p=p)
    # Dispersive control needs a different T grid (smaller X_p slope -> larger T)
    # For dispersive: X_p = pi^2 T / (a B)  => T for X_p=5 at B=0.03 : T = 5 a B / pi^2
    T_grid_disp = np.logspace(
        np.log10(0.5 * p.a * 0.03 / np.pi ** 2),
        np.log10(5.0 * p.a * 0.03 / np.pi ** 2),
        8)
    print(f"T grid (anomalous): {T_grid_anom}")
    print(f"T grid (dispersive): {T_grid_disp}")

    metrics_rows = []

    # Panel 1: DOS vs 1/B at several T, one axis per rho
    fig1, axes1 = plt.subplots(len(RHO_LIST), 1,
                                figsize=(9, 2.6 * len(RHO_LIST)), sharex=True)
    fig2, axes2 = plt.subplots(1, len(RHO_LIST),
                                figsize=(4.0 * len(RHO_LIST), 4.0), sharey=False)
    fig3, axes3 = plt.subplots(1, 1, figsize=(8, 5.5))
    fig5, axes5 = plt.subplots(1, len(RHO_LIST),
                                figsize=(4.0 * len(RHO_LIST), 4.0))

    meff_vs_B_storage = {}

    for idx, rho in enumerate(RHO_LIST):
        is_disp = (rho > 0)
        T_grid = T_grid_disp if is_disp else T_grid_anom
        delta_mu_est = luttinger_mu_bar(rho, p) - p.E0
        F_target = 2.0 * np.pi * abs(rho)

        print(f"\n--- rho = {rho:+.4f}  (delta_mu_est = {delta_mu_est:+.4f}) ---")
        print(f"    predicted F = 2 pi |rho| = {F_target:.5f}")
        print(f"    FFT bin width dF ~ 1/{1.0/B_grid.min()-1.0/B_grid.max():.1f} "
              f"= {1.0/(1.0/B_grid.min()-1.0/B_grid.max()):.5f}")

        # DOS(B) for each T
        all_dos = []
        all_mus = []
        amps = np.zeros(T_grid.size)
        for j, T in enumerate(T_grid):
            mus, dos = dos_vs_B(rho, B_grid, T, p)
            all_dos.append(dos)
            all_mus.append(mus)
            amp, freqs, spectrum = fft_amplitude(B_grid, dos, F_target)
            amps[j] = amp
            # Refine amp: take the max of spectrum in a small window around F_target
            # to handle slight drift
            if F_target > 0 and freqs.size > 2:
                bin_w = freqs[1] - freqs[0]
                win_mask = (freqs >= F_target - 3 * bin_w) & (freqs <= F_target + 3 * bin_w)
                if win_mask.any():
                    amps[j] = max(amp, float(spectrum[win_mask].max()))

        # mu_bar: average of the T-min solution
        mu_bar = float(np.mean(all_mus[0]))
        delta_mu_bar = mu_bar - p.E0

        # Panel 1: DOS vs 1/B at several T (thin for clarity, only 3 T values)
        ax = axes1[idx]
        T_show = [T_grid[0], T_grid[len(T_grid)//2], T_grid[-1]]
        for Tv in T_show:
            kk = int(np.argmin(np.abs(T_grid - Tv)))
            ax.plot(1.0 / B_grid, all_dos[kk],
                    lw=0.9, label=f"T={Tv:.2e}")
        ax.set_ylabel(r"$\mathrm{DOS}_T$")
        ax.set_title(rf"$\rho={rho:+.4f}$, $\bar\mu - E_0 = {delta_mu_bar:+.4f}$")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        if idx == len(RHO_LIST) - 1:
            ax.set_xlabel(r"$1/B$")

        # Panel 2: A(T)/A(T_min) vs T with predicted overlay
        ax2 = axes2[idx]
        amp_ratio = amps / amps[0]
        ax2.plot(T_grid, amp_ratio, "o-", label="numerical")

        # Predicted X/sinh X with mu_bar
        if is_disp:
            m_pred = 1.0 / (2.0 * p.a)
            T_curve = np.logspace(np.log10(T_grid[0] * 0.5),
                                   np.log10(T_grid[-1] * 2.0), 200)
            X_pred = 2.0 * np.pi ** 2 * T_curve * m_pred / 0.03   # B_center = 0.03
            ax2.plot(T_curve, lk_damping(X_pred), "k--",
                     label=rf"std-LK $m^*=1/(2a)$")
        else:
            T_curve = np.logspace(np.log10(T_grid[0] * 0.5),
                                   np.log10(T_grid[-1] * 2.0), 200)
            X_pred = np.array([predicted_Xp(Tv, 0.03, delta_mu_bar, p) for Tv in T_curve])
            ax2.plot(T_curve, lk_damping(X_pred), "k--",
                     label=rf"derived $X_p$")
        ax2.set_xscale("log")
        ax2.set_xlabel("T")
        ax2.set_ylabel(r"$A(T)/A(T_{\min})$")
        ax2.set_title(rf"$\rho={rho:+.4f}$")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Panel 3 (later): m*_eff(B) from sliding B-windows
        # Slide window of width 1/3 of the full 1/B range; step 1/12
        invB_all = 1.0 / B_grid
        invB_min, invB_max = invB_all.min(), invB_all.max()
        span = invB_max - invB_min
        win_w = span / 3.0
        win_centers_inv = np.linspace(invB_min + win_w / 2,
                                       invB_max - win_w / 2, 10)
        meff_list = []
        Bcenter_list = []
        for wc in win_centers_inv:
            mask = (invB_all >= wc - win_w / 2) & (invB_all <= wc + win_w / 2)
            if mask.sum() < 20:
                continue
            Bc = 1.0 / wc
            # Recompute amp for each T within this window
            amps_win = np.zeros(T_grid.size)
            for j, T in enumerate(T_grid):
                a_j, _, _ = fft_amplitude(B_grid[mask], all_dos[j][mask], F_target)
                amps_win[j] = a_j
            if amps_win[0] <= 0:
                continue
            ratio = amps_win / amps_win[0]
            m_eff, rmse = fit_meff_lk(T_grid, ratio, Bc)
            if np.isfinite(m_eff):
                meff_list.append(m_eff)
                Bcenter_list.append(Bc)
                metrics_rows.append(dict(
                    rho=rho, B_center=Bc, T_min=T_grid[0], T_max=T_grid[-1],
                    m_eff_fit=m_eff, rmse=rmse,
                    F_target=F_target, delta_mu_bar=delta_mu_bar,
                    predicted_slope=predicted_meff_slope(delta_mu_bar, p) if not is_disp else 0.0,
                ))
        Bcenter_arr = np.array(Bcenter_list)
        meff_arr = np.array(meff_list)
        meff_vs_B_storage[rho] = (Bcenter_arr, meff_arr)

        # Panel 5: Onsager check FFT spectrum at low T
        ax5 = axes5[idx]
        _, freqs_, spec_ = fft_amplitude(B_grid, all_dos[0], F_target)
        ax5.plot(freqs_, spec_, lw=1.0)
        ax5.axvline(F_target, color="red", ls="--", lw=1.0,
                    label=rf"$2\pi|\rho|={F_target:.4f}$")
        F_onsager_from_mu = (p.E_flat - mu_bar) / (2.0 * p.a) if not is_disp else (mu_bar - p.E_flat) / (2.0 * p.a)
        if F_onsager_from_mu > 0:
            ax5.axvline(F_onsager_from_mu, color="green", ls=":", lw=1.0,
                        label=rf"$(\mu-E_\mathrm{{ref}})/2a={F_onsager_from_mu:.4f}$")
        ax5.set_xlim(0, 2 * max(F_target, F_onsager_from_mu, 1e-3))
        ax5.set_xlabel("F (FFT of DOS in 1/B)")
        ax5.set_ylabel("amp")
        ax5.set_title(rf"$\rho={rho:+.4f}$")
        ax5.legend(fontsize=8)
        ax5.grid(True, alpha=0.3)

    # Panel 3: m*_eff vs B for anomalous rhos (dispersive shown separately)
    for rho in RHO_LIST:
        if rho > 0:
            continue
        Bc, me = meff_vs_B_storage[rho]
        if Bc.size == 0:
            continue
        axes3.plot(Bc, me, "o-", label=rf"$\rho={rho:+.4f}$")
        # Predicted line through origin with slope c^2/(2 a delta_mu^2)
        delta_mu_bar = luttinger_mu_bar(rho, p) - p.E0
        slope = predicted_meff_slope(delta_mu_bar, p)
        Bline = np.linspace(Bc.min(), Bc.max(), 50)
        axes3.plot(Bline, slope * Bline, "k--", alpha=0.6,
                   label=rf"slope$= c^2/(2a\delta\mu^2)={slope:.2f}$")
    axes3.set_xlabel("B (window center)")
    axes3.set_ylabel(r"$m^*_\mathrm{eff}(B)$ (std-LK form)")
    axes3.set_title("Apparent effective mass from standard-LK fit")
    axes3.legend(fontsize=8)
    axes3.grid(True, alpha=0.3)

    fig1.tight_layout()
    fig1.savefig(os.path.join(FIG_ROOT, "dos_vs_invB_at_T.png"), dpi=200)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG_ROOT, "amplitude_vs_T.png"), dpi=200)
    fig3.tight_layout()
    fig3.savefig(os.path.join(FIG_ROOT, "meff_vs_B.png"), dpi=200)
    fig5.tight_layout()
    fig5.savefig(os.path.join(FIG_ROOT, "onsager_check.png"), dpi=200)
    plt.close("all")

    # Dispersive control standalone panel
    if 0.05 in meff_vs_B_storage:
        Bc, me = meff_vs_B_storage[0.05]
        fig4, ax4 = plt.subplots(1, 1, figsize=(7, 5))
        ax4.plot(Bc, me, "o-", label=rf"$\rho=+0.05$")
        ax4.axhline(1.0 / (2.0 * p.a), color="k", ls="--",
                    label=rf"$m^*=1/(2a)={1/(2*p.a):.4f}$")
        ax4.set_xlabel("B (window center)")
        ax4.set_ylabel(r"$m^*_\mathrm{eff}(B)$")
        ax4.set_title("Dispersive control")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        fig4.tight_layout()
        fig4.savefig(os.path.join(FIG_ROOT, "dispersive_control.png"), dpi=200)
        plt.close(fig4)

    # CSV output
    import csv
    csv_path = os.path.join(FIG_ROOT, "lk_thermal_metrics.csv")
    if metrics_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
            w.writeheader()
            w.writerows(metrics_rows)
        print(f"\nSaved: {csv_path}")

    print(f"\nAll figures saved under:\n  {FIG_ROOT}")
    return 0


# ==================================================================
# V2 and V3 (lightweight — piggyback on main's CSV)
# ==================================================================

def run_v2(p: Params = DEFAULT) -> int:
    """Confirm the FFT peak sits at F = 2 pi |rho| for a couple of rhos."""
    print("=" * 72)
    print("V2. Onsager check (finite-T FFT peak vs 2 pi |rho|)")
    print("=" * 72)
    B_grid = _B_grid_uniform_in_invB(0.01, 0.1, 400)
    T = 5e-4
    for rho in [-0.002, -0.004, 0.05]:
        F_target = 2.0 * np.pi * abs(rho)
        _, dos = dos_vs_B(rho, B_grid, T, p)
        _, freqs, spec = fft_amplitude(B_grid, dos, F_target)
        # Peak-locate (ignore the near-DC component)
        mask = freqs > F_target / 3.0
        peak_F = float(freqs[mask][np.argmax(spec[mask])])
        rel = abs(peak_F - F_target) / F_target
        ok = "OK" if rel < 0.1 else "MARGINAL"
        print(f"   rho={rho:+.4f}: predicted F={F_target:.5f}, "
              f"FFT peak F={peak_F:.5f}, rel_err={rel:.2%}  [{ok}]")
    return 0


def run_v3(p: Params = DEFAULT) -> int:
    """Robustness: shift B-window and check m*_eff fit is smooth."""
    print("=" * 72)
    print("V3. Robustness checks")
    print("=" * 72)
    B_grid = _B_grid_uniform_in_invB(0.01, 0.1, 400)
    rho = -0.002
    T_grid = build_T_grid(rho, 0.03, n_T=8, p=p)

    # Precompute all DOS curves
    all_dos = [dos_vs_B(rho, B_grid, T, p)[1] for T in T_grid]
    F_target = 2.0 * np.pi * abs(rho)

    invB_all = 1.0 / B_grid
    span = invB_all.max() - invB_all.min()
    for win_w_frac in [0.25, 0.33, 0.5]:
        win_w = win_w_frac * span
        wcs = np.linspace(invB_all.min() + win_w / 2,
                           invB_all.max() - win_w / 2, 8)
        meffs = []
        Bcs = []
        for wc in wcs:
            mask = (invB_all >= wc - win_w / 2) & (invB_all <= wc + win_w / 2)
            if mask.sum() < 20:
                continue
            amps_win = np.zeros(T_grid.size)
            for j, T in enumerate(T_grid):
                a_j, _, _ = fft_amplitude(B_grid[mask], all_dos[j][mask], F_target)
                amps_win[j] = a_j
            if amps_win[0] <= 0:
                continue
            ratio = amps_win / amps_win[0]
            m_eff, rmse = fit_meff_lk(T_grid, ratio, 1.0 / wc)
            meffs.append(m_eff)
            Bcs.append(1.0 / wc)
        if len(meffs) >= 2:
            slope, intercept = np.polyfit(Bcs, meffs, 1)
            print(f"   win_w_frac={win_w_frac:.2f}: slope={slope:.3f}, "
                  f"intercept={intercept:+.4f}, n_points={len(meffs)}")

    return 0


# ==================================================================
# Plan-compliant Step 4 implementation
# ==================================================================

PLAN_RHOS = [-0.018, -0.010, +0.010, +0.018, +0.050]


def run_step4_plan(p: Params = DEFAULT, quick: bool = False) -> int:
    """Run the fixed-density normal-vs-anomalous Step 4 calculation.

    This driver implements the current verification plan:
      - measured/smoothed mu_bar_rho(B),
      - fixed-density Onsager FFT target F = 2*pi*|rho|,
      - branch-aware anomalous lower-branch Xp,
      - exact upper-branch dispersive Xp and standard-LK control,
      - side-by-side normal versus anomalous diagnostic figures.
    """
    _ensure_figdir()
    import csv

    rhos = [-0.010, +0.010, +0.050] if quick else PLAN_RHOS
    n_B = 180 if quick else 260
    B_grid = _B_grid_uniform_in_invB(0.012, 0.1, n_B)
    anom_T = np.geomspace(2.0e-4, 3.0e-3, 6)
    disp_T = np.geomspace(8.0e-4, 1.8e-2, 6)

    print("=" * 72)
    print("Step 4 fixed-density LK: normal dispersive vs anomalous flat-band")
    print("=" * 72)
    print(f"Output directory: {FIG_ROOT}")
    print(f"B grid: {n_B} points, B in [{B_grid.min():.4f}, {B_grid.max():.4f}]")
    print(f"rho list: {[f'{r:+.4f}' for r in rhos]}")

    results = {}
    metrics_rows = []
    onsager_rows = []
    thermal_rows = []
    upper_lower_rows = []
    mu_rows = []
    omega_rows = []
    free_energy_rows = []

    for rho in rhos:
        sector = "dispersive" if rho > 0.0 else "anomalous"
        T_grid = disp_T if sector == "dispersive" else anom_T
        F_target = 2.0 * np.pi * abs(rho)
        print(f"\n--- rho={rho:+.4f} ({sector}), F_target={F_target:.5f} ---")

        all_mu = []
        all_chi = []
        all_omega = []
        all_free_energy = []
        amp = []
        peak_by_T = []
        spec_low = None
        mu_fixed = None

        # First pass at Tmin fixes the benchmark chemical potential for the
        # fixed-mu grand-potential diagnostic.
        thermo0 = thermodynamics_vs_B(rho, B_grid, T_grid[0], p)
        mu_fixed = float(np.nanmedian(smooth_background(thermo0["mu"])))

        for T in T_grid:
            thermo = thermodynamics_vs_B(rho, B_grid, T, p, mu_fixed=mu_fixed)
            mu = thermo["mu"]
            signal = thermo["free_energy"]
            all_mu.append(mu)
            all_chi.append(thermo["chi"])
            all_omega.append(thermo["omega_fixed_mu"])
            all_free_energy.append(signal)
            _, freqs, spec = fft_amplitude(B_grid, signal, F_target)
            F_peak, A_peak, delta_F = peak_near_target(freqs, spec, F_target)
            amp.append(A_peak)
            peak_by_T.append(F_peak)
            if spec_low is None:
                spec_low = (freqs, spec)
            print(f"   T={T:.2e}: peak={F_peak:.5f}, amp={A_peak:.3e}")

            omega_amp, omega_freqs, omega_spec = fft_amplitude(B_grid, thermo["omega_fixed_mu"], F_target)
            omega_peak, omega_A_peak, omega_delta_F = peak_near_target(omega_freqs, omega_spec, F_target)
            omega_rows.append(dict(
                rho=rho,
                sector=sector,
                T=T,
                mu_fixed=mu_fixed,
                F_target=F_target,
                F_peak=omega_peak,
                FFT_bin_width=omega_delta_F,
                amplitude=omega_A_peak,
            ))
            free_energy_rows.append(dict(
                rho=rho,
                sector=sector,
                T=T,
                density_convention="n_target(B,T)=rho_ref(E_ref(B),T)+rho_signed",
                F_target=F_target,
                F_peak=F_peak,
                FFT_bin_width=delta_F,
                amplitude=A_peak,
                max_solver_err=float(np.nanmax(thermo["solver_err"])),
                max_omega_identity_err=float(np.nanmax(thermo["omega_identity_err"])),
            ))

        all_mu = np.asarray(all_mu)
        all_chi = np.asarray(all_chi)
        all_omega = np.asarray(all_omega)
        all_free_energy = np.asarray(all_free_energy)
        amp = np.asarray(amp)
        amp_ratio = amp / amp[0] if amp[0] != 0 else amp

        mu_raw = all_mu[0]
        mu_bar = smooth_background(mu_raw)
        delta = mu_bar - p.E0
        labels = classify_lower_branch(delta)
        branch = "LL+" if sector == "dispersive" else branch_mode(labels)

        if sector == "anomalous":
            pred_ratio = normalized_prediction(T_grid, B_grid, mu_bar, "anomalous", p)
        else:
            pred_ratio = normalized_prediction(T_grid, B_grid, mu_bar, "dispersive", p)
        std_ratio = normalized_prediction(T_grid, B_grid, mu_bar, "standard", p)

        m_eff, rmse_std = fit_meff_lk(T_grid, amp_ratio, float(np.median(B_grid)))
        pred_rmse = float(np.sqrt(np.mean((amp_ratio - pred_ratio) ** 2)))
        std_rmse = float(np.sqrt(np.mean((amp_ratio - std_ratio) ** 2)))

        freqs, spec = spec_low
        F_peak0, A_peak0, delta_F = peak_near_target(freqs, spec, F_target)
        residual = F_peak0 - F_target
        onsager_rows.append(dict(
            rho=rho,
            sector=sector,
            F_target=F_target,
            F_peak=F_peak0,
            delta_F=delta_F,
            error_over_bin=residual / delta_F if delta_F else np.nan,
            resolved=abs(residual) <= max(delta_F, 0.02 * F_target),
        ))

        X_anom_min = anomalous_Xp(T_grid[0], B_grid, mu_bar, p)
        X_disp_min = dispersive_Xp(T_grid[0], B_grid, mu_bar, "LL+", p)
        X_std_min = standard_Xp(T_grid[0], B_grid, p)
        RT_plan_min = lk_damping(np.abs(X_anom_min if sector == "anomalous" else X_disp_min))

        metrics_rows.append(dict(
            rho=rho,
            sector=sector,
            branch=branch,
            T_min=T_grid[0],
            T_max=T_grid[-1],
            F_target=F_target,
            F_peak=F_peak0,
            FFT_bin_width=delta_F,
            amp_Tmin=amp[0],
            observable="fixed_density_free_energy",
            mu_bar_mean=float(np.nanmean(mu_bar)),
            mu_bar_std=float(np.nanstd(mu_bar)),
            delta_mean=float(np.nanmean(delta)),
            Xp_median_Tmin=float(np.nanmedian(np.abs(X_anom_min if sector == "anomalous" else X_disp_min))),
            RT_median_Tmin=float(np.nanmedian(RT_plan_min)),
            m_eff_forced_std=m_eff,
            pred_rmse=pred_rmse,
            std_rmse=std_rmse,
            max_solver_err=float(np.nanmax([row["max_solver_err"] for row in free_energy_rows if row["rho"] == rho])),
            max_omega_identity_err=float(np.nanmax([row["max_omega_identity_err"] for row in free_energy_rows if row["rho"] == rho])),
            density_convention="n_target(B,T)=rho_ref(E_ref(B),T)+rho_signed",
        ))

        for B, mu_raw_i, mu_bar_i, rho_ref_i, n_target_i in zip(
            B_grid, mu_raw, mu_bar, thermo0["rho_ref"], thermo0["n_target"]
        ):
            mu_rows.append(dict(
                rho=rho,
                sector=sector,
                B=B,
                T=T_grid[0],
                mu_raw=mu_raw_i,
                mu_bar=mu_bar_i,
                rho_ref=rho_ref_i,
                n_target=n_target_i,
                delta=mu_bar_i - p.E0,
                branch=("LL+" if sector == "dispersive" else classify_lower_branch(mu_bar_i - p.E0)),
            ))

        for T, a_num, a_pred, a_std in zip(T_grid, amp_ratio, pred_ratio, std_ratio):
            thermal_rows.append(dict(
                rho=rho,
                sector=sector,
                branch=branch,
                T=T,
                amp_ratio=a_num,
                predicted_ratio=a_pred,
                standard_ratio=a_std,
                residual_plan=a_num - a_pred,
                residual_standard=a_num - a_std,
            ))

        for B, mu_b, xa, xd, xs in zip(B_grid, mu_bar, X_anom_min, X_disp_min, X_std_min):
            upper_lower_rows.append(dict(
                rho=rho,
                sector=sector,
                B=B,
                mu_bar=mu_b,
                delta=mu_b - p.E0,
                X_LLminus=xa,
                X_LLplus=xd,
                X_parabolic=xs,
                RT_LLminus=lk_damping(abs(xa)) if np.isfinite(xa) else np.nan,
                RT_LLplus=lk_damping(abs(xd)) if np.isfinite(xd) else np.nan,
                RT_parabolic=lk_damping(abs(xs)),
            ))

        results[rho] = dict(
            rho=rho,
            sector=sector,
            T_grid=T_grid,
            B_grid=B_grid,
            invB=1.0 / B_grid,
            all_mu=all_mu,
            all_chi=all_chi,
            all_omega=all_omega,
            all_free_energy=all_free_energy,
            mu_raw=mu_raw,
            mu_bar=mu_bar,
            labels=labels,
            branch=branch,
            amp=amp,
            amp_ratio=amp_ratio,
            pred_ratio=pred_ratio,
            std_ratio=std_ratio,
            freqs=freqs,
            spec=spec,
            F_target=F_target,
            F_peak=F_peak0,
            delta_F=delta_F,
            m_eff=m_eff,
            pred_rmse=pred_rmse,
            std_rmse=std_rmse,
        )

    write_csv(os.path.join(FIG_ROOT, "lk_fixed_density_metrics.csv"), metrics_rows)
    write_csv(os.path.join(FIG_ROOT, "thermo_lk_fixed_density_metrics.csv"), metrics_rows)
    write_csv(os.path.join(FIG_ROOT, "omega_fixed_mu_metrics.csv"), omega_rows)
    write_csv(os.path.join(FIG_ROOT, "free_energy_fixed_density_metrics.csv"), free_energy_rows)
    write_csv(os.path.join(FIG_ROOT, "mu_rho_summary.csv"), mu_rows)
    write_csv(os.path.join(FIG_ROOT, "onsager_summary.csv"), onsager_rows)
    write_csv(os.path.join(FIG_ROOT, "thermal_fit_summary.csv"), thermal_rows)
    write_csv(os.path.join(FIG_ROOT, "upper_lower_xp_summary.csv"), upper_lower_rows)
    write_csv(os.path.join(FIG_ROOT, "lk_thermal_metrics.csv"), metrics_rows)

    plot_step4_outputs(results, p)

    print("\nSaved Step 4 outputs:")
    for name in [
        "thermo_onsager_fft_by_density.png",
        "omega_fixed_mu_thermal_damping.png",
        "free_energy_fixed_density_thermal_damping.png",
        "mu_bar_by_density.png",
        "branch_map.png",
        "thermo_thermal_damping_by_density.png",
        "Xp_RT_by_density.png",
        "upper_vs_lower_Xp_comparison.png",
        "normal_vs_anomalous_thermo_overview.png",
        "compressibility_diagnostic.png",
        "forced_standard_lk_comparison.png",
        "dispersive_control.png",
        "thermo_lk_fixed_density_metrics.csv",
        "omega_fixed_mu_metrics.csv",
        "free_energy_fixed_density_metrics.csv",
        "mu_rho_summary.csv",
        "onsager_summary.csv",
        "thermal_fit_summary.csv",
        "upper_lower_xp_summary.csv",
    ]:
        print(f"  {os.path.join(FIG_ROOT, name)}")
    return 0


def write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ==================================================================
# Lambda = 1 Step 4 theory-test implementation
# ==================================================================

THEORY_ANOMALOUS_RHOS = [-0.040, -0.030, -0.024, -0.018, -0.015, -0.010, -0.005,
                         0.005, 0.010, 0.015, 0.018]
THEORY_DISPERSIVE_RHOS = [0.030, 0.050, 0.065]


def theory_output_dir() -> str:
    os.makedirs(THEORY_FIG_ROOT, exist_ok=True)
    return THEORY_FIG_ROOT


def detrended_uniform_signal(B_grid: np.ndarray, signal: np.ndarray,
                             detrend_deg: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return uniform x=1/B grid, detrended signal, and smooth background."""
    x = 1.0 / np.asarray(B_grid, dtype=float)
    y = np.asarray(signal, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    coeffs = np.polyfit(x, y, detrend_deg)
    bg = np.polyval(coeffs, x)
    return x, y - bg, bg


def global_fft_peak(B_grid: np.ndarray, signal: np.ndarray,
                    F_target: float) -> dict:
    amp_interp, freqs, amps = fft_amplitude(B_grid, signal, F_target)
    F_peak, A_peak, dF = peak_near_target(freqs, amps, F_target)
    return {
        "amp_interp": amp_interp,
        "freqs": freqs,
        "amps": amps,
        "F_peak": F_peak,
        "A_peak": A_peak,
        "delta_F": dF,
        "resolved": abs(F_peak - F_target) <= max(dF, 0.02 * F_target),
    }


def period_windows(B_grid: np.ndarray, F_target: float,
                   min_points: int = 12) -> list[dict]:
    """Build one-period windows in x=1/B."""
    x = np.sort(1.0 / B_grid)
    period = 1.0 / F_target
    if F_target <= 0 or period >= (x.max() - x.min()):
        return []
    starts = np.arange(x.min(), x.max() - period, period)
    windows = []
    for idx, start in enumerate(starts):
        stop = start + period
        mask = (1.0 / B_grid >= start) & (1.0 / B_grid < stop)
        if int(mask.sum()) < min_points:
            continue
        x_mid = 0.5 * (start + stop)
        windows.append({
            "period_index": idx,
            "x_start": float(start),
            "x_stop": float(stop),
            "B_avg": float(np.mean(B_grid[mask])),
            "B_center": float(1.0 / x_mid),
            "mask": mask,
        })
    return windows


def local_period_amplitude(B_grid: np.ndarray, signal: np.ndarray,
                           F_target: float, mask: np.ndarray) -> float:
    """Fit one local sinusoid at F_target to the detrended signal in a window."""
    x_all, y_all, _ = detrended_uniform_signal(B_grid, signal)
    x_mask = 1.0 / B_grid[mask]
    order = np.argsort(x_mask)
    xw = x_mask[order]
    yw = np.interp(xw, x_all, y_all)
    if xw.size < 4:
        return float("nan")
    phase = 2.0 * np.pi * F_target * xw
    design = np.column_stack([np.sin(phase), np.cos(phase), np.ones_like(phase)])
    coeffs, *_ = np.linalg.lstsq(design, yw, rcond=None)
    return float(np.hypot(coeffs[0], coeffs[1]))


def theory_branch_for(rho: float, sector: str, B: float, mu_bar: float,
                      p: Params = THEORY_PARAMS) -> str:
    if sector == "normal":
        delta = mu_bar - p.E0
        return "LL+" if delta * delta >= p.c ** 2 * B else "LLTR+"
    delta = mu_bar - p.E0
    return "LL-" if delta > 0.0 else "LLTR-"


def theory_Xp_for_branch(T: float, B: np.ndarray | float,
                         mu_bar: np.ndarray | float, branch: str,
                         p: Params = THEORY_PARAMS) -> np.ndarray | float:
    B_arr = np.asarray(B, dtype=float)
    mu_arr = np.asarray(mu_bar, dtype=float)
    if branch == "LL+":
        val = dispersive_Xp(T, B_arr, mu_arr, "LL+", p)
    elif branch == "LLTR+":
        val = dispersive_Xp(T, B_arr, mu_arr, "LLTR+", p)
    elif branch in ("LL-", "LLTR-"):
        # anomalous_Xp chooses LL-/LLTR- by the sign of mu-E0.
        val = anomalous_Xp(T, B_arr, mu_arr, p)
    elif branch == "standard":
        val = standard_Xp(T, B_arr, p)
    else:
        raise ValueError(branch)
    return float(val) if np.ndim(val) == 0 else val


def normalized_rt_for_window(T_grid: np.ndarray, B_avg: float, mu_bar: float,
                             branch: str, p: Params = THEORY_PARAMS) -> np.ndarray:
    vals = []
    for T in T_grid:
        X = abs(theory_Xp_for_branch(T, B_avg, mu_bar, branch, p))
        vals.append(lk_damping(X))
    vals = np.asarray(vals, dtype=float)
    return vals / vals[0] if vals[0] != 0 else vals


def copy_or_write_step4_captions(outdir: str) -> None:
    """Ensure the planned caption file is present in the output directory."""
    path = os.path.join(outdir, "figure_captions.md")
    if os.path.exists(path):
        return
    lines = [
        "# Step 4 Figure Captions",
        "",
        "This file is generated by `Codes/step4_lk_thermal.py --mode theory`.",
        "It should be replaced by the curated caption file if the planned figure list changes.",
        "",
        "See `Implementation and Verification Plan for Step 4.1.md` for the required caption content.",
    ]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_step4_theory_tests(p: Params = THEORY_PARAMS, quick: bool = False) -> int:
    outdir = theory_output_dir()
    copy_or_write_step4_captions(outdir)

    rhos = [-0.010, 0.010, 0.050] if quick else THEORY_ANOMALOUS_RHOS + THEORY_DISPERSIVE_RHOS
    rho_lim = rho_max(p)
    B_grid = _B_grid_uniform_in_invB(0.008, 0.080, 420 if not quick else 220)
    T_anom = np.geomspace(2.0e-4, 3.0e-3, 7)
    T_norm = np.geomspace(6.0e-4, 1.6e-2, 7)

    print("=" * 72)
    print("Step 4 theory tests: Lambda = 1 thermodynamic oscillations")
    print("=" * 72)
    print(f"Output directory: {outdir}")
    print(f"rho_max = {rho_lim:.6f}")

    cutoff_rows = []
    for rho in rhos:
        cutoff_rows.append({
            "Lambda": p.Lambda,
            "rho_max": rho_lim,
            "rho": rho,
            "abs_rho": abs(rho),
            "pass": abs(rho) < rho_lim,
        })
    if not all(row["pass"] for row in cutoff_rows):
        write_csv(os.path.join(outdir, "density_cutoff_check.csv"), cutoff_rows)
        raise ValueError("At least one test density exceeds the Lambda=1 cutoff.")

    results: dict[float, dict] = {}
    frequency_rows = []
    period_rows = []
    rt_rows = []
    meff_rows = []
    identity_rows = []

    for rho in rhos:
        sector = "normal" if rho > 0.0 and abs(rho) > 0.020 else "anomalous"
        T_grid = T_anom if sector == "anomalous" else T_norm
        F_target = 2.0 * np.pi * abs(rho)
        print(f"\n--- rho={rho:+.4f}, sector={sector}, F={F_target:.5f} ---")

        thermo0 = thermodynamics_sector_vs_B(rho, B_grid, T_grid[0], p)
        mu_bar = smooth_background(thermo0["fermi_energy"])
        mu_sector_bar = smooth_background(thermo0["mu"])
        mu_fixed = float(np.nanmedian(mu_sector_bar))

        observable_data = {
            "omega": [],
            "free_energy": [],
            "internal_energy": [],
            "magnetization": [],
            "mu": [],
            "compressibility": [],
        }
        for T in T_grid:
            thermo = thermodynamics_sector_vs_B(rho, B_grid, T, p, mu_fixed_sector=mu_fixed)
            observable_data["omega"].append(thermo["omega_fixed_mu"])
            observable_data["free_energy"].append(thermo["free_energy"])
            observable_data["internal_energy"].append(thermo["internal_energy"])
            observable_data["magnetization"].append(thermo["magnetization"])
            observable_data["mu"].append(thermo["fermi_energy"])
            observable_data["compressibility"].append(thermo["chi"])
            regime_values = sorted(set(str(x) for x in thermo["regime"]))
            sector_values = sorted(set(str(x) for x in thermo["carrier_sector"]))
            identity_rows.append({
                "rho": rho,
                "sector": sector,
                "regime": ";".join(regime_values),
                "carrier_sector": ";".join(sector_values),
                "T": T,
                "max_solver_err": float(np.nanmax(thermo["solver_err"])),
                "max_omega_identity_err": float(np.nanmax(thermo["omega_identity_err"])),
                "density_convention": "three_regime_carrier_resolved",
                "min_rho_up": float(np.nanmin(thermo["rho_up"])),
                "min_rho_down": float(np.nanmin(thermo["rho_down"])),
                "min_rho_prime_e": (
                    float(np.nanmin(thermo["rho_prime_e"]))
                    if np.any(np.isfinite(thermo["rho_prime_e"])) else np.nan
                ),
            })

        for key in observable_data:
            observable_data[key] = np.asarray(observable_data[key])

        spectra = {}
        global_amp = {}
        for obs in ["omega", "free_energy", "mu", "magnetization"]:
            spectra[obs] = []
            global_amp[obs] = []
            for j, T in enumerate(T_grid):
                peak = global_fft_peak(B_grid, observable_data[obs][j], F_target)
                spectra[obs].append(peak)
                global_amp[obs].append(peak["A_peak"])
                frequency_rows.append({
                    "rho": rho,
                    "sector": sector,
                    "observable": obs,
                    "T": T,
                    "F_target": F_target,
                    "thermo_F_peak": peak["F_peak"],
                    "DOS_F_peak_reference": np.nan,
                    "delta_F": peak["delta_F"],
                    "error_over_bin": ((peak["F_peak"] - F_target) / peak["delta_F"]
                                       if peak["delta_F"] else np.nan),
                    "resolved": peak["resolved"],
                })
            global_amp[obs] = np.asarray(global_amp[obs])

        wins = period_windows(B_grid, F_target)
        rt_by_window = []
        meff_by_window = []
        for win in wins:
            amps = []
            for j, T in enumerate(T_grid):
                amp = local_period_amplitude(B_grid, observable_data["free_energy"][j],
                                             F_target, win["mask"])
                amps.append(amp)
            amps = np.asarray(amps)
            if not np.isfinite(amps[0]) or amps[0] == 0:
                continue
            amp_norm = amps / amps[0]
            mu_w = float(np.interp(win["B_avg"], B_grid, mu_bar))
            branch = theory_branch_for(rho, sector, win["B_avg"], mu_w, p)
            rt_pred = normalized_rt_for_window(T_grid, win["B_avg"], mu_w, branch, p)
            std_pred = normalized_rt_for_window(T_grid, win["B_avg"], mu_w, "standard", p)
            m_fit, m_rmse = fit_meff_lk(T_grid, amp_norm, win["B_avg"])
            m_pred = apparent_mass_theory(win["B_avg"], mu_w, branch, p)
            m_std = 1.0 / (2.0 * p.a)

            rt_by_window.append((win, amp_norm, rt_pred, std_pred, branch))
            meff_by_window.append((win["B_avg"], m_fit, m_pred, m_std, branch))
            for T, amp, an, rp, sp in zip(T_grid, amps, amp_norm, rt_pred, std_pred):
                period_rows.append({
                    "rho": rho,
                    "sector": sector,
                    "observable": "free_energy",
                    "period_index": win["period_index"],
                    "invB_start": win["x_start"],
                    "invB_stop": win["x_stop"],
                    "B_avg": win["B_avg"],
                    "T": T,
                    "amplitude": amp,
                    "normalized_amplitude": an,
                })
                rt_rows.append({
                    "rho": rho,
                    "sector": sector,
                    "branch": branch,
                    "period_index": win["period_index"],
                    "B_avg": win["B_avg"],
                    "T": T,
                    "A_norm": an,
                    "RT_pred_norm": rp,
                    "standard_pred_norm": sp,
                    "residual_theory": an - rp,
                    "residual_standard": an - sp,
                })
            meff_rows.append({
                "rho": rho,
                "sector": sector,
                "branch": branch,
                "period_index": win["period_index"],
                "B_avg": win["B_avg"],
                "m_eff_fit": m_fit,
                "m_eff_pred_exact_branch": m_pred,
                "m_eff_parabolic": m_std,
                "fit_rmse": m_rmse,
            })

        results[rho] = {
            "rho": rho,
            "sector": sector,
            "T_grid": T_grid,
            "B_grid": B_grid,
            "invB": 1.0 / B_grid,
            "F_target": F_target,
            "mu_bar": mu_bar,
            "mu_raw": thermo0["mu"],
            "observable_data": observable_data,
            "spectra": spectra,
            "global_amp": global_amp,
            "windows": wins,
            "rt_by_window": rt_by_window,
            "meff_by_window": meff_by_window,
        }

    write_csv(os.path.join(outdir, "density_cutoff_check.csv"), cutoff_rows)
    write_csv(os.path.join(outdir, "frequency_check.csv"), frequency_rows)
    write_csv(os.path.join(outdir, "period_amplitude_summary.csv"), period_rows)
    write_csv(os.path.join(outdir, "RT_fit_summary.csv"), rt_rows)
    write_csv(os.path.join(outdir, "effective_mass_summary.csv"), meff_rows)
    write_csv(os.path.join(outdir, "thermodynamic_identity_checks.csv"), identity_rows)

    plot_step4_theory_outputs(results, outdir, p)

    print("\nSaved Step 4 theory-test outputs:")
    for name in [
        "00_mu_vs_invB.png",
        "00_raw_fermi_energy_vs_invB.png",
        "00e_hole_raw_EF_vs_magnetization.png",
        "00f_hole_tilde_EF_vs_magnetization.png",
        "00_omega_vs_invB.png",
        "00_free_energy_vs_invB.png",
        "00_internal_energy_vs_invB.png",
        "00_magnetization_vs_invB.png",
        "00_raw_magnetization_vs_invB.png",
        "00_omega_f_mu_vs_invB.png",
        "01_frequency_check_omega_f_mu.png",
        "01b_frequency_check_magnetization.png",
        "02_RT_period_amplitudes_normal_vs_anomalous.png",
        "03_effective_mass_normal_vs_anomalous.png",
        "04_mu_branch_map.png",
        "05_theory_Xp_and_RT_scales.png",
        "06_global_fft_amplitudes.png",
        "diagnostic_spectral_DOS.png",
        "diagnostic_compressibility.png",
        "figure_captions.md",
        "frequency_check.csv",
        "period_amplitude_summary.csv",
        "RT_fit_summary.csv",
        "effective_mass_summary.csv",
        "thermodynamic_identity_checks.csv",
        "density_cutoff_check.csv",
    ]:
        print(f"  {os.path.join(outdir, name)}")
    return 0


def plot_step4_theory_outputs(results: dict, outdir: str,
                              p: Params = THEORY_PARAMS) -> None:
    rhos = list(results.keys())
    colors = {"anomalous": "#b02a30", "normal": "#1f6f3f"}

    def plot_single_signal(obs: str, title: str, filename: str) -> None:
        fig, axes = plt.subplots(len(rhos), 1, figsize=(7.8, 1.75 * len(rhos)),
                                 sharex=True, squeeze=False)
        for row, rho in enumerate(rhos):
            r = results[rho]
            ax = axes[row, 0]
            x, y_detrended, _ = detrended_uniform_signal(
                r["B_grid"], r["observable_data"][obs][0]
            )
            ax.plot(x, y_detrended, color=colors[r["sector"]], lw=0.95)
            ax.axhline(0.0, color="0.45", lw=0.5, alpha=0.5)
            ax.set_ylabel(f"{rho:+.3f}", rotation=0, ha="right", va="center")
            ax.grid(True, alpha=0.22)
        axes[-1, 0].set_xlabel("1/B")
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(os.path.join(outdir, filename), dpi=220)
        plt.close(fig)

    plot_single_signal(
        "mu",
        r"Sector-consistent Fermi-energy oscillation before FFT at $T_{\min}$",
        "00_mu_vs_invB.png",
    )

    # 00. Raw sector-consistent Fermi energy, no background subtraction.
    fig, axes = plt.subplots(len(rhos), 1, figsize=(7.8, 1.75 * len(rhos)),
                             sharex=True, squeeze=False)
    for row, rho in enumerate(rhos):
        r = results[rho]
        ax = axes[row, 0]
        x = 1.0 / r["B_grid"]
        ax.plot(x, r["observable_data"]["mu"][0],
                color=colors[r["sector"]], lw=0.95)
        ax.set_ylabel(f"{rho:+.3f}", rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.22)
    axes[-1, 0].set_xlabel("1/B")
    fig.suptitle(r"Raw sector-consistent Fermi energy at $T_{\min}$")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(outdir, "00_raw_fermi_energy_vs_invB.png"), dpi=220)
    plt.close(fig)

    # 00e. Compare raw Fermi-energy steps and raw magnetization for hole densities.
    hole_rhos = [rho for rho in rhos if rho < 0.0]
    if hole_rhos:
        fig, axes = plt.subplots(len(hole_rhos), 2, figsize=(11.2, 2.0 * len(hole_rhos)),
                                 sharex=True, squeeze=False)
        for row, rho in enumerate(hole_rhos):
            r = results[rho]
            x = 1.0 / r["B_grid"]
            ef = r["observable_data"]["mu"][0]
            mag = r["observable_data"]["magnetization"][0]
            color = colors[r["sector"]]
            axes[row, 0].plot(x, ef, color=color, lw=0.95)
            axes[row, 1].plot(x, mag, color=color, lw=0.95)
            axes[row, 0].set_ylabel(f"{rho:+.3f}", rotation=0, ha="right", va="center")
            for ax in axes[row, :]:
                ax.grid(True, alpha=0.22)
        axes[0, 0].set_title(r"raw $E_F(1/B)$")
        axes[0, 1].set_title(r"raw $M_\rho(1/B)$")
        axes[-1, 0].set_xlabel("1/B")
        axes[-1, 1].set_xlabel("1/B")
        fig.suptitle(r"Hole-side comparison: Fermi-level steps vs magnetization")
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(os.path.join(outdir, "00e_hole_raw_EF_vs_magnetization.png"), dpi=220)
        plt.close(fig)

        fig, axes = plt.subplots(len(hole_rhos), 2, figsize=(11.2, 2.0 * len(hole_rhos)),
                                 sharex=True, squeeze=False)
        for row, rho in enumerate(hole_rhos):
            r = results[rho]
            x_ef, ef_tilde, _ = detrended_uniform_signal(
                r["B_grid"], r["observable_data"]["mu"][0]
            )
            x_m, m_tilde, _ = detrended_uniform_signal(
                r["B_grid"], r["observable_data"]["magnetization"][0]
            )
            color = colors[r["sector"]]
            axes[row, 0].plot(x_ef, ef_tilde, color=color, lw=0.95)
            axes[row, 1].plot(x_m, m_tilde, color=color, lw=0.95)
            axes[row, 0].axhline(0.0, color="0.45", lw=0.5, alpha=0.5)
            axes[row, 1].axhline(0.0, color="0.45", lw=0.5, alpha=0.5)
            axes[row, 0].set_ylabel(f"{rho:+.3f}", rotation=0, ha="right", va="center")
            for ax in axes[row, :]:
                ax.grid(True, alpha=0.22)
        axes[0, 0].set_title(r"$\widetilde E_F(1/B)$")
        axes[0, 1].set_title(r"$\widetilde M_\rho(1/B)$")
        axes[-1, 0].set_xlabel("1/B")
        axes[-1, 1].set_xlabel("1/B")
        fig.suptitle(r"Hole-side comparison: background-subtracted Fermi energy and magnetization")
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(os.path.join(outdir, "00f_hole_tilde_EF_vs_magnetization.png"), dpi=220)
        plt.close(fig)

    plot_single_signal(
        "omega",
        r"Carrier-sector grand-potential oscillation before FFT at $T_{\min}$",
        "00_omega_vs_invB.png",
    )
    plot_single_signal(
        "free_energy",
        r"Carrier-sector free-energy oscillation before FFT at $T_{\min}$",
        "00_free_energy_vs_invB.png",
    )
    plot_single_signal(
        "magnetization",
        r"Carrier-sector magnetization oscillation before FFT at $T_{\min}$",
        "00_magnetization_vs_invB.png",
    )

    # 00. Raw total sector magnetization, no background subtraction.
    fig, axes = plt.subplots(len(rhos), 1, figsize=(7.8, 1.75 * len(rhos)),
                             sharex=True, squeeze=False)
    for row, rho in enumerate(rhos):
        r = results[rho]
        ax = axes[row, 0]
        x = 1.0 / r["B_grid"]
        ax.plot(x, r["observable_data"]["magnetization"][0],
                color=colors[r["sector"]], lw=0.95)
        ax.set_ylabel(f"{rho:+.3f}", rotation=0, ha="right", va="center")
        ax.grid(True, alpha=0.22)
    axes[-1, 0].set_xlabel("1/B")
    fig.suptitle(r"Raw carrier-sector magnetization at $T_{\min}$")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(outdir, "00_raw_magnetization_vs_invB.png"), dpi=220)
    plt.close(fig)

    # 00. T=0 raw Fermi energy and direct internal energy, no background subtraction.
    fig, axes = plt.subplots(len(rhos), 2, figsize=(10.8, 1.85 * len(rhos)),
                             sharex=True, squeeze=False)
    for row, rho in enumerate(rhos):
        r = results[rho]
        zt = zero_temperature_sector_thermo_vs_B(rho, r["B_grid"], p)
        x = 1.0 / r["B_grid"]
        axes[row, 0].plot(x, zt["fermi_energy"], color=colors[r["sector"]], lw=0.95)
        axes[row, 1].plot(x, zt["internal_energy"], color=colors[r["sector"]], lw=0.95)
        axes[row, 0].set_ylabel(f"{rho:+.3f}", rotation=0, ha="right", va="center")
        for ax in axes[row, :]:
            ax.grid(True, alpha=0.22)
    axes[0, 0].set_title(r"raw $E_F(1/B)$ at $T=0$")
    axes[0, 1].set_title(r"raw direct $u_{\cal S}(1/B)$ at $T=0$")
    axes[-1, 0].set_xlabel("1/B")
    axes[-1, 1].set_xlabel("1/B")
    fig.suptitle(r"Zero-temperature carrier-sector filling: Fermi energy and internal energy")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(os.path.join(outdir, "00_internal_energy_vs_invB.png"), dpi=220)
    plt.close(fig)

    # 00. Show the actual oscillatory signals before FFT.
    nrows_signal = len(rhos)
    fig, axes = plt.subplots(nrows_signal, 3, figsize=(12, 2.2 * nrows_signal),
                             sharex=True, squeeze=False)
    for row, rho in enumerate(rhos):
        r = results[rho]
        for col, obs in enumerate(["omega", "free_energy", "mu"]):
            ax = axes[row, col]
            x, y_detrended, _ = detrended_uniform_signal(
                r["B_grid"], r["observable_data"][obs][0]
            )
            ax.plot(x, y_detrended, color=colors[r["sector"]], lw=0.9)
            ax.set_ylabel(f"{rho:+.3f}")
            if row == 0:
                ax.set_title(obs)
            if row == nrows_signal - 1:
                ax.set_xlabel("1/B")
            ax.grid(True, alpha=0.22)
    fig.suptitle(r"Thermodynamic oscillation signals before FFT at $T_{\min}$")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "00_omega_f_mu_vs_invB.png"), dpi=220)
    plt.close(fig)

    # 01. Frequency check for omega, f, and mu.
    ncols = min(4, len(rhos))
    nrows = int(np.ceil(len(rhos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, rho in zip(axes.ravel(), rhos):
        ax.axis("on")
        r = results[rho]
        for obs, color in [("omega", "#6a3d9a"), ("free_energy", "#1b9e77"), ("mu", "#d95f02")]:
            peak = r["spectra"][obs][0]
            ax.plot(peak["freqs"], peak["amps"], lw=1.0, color=color, label=obs)
        ax.axvline(r["F_target"], color="k", ls="--", lw=1.0, label=r"$2\pi|\rho|$")
        ax.set_xlim(0, max(0.45, 2.2 * r["F_target"]))
        ax.set_title(f"rho={rho:+.3f} ({r['sector']})")
        ax.set_xlabel("F")
        ax.set_ylabel("|FFT|")
        ax.grid(True, alpha=0.25)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle(r"Frequency check: $\omega$, $f$, and $\mu_\rho$")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01_frequency_check_omega_f_mu.png"), dpi=220)
    plt.close(fig)

    # 01b. Frequency check for exact discrete-spectrum magnetization.
    ncols = min(4, len(rhos))
    nrows = int(np.ceil(len(rhos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, rho in zip(axes.ravel(), rhos):
        r = results[rho]
        ax.axis("on")
        spec = r["spectra"]["magnetization"][0]
        ax.plot(spec["freqs"], spec["amps"], color="#6a3d9a", lw=1.0)
        ax.axvline(r["F_target"], color="k", ls="--", lw=0.9, label=r"$2\pi|\rho|$")
        ax.axvline(spec["F_peak"], color="#e7298a", ls=":", lw=1.0, label="M peak")
        ax.set_xlim(0.0, max(0.55, 2.6 * r["F_target"]))
        ax.set_title(f"rho={rho:+.3f}, peak={spec['F_peak']:.3f}")
        ax.set_xlabel("F")
        ax.set_ylabel("|FFT|")
        ax.grid(True, alpha=0.24)
        ax.legend(fontsize=7)
    fig.suptitle(r"Frequency check: exact discrete carrier-sector magnetization")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "01b_frequency_check_magnetization.png"), dpi=220)
    plt.close(fig)

    # 02. Period-resolved RT comparison. Use one representative anomalous and one normal rho.
    chosen = []
    for sector in ["normal", "anomalous"]:
        candidates = [rho for rho in rhos if results[rho]["sector"] == sector and results[rho]["rt_by_window"]]
        if candidates:
            chosen.append(candidates[len(candidates) // 2])
    fig, axes = plt.subplots(max(1, len(chosen)), 2, figsize=(10, 4.0 * max(1, len(chosen))),
                             squeeze=False)
    for row, rho in enumerate(chosen):
        r = results[rho]
        ax_data, ax_resid = axes[row]
        # Show up to three period windows: low, middle, high B.
        entries = r["rt_by_window"]
        indices = np.linspace(0, len(entries) - 1, min(3, len(entries)), dtype=int)
        for idx in indices:
            win, amp_norm, rt_pred, std_pred, branch = entries[idx]
            label = f"Bavg={win['B_avg']:.3f}, {branch}"
            ax_data.plot(r["T_grid"], amp_norm, "o-", label=label)
            ax_data.plot(r["T_grid"], rt_pred, "k--", lw=1.0)
            ax_resid.plot(r["T_grid"], amp_norm - rt_pred, "o-", label=label)
        ax_data.set_xscale("log")
        ax_data.set_title(f"rho={rho:+.3f} {r['sector']}: period amplitudes")
        ax_data.set_ylabel(r"$A_w(T)/A_w(T_{\min})$")
        ax_data.set_xlabel("T")
        ax_data.grid(True, alpha=0.25)
        ax_data.legend(fontsize=8)
        ax_resid.axhline(0.0, color="k", lw=0.8)
        ax_resid.set_xscale("log")
        ax_resid.set_title("data - theory")
        ax_resid.set_ylabel("residual")
        ax_resid.set_xlabel("T")
        ax_resid.grid(True, alpha=0.25)
    fig.suptitle(r"Period-resolved $R_T$ comparison")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "02_RT_period_amplitudes_normal_vs_anomalous.png"), dpi=220)
    plt.close(fig)

    # 03. Effective mass comparison.
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    for rho in rhos:
        r = results[rho]
        if not r["meff_by_window"]:
            continue
        Bv = np.array([entry[0] for entry in r["meff_by_window"]])
        mf = np.array([entry[1] for entry in r["meff_by_window"]])
        mp = np.array([entry[2] for entry in r["meff_by_window"]])
        color = colors[r["sector"]]
        ax.plot(Bv, mf, "o", color=color, alpha=0.75, label=f"fit rho={rho:+.3f}")
        ax.plot(Bv, mp, "-", color=color, alpha=0.55)
    ax.axhline(1.0 / (2.0 * p.a), color="k", ls="--", label=r"$1/(2a)$")
    ax.set_xlabel(r"$\bar B_w$")
    ax.set_ylabel(r"$m_{\rm eff}$")
    ax.set_title("Apparent effective mass: fitted points and branch predictions")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "03_effective_mass_normal_vs_anomalous.png"), dpi=220)
    plt.close(fig)

    # 04. Chemical potential branch map.
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for rho in rhos:
        r = results[rho]
        color = colors[r["sector"]]
        axes[0].plot(r["B_grid"], r["mu_raw"], color=color, alpha=0.18, lw=0.7)
        axes[0].plot(r["B_grid"], r["mu_bar"], color=color, lw=1.4,
                     label=f"rho={rho:+.3f}")
        delta = r["mu_bar"] - p.E0
        axes[1].plot(r["B_grid"], delta, color=color, lw=1.2,
                     label=f"rho={rho:+.3f}")
    axes[0].axhline(p.E0, color="k", ls=":", lw=1.0, label=r"$E_0$")
    axes[1].axhline(0.0, color="k", ls=":", lw=1.0)
    axes[0].set_ylabel(r"$\mu_\rho$")
    axes[1].set_ylabel(r"$\bar\mu_\rho-E_0$")
    axes[1].set_xlabel("B")
    axes[0].set_title("Fixed-density chemical potential and branch assignment")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "04_mu_branch_map.png"), dpi=220)
    plt.close(fig)

    # 05. Predicted Xp and RT scales at Tmin.
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for rho in rhos:
        r = results[rho]
        color = colors[r["sector"]]
        T0 = r["T_grid"][0]
        Xvals = []
        for B, mu in zip(r["B_grid"], r["mu_bar"]):
            br = theory_branch_for(rho, r["sector"], B, mu, p)
            Xvals.append(abs(theory_Xp_for_branch(T0, B, mu, br, p)))
        Xvals = np.asarray(Xvals)
        axes[0].plot(r["B_grid"], Xvals, color=color, lw=1.2, label=f"rho={rho:+.3f}")
        axes[1].plot(r["B_grid"], lk_damping(Xvals), color=color, lw=1.2, label=f"rho={rho:+.3f}")
    axes[0].set_ylabel(r"$|X_1(B,T_{\min})|$")
    axes[1].set_ylabel(r"$R_T$")
    axes[1].set_xlabel("B")
    axes[0].set_title("Predicted local LK scales")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "05_theory_Xp_and_RT_scales.png"), dpi=220)
    plt.close(fig)

    # 06. Global FFT amplitudes.
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=False)
    for ax, obs in zip(axes, ["omega", "free_energy", "mu"]):
        for rho in rhos:
            r = results[rho]
            amp = r["global_amp"][obs]
            ratio = amp / amp[0] if amp[0] != 0 else amp
            ax.plot(r["T_grid"], ratio, "o-", color=colors[r["sector"]], alpha=0.75,
                    label=f"{rho:+.3f}")
        ax.set_xscale("log")
        ax.set_title(obs)
        ax.set_xlabel("T")
        ax.set_ylabel(r"global $A(T)/A(T_{\min})$")
        ax.grid(True, alpha=0.25)
    axes[-1].legend(fontsize=7, ncol=2)
    fig.suptitle("Global FFT amplitudes for thermodynamic observables")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "06_global_fft_amplitudes.png"), dpi=220)
    plt.close(fig)

    # Diagnostic spectral DOS proxy: LL spectrum plus mu for representative densities.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    rep = []
    for sector in ["normal", "anomalous"]:
        candidates = [rho for rho in rhos if results[rho]["sector"] == sector]
        if candidates:
            rep.append(candidates[len(candidates) // 2])
    for ax, rho in zip(axes, rep):
        r = results[rho]
        B_plot = r["B_grid"][::8]
        for B in B_plot:
            ll = analytical_LL_spectrum(float(B), p)
            ax.plot(np.full_like(ll, B), ll, ".", color="0.75", ms=1.5)
        ax.plot(r["B_grid"], r["mu_bar"], color=colors[r["sector"]], lw=2.0,
                label=rf"$\bar\mu_\rho$, $\rho={rho:+.3f}$")
        ax.axhline(p.E0, color="k", ls=":", lw=1.0)
        ax.set_title(f"{r['sector']} diagnostic")
        ax.set_xlabel("B")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Energy")
    fig.suptitle("Diagnostic LL spectrum and fixed-density chemical potential")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "diagnostic_spectral_DOS.png"), dpi=220)
    plt.close(fig)

    # Diagnostic compressibility.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 3.0 * nrows),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, rho in zip(axes.ravel(), rhos):
        ax.axis("on")
        r = results[rho]
        ax.plot(r["invB"], r["observable_data"]["compressibility"][0],
                color=colors[r["sector"]], lw=1.0)
        ax.set_title(f"rho={rho:+.3f}")
        ax.set_xlabel("1/B")
        ax.set_ylabel(r"$\partial n/\partial\mu$")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Compressibility diagnostic only")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "diagnostic_compressibility.png"), dpi=220)
    plt.close(fig)


def plot_step4_outputs(results: dict, p: Params = DEFAULT) -> None:
    rhos = list(results.keys())
    colors = {
        "anomalous": "#b02a30",
        "dispersive": "#1f6f3f",
    }

    # Onsager FFT spectra from fixed-density Helmholtz free energy.
    n = len(rhos)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.2), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, rho in zip(axes, rhos):
        r = results[rho]
        ax.plot(r["freqs"], r["spec"], color=colors[r["sector"]], lw=1.0)
        ax.axvline(r["F_target"], color="#d95f02", ls="--", lw=1.2, label="2*pi|rho|")
        ax.axvline(r["F_peak"], color="#009e73", ls=":", lw=1.5, label="FFT peak")
        ax.set_xlim(0, max(0.35, 2.5 * r["F_target"]))
        ax.set_title(f"rho={rho:+.3f}\npeak={r['F_peak']:.3f}")
        ax.set_xlabel("F")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("|FFT|")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Fixed-density Onsager check: free-energy FFT peaks")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_ROOT, "thermo_onsager_fft_by_density.png"), dpi=220)
    plt.close(fig)

    # Chemical potential backgrounds.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for rho in rhos:
        r = results[rho]
        ax.plot(r["B_grid"], r["mu_raw"], color=colors[r["sector"]], alpha=0.25, lw=0.8)
        ax.plot(r["B_grid"], r["mu_bar"], color=colors[r["sector"]], lw=1.8,
                label=f"rho={rho:+.3f} ({r['sector']})")
    ax.axhline(p.E0, color="k", ls=":", lw=1.0, label="E0")
    ax.set_xlabel("B")
    ax.set_ylabel("mu_rho(B)")
    ax.set_title("Raw and smoothed fixed-density chemical potential")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_ROOT, "mu_bar_by_density.png"), dpi=220)
    plt.close(fig)

    # Branch map.
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.8))
    for j, rho in enumerate(rhos):
        r = results[rho]
        vals = np.where(r["mu_bar"] - p.E0 > 0.0, 1.0, -1.0)
        if r["sector"] == "dispersive":
            vals = np.full_like(vals, 2.0)
        ax.scatter(r["B_grid"], np.full_like(r["B_grid"], j), c=vals,
                   cmap="coolwarm", vmin=-2, vmax=2, s=10)
    ax.set_yticks(range(len(rhos)))
    ax.set_yticklabels([f"{rho:+.3f}" for rho in rhos])
    ax.set_xlabel("B")
    ax.set_ylabel("rho")
    ax.set_title("Branch map: LLTR- (blue), LL- (red), dispersive control (dark red)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_ROOT, "branch_map.png"), dpi=220)
    plt.close(fig)

    # Thermal damping by density from fixed-density Helmholtz free energy.
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, rho in zip(axes, rhos):
        r = results[rho]
        ax.plot(r["T_grid"], r["amp_ratio"], "o-", color=colors[r["sector"]], label="numerical")
        ax.plot(r["T_grid"], r["pred_ratio"], "k--", label="branch-aware" if r["sector"] == "anomalous" else "exact upper")
        ax.plot(r["T_grid"], r["std_ratio"], color="0.5", ls=":", label="standard LK")
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.set_title(f"rho={rho:+.3f}\n{r['sector']}")
        ax.set_xlabel("T")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("A(T)/A(Tmin)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Free-energy thermal damping: same extraction, different LL spectrum")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_ROOT, "free_energy_fixed_density_thermal_damping.png"), dpi=220)
    fig.savefig(os.path.join(FIG_ROOT, "thermo_thermal_damping_by_density.png"), dpi=220)
    plt.close(fig)

    # Fixed-mu grand-potential diagnostic.
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.3), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, rho in zip(axes, rhos):
        r = results[rho]
        omega_amp = []
        for omega_signal in r["all_omega"]:
            _, freqs_o, spec_o = fft_amplitude(r["B_grid"], omega_signal, r["F_target"])
            _, A_o, _ = peak_near_target(freqs_o, spec_o, r["F_target"])
            omega_amp.append(A_o)
        omega_amp = np.asarray(omega_amp)
        omega_ratio = omega_amp / omega_amp[0] if omega_amp[0] != 0 else omega_amp
        ax.plot(r["T_grid"], omega_ratio, "o-", color=colors[r["sector"]], label="fixed-mu omega")
        ax.plot(r["T_grid"], r["pred_ratio"], "k--", label="LK scale")
        ax.set_xscale("log")
        ax.set_ylim(bottom=0)
        ax.set_title(f"rho={rho:+.3f}\n{r['sector']}")
        ax.set_xlabel("T")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("A(T)/A(Tmin)")
    axes[-1].legend(fontsize=8)
    fig.suptitle("Fixed-mu grand-potential benchmark")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_ROOT, "omega_fixed_mu_thermal_damping.png"), dpi=220)
    plt.close(fig)

    # Xp and RT by density at Tmin.
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for rho in rhos:
        r = results[rho]
        T0 = r["T_grid"][0]
        X = anomalous_Xp(T0, r["B_grid"], r["mu_bar"], p) if r["sector"] == "anomalous" else dispersive_Xp(T0, r["B_grid"], r["mu_bar"], "LL+", p)
        axes[0].plot(r["B_grid"], np.abs(X), color=colors[r["sector"]], lw=1.4, label=f"rho={rho:+.3f}")
        axes[1].plot(r["B_grid"], lk_damping(np.abs(X)), color=colors[r["sector"]], lw=1.4, label=f"rho={rho:+.3f}")
    axes[0].set_ylabel("|X1(B)| at Tmin")
    axes[1].set_ylabel("R_T")
    axes[1].set_xlabel("B")
    axes[0].set_title("Predicted local thermal arguments")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_ROOT, "Xp_RT_by_density.png"), dpi=220)
    plt.close(fig)

    # Upper versus lower Xp comparison.
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for rho in rhos:
        r = results[rho]
        T0 = r["T_grid"][0]
        if r["sector"] == "anomalous":
            ax.plot(r["B_grid"], np.abs(anomalous_Xp(T0, r["B_grid"], r["mu_bar"], p)),
                    color="#b02a30", lw=1.4, label=f"LL-/LLTR- rho={rho:+.3f}")
        else:
            ax.plot(r["B_grid"], np.abs(dispersive_Xp(T0, r["B_grid"], r["mu_bar"], "LL+", p)),
                    color="#1f6f3f", lw=1.8, label=f"LL+ rho={rho:+.3f}")
            ax.plot(r["B_grid"], standard_Xp(T0, r["B_grid"], p),
                    color="0.3", ls="--", lw=1.2, label="parabolic limit")
    ax.set_xlabel("B")
    ax.set_ylabel("|X1|")
    ax.set_title("Normal upper-branch versus anomalous lower-branch damping scale")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_ROOT, "upper_vs_lower_Xp_comparison.png"), dpi=220)
    plt.close(fig)

    # Normal-versus-anomalous overview.
    anom_key = next((rho for rho in rhos if results[rho]["sector"] == "anomalous"), rhos[0])
    disp_key = next((rho for rho in rhos if results[rho]["sector"] == "dispersive"), rhos[-1])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8))
    for row, rho in enumerate([disp_key, anom_key]):
        r = results[rho]
        axes[row, 0].plot(r["invB"], r["all_free_energy"][0], color=colors[r["sector"]], lw=1.0)
        axes[row, 0].set_ylabel("f")
        axes[row, 0].set_title(f"{r['sector']} rho={rho:+.3f}: free energy")
        axes[row, 1].plot(r["freqs"], r["spec"], color=colors[r["sector"]], lw=1.0)
        axes[row, 1].axvline(r["F_target"], color="#d95f02", ls="--", lw=1.2)
        axes[row, 1].axvline(r["F_peak"], color="#009e73", ls=":", lw=1.4)
        axes[row, 1].set_xlim(0, max(0.35, 2.5 * r["F_target"]))
        axes[row, 1].set_title("FFT")
        axes[row, 2].plot(r["T_grid"], r["amp_ratio"], "o-", color=colors[r["sector"]])
        axes[row, 2].plot(r["T_grid"], r["pred_ratio"], "k--")
        axes[row, 2].plot(r["T_grid"], r["std_ratio"], color="0.5", ls=":")
        axes[row, 2].set_xscale("log")
        axes[row, 2].set_title("thermal damping")
        for col in range(3):
            axes[row, col].grid(True, alpha=0.25)
    axes[1, 0].set_xlabel("1/B")
    axes[1, 1].set_xlabel("F")
    axes[1, 2].set_xlabel("T")
    axes[0, 2].set_ylabel("A/Amin")
    axes[1, 2].set_ylabel("A/Amin")
    fig.suptitle("Step 4 core comparison: normal dispersive LLs vs anomalous flat-band LLs")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_ROOT, "normal_vs_anomalous_thermo_overview.png"), dpi=220)
    plt.close(fig)

    # Compressibility diagnostic, retained only for code/spectrum sanity.
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.0), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, rho in zip(axes, rhos):
        r = results[rho]
        ax.plot(r["invB"], r["all_chi"][0], color=colors[r["sector"]], lw=1.0)
        ax.set_title(f"rho={rho:+.3f}")
        ax.set_xlabel("1/B")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$\partial\rho/\partial\mu$")
    fig.suptitle("Compressibility diagnostic only: not the LK thermal observable")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_ROOT, "compressibility_diagnostic.png"), dpi=220)
    plt.close(fig)

    # Forced standard-LK comparison.
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.8))
    for rho in rhos:
        r = results[rho]
        ax.scatter([rho], [r["m_eff"]], color=colors[r["sector"]], s=70)
        ax.text(rho, r["m_eff"], f" {rho:+.3f}", va="center", fontsize=8)
    ax.axhline(1.0 / (2.0 * p.a), color="k", ls="--", label="1/(2a)")
    ax.set_xlabel("rho")
    ax.set_ylabel("forced standard-LK m_eff")
    ax.set_title("Same standard-LK fit applied to both sectors")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_ROOT, "forced_standard_lk_comparison.png"), dpi=220)
    fig.savefig(os.path.join(FIG_ROOT, "dispersive_control.png"), dpi=220)
    plt.close(fig)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 4.1 LK thermal pipeline")
    parser.add_argument("--mode", choices=["v1", "main", "theory", "v2", "v3", "all"],
                        default="v1")
    parser.add_argument("--quick", action="store_true",
                        help="Run a smaller diagnostic density/B-grid set for the plan-compliant main mode.")
    args = parser.parse_args()

    rc = 0
    if args.mode in ("v1", "all"):
        rc |= run_v1()
    if args.mode in ("main", "all"):
        rc |= run_step4_plan(quick=args.quick)
    if args.mode in ("theory", "all"):
        rc |= run_step4_theory_tests(quick=args.quick)
    if args.mode in ("v2", "all"):
        rc |= run_v2()
    if args.mode in ("v3", "all"):
        rc |= run_v3()
    sys.exit(rc)


if __name__ == "__main__":
    main()
