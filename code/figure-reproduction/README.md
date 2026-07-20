# Figure and Table Reproduction

Run commands from the publication repository root. The Python scripts write
directly to the canonical staged artifact paths under `figures/`.

## Environment

Create the pinned environment described in `environment/README.md`, then use
`.\.venv\Scripts\python.exe` as the runner prefix on Windows.

## Main-figure grouped wrapper

```powershell
.\.venv\Scripts\python.exe code\figure-reproduction\Codes\make_manuscript_figures.py
```

`make_manuscript_figures.py` is an intentional grouped wrapper for main Figs.
1--4. It reads the accepted Appendix D.4/D.5 CSVs directly from
`figures/appendix_numerics_fresh/` and writes all PDF/PNG outputs to
`figures/manuscript figures/`.

| Figure/Table | Paper artifact | Script | Inputs | Generated output | Status | Notes |
|---|---|---|---|---|---|---|
| Main Fig. 1 | `figures/manuscript figures/main_fig1_ideal_flatband_LL_spectrum.pdf` | `code/figure-reproduction/Codes/make_manuscript_figures.py` | `code/figure-reproduction/Codes/ideal_flatband_model.py` | same PDF plus PNG | `reproduced` | Included in the grouped main-figure wrapper. |
| Main Fig. 2 | `figures/manuscript figures/main_fig2_normal_anomalous_magnetization_oscillations.pdf` | `code/figure-reproduction/Codes/make_manuscript_figures.py` | staged model/response modules under `code/figure-reproduction/Codes/` | same PDF plus PNG | `reproduced` | Included in the grouped main-figure wrapper. |
| Main Fig. 3 | `figures/manuscript figures/main_fig3_thermal_damping_local_windows.pdf` | `code/figure-reproduction/Codes/make_manuscript_figures.py` | `figures/appendix_numerics_fresh/appD4_lambda1_window_p1_damping.csv` | same PDF plus PNG | `reproduced` | Uses current D.4 production data directly. |
| Main Fig. 4 | `figures/manuscript figures/main_fig4_effective_mass_normal_anomalous.pdf` | `code/figure-reproduction/Codes/make_manuscript_figures.py` | `figures/appendix_numerics_fresh/appD5_lambda1_effective_mass_summary.csv` | same PDF plus PNG | `reproduced` | Uses current D.5 production data directly. |

## Appendix figures and tables

These commands write to the exact public artifact directories shown below:

```powershell
.\.venv\Scripts\python.exe code\figure-reproduction\Codes\slide12_gap_b2_scaling.py
.\.venv\Scripts\python.exe code\figure-reproduction\Codes\make_appendix_figD1_lambda1_deltaOmega_M.py
.\.venv\Scripts\python.exe code\figure-reproduction\Codes\make_appendix_figD2_lambda1_magnetization_fft.py
.\.venv\Scripts\python.exe code\figure-reproduction\Codes\make_appendix_figD3_lambda1_zero_aligned_windows.py
.\.venv\Scripts\python.exe code\figure-reproduction\Codes\make_appendix_figD4_lambda1_window_damping.py
.\.venv\Scripts\python.exe code\figure-reproduction\Codes\make_appendix_figD5_lambda1_effective_mass.py
.\.venv\Scripts\python.exe code\figure-reproduction\Codes\make_electron_mu_splitting_comparison.py
```

| Figure/Table | Paper artifact | Script | Inputs | Generated output | Status | Notes |
|---|---|---|---|---|---|---|
| App. Fig. B.1 | `figures/appendix_numerics_fresh/slide12_gap_b2_scaling.png` | `code/figure-reproduction/Codes/slide12_gap_b2_scaling.py` | model and analytical-gap modules in the same `Codes/` directory | same PNG | `reproduced` | Low-field slopes and asymptotic ratios were checked in the reproduction report. |
| App. Fig. D.1 | `figures/appendix_numerics_fresh/appD1_lambda1_deltaOmega_M_wide.pdf` | `code/figure-reproduction/Codes/make_appendix_figD1_lambda1_deltaOmega_M.py` | fixed-density model/response modules | PDF, PNG, and `appD1_lambda1_deltaOmega_M_wide_metrics.csv` | `reproduced` | Exact LL-sum thermodynamic traces. |
| App. Fig. D.2 | `figures/appendix_numerics_fresh/appD2_lambda1_magnetization_fft.pdf` | `code/figure-reproduction/Codes/make_appendix_figD2_lambda1_magnetization_fft.py` | D.1 response pipeline | PDF, PNG, and `appD2_lambda1_magnetization_fft_peaks.csv` | `reproduced` | FFT peak CSV is the direct counting-frequency evidence. |
| Electron LL diagnostic | `figures/spin_counting_diagnostics/electron_mu_and_near_fermi_splitting_comparison.pdf` | `code/figure-reproduction/Codes/make_electron_mu_splitting_comparison.py` | staged dense nearest-Fermi CSV and fixed-density solver | same PDF plus PNG | `reproduced` | Supports, without uniquely proving, the spin-degenerate-like interpretation. |
| App. Fig. D.3 | `figures/appendix_numerics_fresh/appD3_lambda1_zero_aligned_windows.pdf` | `code/figure-reproduction/Codes/make_appendix_figD3_lambda1_zero_aligned_windows.py` | D.1 response pipeline | PDF, PNG, and `appD3_lambda1_zero_aligned_windows.csv` | `reproduced` | Defines W0--W2. |
| Table `tab:local-window-periods` | rendered in `paper/appendix.pdf` | `code/figure-reproduction/Codes/make_appendix_figD3_lambda1_zero_aligned_windows.py` | `figures/appendix_numerics_fresh/appD3_lambda1_zero_aligned_windows.csv` | manually typeset rows in `paper/appendix.tex` | `manual-only` | Script generates the source CSV; the compact table is manually transcribed and was cross-checked against it. |
| App. Fig. D.4 | `figures/appendix_numerics_fresh/appD4_lambda1_window_p1_damping_Rnum.pdf` | `code/figure-reproduction/Codes/make_appendix_figD4_lambda1_window_damping.py` | D.3 window CSV and response pipeline | PDF/PNG damping figures plus damping and fit-summary CSVs | `reproduced` | W2 for two unreliable hole cases is intentionally omitted. |
| Table `tab:window-meff-fit-summary` | rendered in `paper/appendix.pdf` | `code/figure-reproduction/Codes/make_appendix_figD5_lambda1_effective_mass.py` | `figures/appendix_numerics_fresh/appD5_lambda1_effective_mass_summary.csv` | manually typeset rows in `paper/appendix.tex` | `manual-only` | Script generates the source CSV; the table is manually transcribed and was cross-checked against it. |
| App. Fig. D.5 | `figures/appendix_numerics_fresh/appD5_lambda1_effective_mass_summary.pdf` | `code/figure-reproduction/Codes/make_appendix_figD5_lambda1_effective_mass.py` | `figures/appendix_numerics_fresh/appD4_lambda1_window_p1_fit_summary.csv` | same PDF plus PNG and effective-mass CSV | `reproduced` | Direct source of the reported fitted masses. |

The Appendix D chain is materially heavier than the grouped main-figure
wrapper. On Windows, use a short local checkout if a deeply synchronized path
causes filesystem placeholder or path-length failures.

## Quick claim checks

### 1. Paper builds

Paper anchors: all main and appendix figures/tables. Evidence level: locally
reproduced on the tested platform.

```powershell
cd paper
latexmk -pdf manuscript.tex
latexmk -pdf appendix.tex
```

Expected signature: both builds complete without undefined references or
citations.

### 2. Representative effective-mass enhancement

Paper anchors: main Fig. 4 and Appendix Fig. D.5/Table
`tab:window-meff-fit-summary`. Direct artifact:
`figures/appendix_numerics_fresh/appD5_lambda1_effective_mass_summary.csv`.

```powershell
$rows = Import-Csv 'figures/appendix_numerics_fresh/appD5_lambda1_effective_mass_summary.csv' |
  Where-Object { $_.rho -in @('0.01','-0.15') -and $_.plotted -eq 'True' }
$rows | Sort-Object rho,window | Select-Object rho,window,m_eff_fit,branch_theory
foreach ($window in @('W0','W1','W2')) {
  $normal = $rows | Where-Object { $_.rho -eq '0.01' -and $_.window -eq $window }
  $anomalous = $rows | Where-Object { $_.rho -eq '-0.15' -and $_.window -eq $window }
  [pscustomobject]@{
    window = $window
    normal = [double]$normal.m_eff_fit
    anomalous = [double]$anomalous.m_eff_fit
    ratio = [double]$anomalous.m_eff_fit / [double]$normal.m_eff_fit
  }
}
```

Expected fitted masses: normal `0.625895`, `0.787679`, `0.957589`; anomalous
`5.917824`, `6.733070`, `10.967707`. Paired ratios are approximately `9.45`,
`8.55`, and `11.45`. Evidence level: direct staged production CSV.

### 3. Active-density FFT frequencies

Paper anchors: main Fig. 2 and Appendix Fig. D.2. Direct artifact:
`figures/appendix_numerics_fresh/appD2_lambda1_magnetization_fft_peaks.csv`.

```powershell
Import-Csv 'figures/appendix_numerics_fresh/appD2_lambda1_magnetization_fft_peaks.csv' |
  Select-Object rho,frequency_expected,deltaOmega_frequency_peak,M_no0LL_frequency_peak
```

Expected counting-frequency entries include `0.1256637061`, `0.0628318531`,
`0.1203540570`, and `0.0575222039`. Evidence level: direct staged production
CSV; finite-window peak positions need not equal the reference value exactly.

### 4. Symbolic derivation audit

Paper anchors: displayed LL-spacing, effective-mass, LK-prefactor, and
fixed-density derivative equations. Direct artifacts:
`supplementary/reproduction/deeper-equation-audit.md` and
`supplementary/reproduction/deeper-equation-audit-wolfram.out`.

```powershell
rg -n "checks=26|failures=0|PASS" supplementary/reproduction/deeper-equation-audit-wolfram.out
```

Expected signature: `checks=26`, `failures=0`, and 26 `PASS` entries. Evidence
level: locally reproduced Wolfram audit.
