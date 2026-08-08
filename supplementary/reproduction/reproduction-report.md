# Staged Reproduction Report

Initial reproduction: 2026-07-03 to 2026-07-17  
Staging-path refresh: 2026-08-05  
Effective repository root: the publication-staging tree

This report records the reproduction evidence retained in the public APP
candidate. The paper, staged code, and staged data remain ground truth; this
report is secondary validation context.

## Canonical sources and entry points

- main source and PDF: `paper/manuscript.tex`, `paper/manuscript.pdf`;
- appendix source and PDF: `paper/appendix.tex`, `paper/appendix.pdf`;
- bibliography: `paper/refs.bib`;
- figure map and commands: `code/figure-reproduction/README.md`;
- pinned Python environment: `environment/requirements.txt`;
- staging-root quick wrapper: `supplementary/reproduction/run_reproduction_checks.ps1`.

## Reproduction results

| Scope | Status | Evidence |
|---|---|---|
| Main Figs. 1--4 grouped wrapper | `reproduced` | Regenerated PDFs/PNGs under `figures/manuscript figures/`. The wrapper reads current D.4/D.5 CSVs directly. |
| Appendix Fig. B.1 | `reproduced` | `figures/appendix_numerics_fresh/slide12_gap_b2_scaling.png`; low-field slopes are near 2 and asymptotic ratios range from about 0.976 to 0.999. |
| Appendix Figs. D.1--D.5 | `reproduced` | Figures and figure-level CSVs under `figures/appendix_numerics_fresh/`; regenerated CSVs matched the retained production files. |
| Electron nearest-Fermi LL diagnostic | `reproduced` | Figure and CSV evidence under `figures/spin_counting_diagnostics/`. |
| Appendix local-window and fitted-mass tables | `manual-only` | Rows are manually typeset in `paper/appendix.tex` and were cross-checked against the staged D.3 and D.5 CSVs. |
| Main and appendix LaTeX builds | `reproduced` | Both staged PDFs built without undefined references, undefined citations, missing figures, or fatal errors. |
| Deeper symbolic equation audit | `reproduced` | `supplementary/reproduction/deeper-equation-audit-wolfram.out` reports 26/26 checks passed. |
| Appendix C.3 effective-mass audit | `reproduced` | `supplementary/reproduction/check_appendix_C3_effective_mass.out` reports 17/17 checks passed. |

## Primary numerical evidence

For the representative normal density `rho=+0.01`, the D.5 fitted masses are
`0.6258952435`, `0.7876785131`, and `0.9575885300`. For the representative
anomalous density `rho=-0.15`, they are `5.9178239719`, `6.7330704219`, and
`10.9677073794`. The paired ratios are approximately `9.45`, `8.55`, and
`11.45`, consistent with the manuscript's approximately-one-order-of-magnitude
comparison.

The D.2 FFT CSV contains the active-density counting references
`0.1256637061`, `0.0628318531`, `0.1203540570`, and `0.0575222039`. Measured
finite-window peaks may be offset from the reference values; the paper's claim
is that the observed peaks agree with or are consistent with the counting
frequency at the resolution of the displayed analysis.

## Tested environment

The retained reproduction evidence was produced on Windows with Python 3.12.7,
NumPy 1.26.4, SciPy 1.13.1, Matplotlib 3.9.2, MiKTeX 25.12, latexmk 4.88, and
WolframScript 1.13.0. Exact setup and runner commands are in
`environment/README.md`.

## Scope and limitations

- Full Appendix D regeneration is more expensive than the main-figure wrapper
  and can be path-sensitive on deeply synchronized Windows checkouts.
- The paper uses cached, staged production CSVs for lightweight reader checks;
  those checks are direct numerical audits but not substitutes for every heavy
  end-to-end rerun.
- Chat/session history, exploratory sweeps, backups, abandoned drafts, caches,
  and parent working-repository history are excluded by author request.
- The manuscript now contains the stable APP repository URL in its data and
  code availability note. Version-specific release URLs are carried by the
  repository metadata and citation.

## Release metadata status

The intended target is
`https://github.com/lcxlight/lk-anomalous-landau-levels` at tag `v1.1.0`.
The author approved MIT for code and CC BY 4.0 for all non-code publication
content on 2026-07-18. Final full validation, author freeze approval, and the
verified release manifest remain required before publication.
