# APP v1.1.0 Candidate Validation Report

Date: 2026-08-05  
Stage: `next-version-candidate`, local validation  
Effective repository root: the publication-staging tree  
Result: **passed**

## Passed Checks

- The current author manuscript sources were synchronized into `paper/`:
  `manuscript.tex`, `appendix.tex`, `refs.bib`, and current `.bbl` files.
- The appendix was kept APP-local by redirecting the six staged appendix
  figure paths to `figures/appendix_numerics_fresh/`.
- README, `PUBLICATION_METADATA.md`, `AGENTS.md`, and citation examples now
  agree on the stable repository
  `https://github.com/lcxlight/lk-anomalous-landau-levels` and release tag
  `v1.1.0`; `AGENTS.md` records APP package version `1.1.0`.
- The staged main manuscript rebuilt with `pdflatex` twice, with no undefined
  citations, undefined references, missing figures, or fatal errors.
- The staged appendix rebuilt with `pdflatex` twice, with no undefined
  citations, undefined references, missing figures, or fatal errors.
- The quick APP reproduction wrapper
  `supplementary/reproduction/run_reproduction_checks.ps1` passed.
- Python dependency import check passed.
- The grouped main-figure wrapper regenerated the main figure PDFs/PNGs under
  `figures/manuscript figures/`.
- The retained deeper Wolfram symbolic audit reran and passed 26/26 checks.
- The new Appendix C.3 effective-mass Wolfram audit reran and passed 17/17
  checks, including the \(|\rho_{\mathcal S}|\) fixed-density convention.
- The D.5 quick check again found three plotted \(\rho=-0.15\) anomalous
  effective-mass rows:
  `5.9178239719`, `6.7330704219`, and `10.9677073794`.

## v1.1.0 Scientific/Text Updates

- Main text now includes the stable APP repository URL in a data and code
  availability note.
- Appendix C.3 now makes the fixed-density ensemble point explicit: at fixed
  \(\mu\), the anomalous bulk mass scales as \(B\), while along the
  fixed-density trajectory \(\mu_\rho(B)-E_0\propto B\), giving
  \(m^*_{\mathrm{eff}}\propto 1/B\).
- Appendix C.3 now writes the active-density counting formulas with
  \(|\rho_{\mathcal S}|\), making the positivity convention explicit before
  the detailed sector definition in Appendix D.

## Issues Needing Changes

Errors: none.  
Warnings: none blocking. The LaTeX logs retain ordinary overfull/underfull box
messages and the existing REVTeX float-placement warning, but no unresolved
references/citations or fatal errors.

## Scope Notes

- This was a local next-version validation, not a remote publication action.
- The heavy full Appendix D numerical regeneration was not rerun; retained
  staged D.4/D.5 CSV evidence and the quick wrapper checks were reused.
- `APP_PUBLICATION.json` remains absent during staging. It should be created
  only after a public commit/tag/release exists.

## Public-Release Gate

Before remote publication, the author should approve this `v1.1.0` staged
state and release notes. The release workflow must then publish and verify the
commit, immutable `v1.1.0` tag, GitHub Release, and `APP_PUBLICATION.json`
asset.

## Canonical Artifact Fingerprints (SHA-256)

- `paper/manuscript.tex`: `FDC4A73C365671C40A3F145598D44EF4AF70539A5ED800AA28D5A8BF14F1D01B`
- `paper/manuscript.pdf`: `6CD96A97A93D52BB033E4ACC2CAAACA46B92CBE47D4E441D54DE885E3E53C975`
- `paper/appendix.tex`: `1B839B8022234F261AD1974FD11845E47AB26028722A0E728690B517BE1506A5`
- `paper/appendix.pdf`: `5D5ED4E04AC0E5F3C4A64442B9762D391784EA6677C24D599A5EC0845BB774AF`
- `paper/refs.bib`: `FA9B14497D764D0762416B3FB574A022C2D50BEA730A74C899BD4EBF3D404FCE`
- D.5 CSV: `9378362E9651F712B3583EA953469F1F643F0E6304470BC74706E671773B3AA3`
