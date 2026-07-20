# APP Final Frozen-Tree Validation Report

Date: 2026-07-18  
Stage: `full`, final pre-release validation  
Effective repository root: the publication-staging tree  
Result: **passed**

## Passed checks

- Required APP root files and the canonical `paper/`, `code/`, `data/`,
  `environment/`, `figures/`, and `supplementary/` content are present.
- `AGENTS.md`, README, metadata, citation, repository URL, `v1.0.0` tag, and
  version `1.0.0` agree. The empty public repository and connector write access
  were independently confirmed.
- The main manuscript source and bibliography are byte-identical to the author
  sources. The appendix differs only by the six documented staging-local figure
  redirects to canonical public artifact paths.
- Both LaTeX documents rebuilt from an isolated temporary copy with exit code
  zero and no undefined references, undefined citations, missing figures, or
  fatal errors.
- The figure/table map covers main Figs. 1--4, Appendix Figs. B.1 and D.1--D.5,
  the electron LL diagnostic, and both appendix tables. It uses 11
  `reproduced` and two `manual-only` final statuses with explicit scripts,
  inputs, outputs, and limitations.
- The grouped main-figure wrapper regenerated all eight expected PDF/PNG
  outputs in an isolated temporary tree. The electron diagnostic regenerated
  its PDF and PNG from the staged dense CSV.
- The primary D.5 quick check directly reported normal masses `0.625895`,
  `0.787679`, `0.957589`, anomalous masses `5.917824`, `6.733070`, `10.967707`,
  and paired ratios `9.454975`, `8.547993`, `11.453466`.
- All 18 Python files passed AST parsing; the PowerShell reproduction wrapper
  parsed without errors. The tested Python versions match the pinned
  environment specification.
- The Wolfram symbolic audit was rerun and passed 26/26 checks. Its retained
  `.out` evidence is explicitly included despite the general build-output
  ignore rule.
- All documented staging paths resolve. The Git-ignore-respecting publication
  set contains 74 readable files; undocumented legacy/generated output trees
  and broken OneDrive placeholders are excluded.
- Full release-file privacy screening found no credentials, tokens, email
  addresses, private/internal URLs, private IP addresses, or absolute user
  directory paths.
- The approved root license clearly applies MIT to software/scripts and CC BY
  4.0 to all non-code publication content.
- A fresh independent staging-root reader-agent session passed identity,
  scientific-summary, primary-number, reproduction-command, limitation,
  release-metadata, and privacy checks. Its record is
  `supplementary/paper-agent-test.md`.

## Corrections made during final validation

- Corrected the D.5 quick-check field from the nonexistent `window_label` to
  `window` and made the command calculate the three advertised ratios.
- Replaced an unavailable OneDrive placeholder input with the readable,
  hash-matched staged file
  `figures/spin_counting_diagnostics/electron_LL_near_fermi_summary_dense_public.csv`
  and updated its reader/script pointers.
- Updated stale repository-creation language after confirming the public empty
  repository and connector access.
- Excluded undocumented legacy default-output trees and explicitly unignored
  `supplementary/reproduction/deeper-equation-audit-wolfram.out`.

## Issues needing changes

Errors: none.  
Warnings: none.

## Manual verification and scope notes

- The materially heavier full Appendix D chain was not rerun during this final
  pass. Its prior successful reproduction, canonical CSVs/figures, and retained
  evidence were rechecked; the lightweight main wrapper and direct numerical
  audits were rerun.
- `APP_PUBLICATION.json` is correctly absent during staging. It can be created
  only after the public commit and tree hashes exist.

## Public-release gate

No APP compliance blocker remains in the staged content. Before any remote
publication action, the author must approve this frozen state and release
notes. The release workflow must then publish and verify the commit, immutable
`v1.0.0` tag, GitHub Release, and `APP_PUBLICATION.json` asset.

## Canonical artifact fingerprints (SHA-256)

- `paper/manuscript.tex`: `56BB7B2C843A9BD3C8898C2D9B8CB34916740AFF3202951616263705CC472FA1`
- `paper/manuscript.pdf`: `44808E96117EF3BAF3037AEE36AC09CB37424E4561A8C5E1F3C3D79AC63A7232`
- `paper/appendix.tex`: `92E47FEC1A1EE578EC05B2AB05C978A710FA53AD044186307471F8EA4F562704`
- `paper/appendix.pdf`: `8ED4E2EB922912681893EDAAB65FF584E827D53BC477A221E9F1334B4C8DD38C`
- `paper/refs.bib`: `FA9B14497D764D0762416B3FB574A022C2D50BEA730A74C899BD4EBF3D404FCE`
- D.5 CSV: `9378362E9651F712B3583EA953469F1F643F0E6304470BC74706E671773B3AA3`
