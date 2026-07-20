# Paper-Agent Final Frozen-Tree Smoke Test

Date: 2026-07-18  
Mode: real-publication candidate, final pre-freeze reader test  
Method: fresh independent read-only agent session with the publication-staging
tree as its effective repository root; parent-working-tree files were excluded

## Result

Passed after two release-record corrections identified by the fresh session:
the prior validation/smoke reports were stale, and the global `*.out` ignore
rule would have omitted the retained Wolfram audit output. The reports were
refreshed and the audit output was explicitly unignored. No scientific source,
numerical result, or figure artifact changed after the reader test.

## Representative questions and checked answers

### 1. What is the paper and what is ground truth?

The paper is *Lifshitz--Kosevich Theory of Anomalous Landau Levels in
Topological Flat Bands* by Chao-Xing Liu. `paper/manuscript.tex`,
`paper/appendix.tex`, staged code/data, and shipped figures are primary ground
truth; supplementary reproduction reports and audits are secondary context.

### 2. What scientific distinction should a reader retain?

Landau-level counting fixes the oscillation frequency and period, whereas the
local Fermi-level LL spacing controls the LK thermal-damping scale and apparent
effective mass. In the anomalous weak-field regime, the spacing is tied to the
quantum metric and gives `m_eff ~ 1/(B tr g)`.

### 3. What does the primary staged D.5 numerical evidence show?

Direct inspection of
`figures/appendix_numerics_fresh/appD5_lambda1_effective_mass_summary.csv`
gave:

| Window | Normal, `rho=+0.01` | Anomalous, `rho=-0.15` | Ratio |
|---|---:|---:|---:|
| W0 | 0.6258952435 | 5.9178239719 | 9.454975 |
| W1 | 0.7876785131 | 6.7330704219 | 8.547993 |
| W2 | 0.9575885300 | 10.9677073794 | 11.453466 |

The corrected PowerShell quick check in
`code/figure-reproduction/README.md` executed successfully and reported these
six values and three ratios.

### 4. How does a reader reproduce the figures, and what is limited?

The pinned Windows runner and setup are in `environment/README.md`. The grouped
main-figure command and exact outputs, plus the Appendix B/D commands, are in
`code/figure-reproduction/README.md`. The two appendix tables are correctly
marked `manual-only` because their CSV-generated values are manually typeset.
The full Appendix D chain is materially heavier and can be path-sensitive in a
deeply synchronized Windows checkout; a short local checkout is recommended.

### 5. What truly remains before public release?

The repository URL, `v1.0.0` tag, approved license, public empty repository,
and connector access are resolved. The remaining gate is author approval of
this frozen staged state and release notes, followed by publication and
verification of the commit, immutable tag, GitHub Release, and
`APP_PUBLICATION.json` manifest.

## Privacy and path check

The session found no credentials, tokens, email addresses, private URLs, or
absolute user-directory paths. All cited reader paths resolve inside the
staging tree.
