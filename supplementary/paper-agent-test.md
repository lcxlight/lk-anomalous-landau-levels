# Paper-Agent v1.1.0 Candidate Smoke Test

Date: 2026-08-05  
Mode: real-publication next-version candidate, local reader test  
Method: local read/check session with the publication-staging tree as the
effective repository root; parent-working-tree files are not publication
sources

## Result

Passed local smoke validation for the `v1.1.0` candidate. The previous
independent `v1.0.0` frozen-tree reader test is superseded for this staging
tree by the updated manuscript/appendix sources and validation report.

## Representative Questions and Checked Answers

### 1. What is the paper and what is ground truth?

The paper is *Lifshitz--Kosevich Theory of Anomalous Landau Levels in
Topological Flat Bands* by Chao-Xing Liu. `paper/manuscript.tex`,
`paper/appendix.tex`, staged code/data, and shipped figures are primary ground
truth; supplementary reproduction reports and audits are secondary context.

### 2. What changed in v1.1.0?

The current manuscript and appendix were synchronized into the APP staging
tree. The main text now includes the stable APP repository URL. Appendix C.3
clarifies the fixed-\(\mu\) versus fixed-density interpretation of the
anomalous effective mass and uses \(|\rho_{\mathcal S}|\) in the local
fixed-density counting argument.

### 3. What scientific distinction should a reader retain?

Landau-level counting fixes the oscillation frequency and period, whereas the
local Fermi-level LL spacing controls the LK thermal-damping scale and apparent
effective mass. In the anomalous weak-field regime, the spacing is tied to the
quantum metric and gives `m_eff ~ 1/(B tr g)` along the fixed-density
trajectory.

### 4. What does the primary staged D.5 numerical evidence show?

Direct inspection of
`figures/appendix_numerics_fresh/appD5_lambda1_effective_mass_summary.csv`
gives:

| Window | Normal, `rho=+0.01` | Anomalous, `rho=-0.15` | Ratio |
|---|---:|---:|---:|
| W0 | 0.6258952435 | 5.9178239719 | 9.454975 |
| W1 | 0.7876785131 | 6.7330704219 | 8.547993 |
| W2 | 0.9575885300 | 10.9677073794 | 11.453466 |

The quick APP reproduction wrapper executed successfully and reported the
three plotted anomalous \(\rho=-0.15\) rows.

### 5. How are the equations checked?

The retained deeper Wolfram audit reports 26/26 checks passed. The new
Appendix C.3 effective-mass audit reports 17/17 checks passed, including the
\(|\rho_{\mathcal S}|\) counting convention, the fixed-density
\(\mu_\rho(B)-E_0\propto B\) scaling, and the contrast with the fixed-\(\mu\)
linear-in-\(B\) mass.

### 6. What remains before public release?

The remaining gate is author approval of this `v1.1.0` staged state and release
notes, followed by publication and verification of the commit, immutable
`v1.1.0` tag, GitHub Release, and `APP_PUBLICATION.json` manifest.

## Privacy and Path Check

A full release-file privacy scan was not rerun in this local pass. The
release-package sources should continue to avoid absolute user-directory paths;
generated logs from local builds are not part of the intended publication
source set.
