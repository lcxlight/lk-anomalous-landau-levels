# Deeper Equation Audit

Date: 2026-07-03  
Staging-path normalization: 2026-07-18

## Inputs

- `paper/manuscript.tex`;
- `paper/appendix.tex`;
- `supplementary/reproduction/check_deeper_equation_audit.wls`.

## Result

The Wolfram audit passed 26/26 checks. The retained output is
`supplementary/reproduction/deeper-equation-audit-wolfram.out`.

The audit covers the THF characteristic polynomial and ideal flat-band
factorization; eigenvectors and interband matrix elements; quantum metric,
Berry curvature, and ideal-geometry trace condition; integrated metric; LL and
LLTR block spectra; gap asymptotics; branch inversion derivatives and local
spacings; anomalous effective masses; LK finite-temperature prefactor algebra;
fixed-density Helmholtz derivative cancellation; and representative numerical
anchors.

In particular, `paper/appendix.tex` contains the corrected LLTR- effective-mass
term

```latex
\frac{1}{2a} + \frac{c^{2}B}{2a(\mu-E_0)^{2}}.
```

The audit is secondary validation evidence. It checks the major displayed
derivation chains supporting the paper but does not replace the canonical
paper source or claim to prove numerical-fit protocol definitions.
