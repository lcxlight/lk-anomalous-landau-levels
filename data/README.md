# Data

The publication ships small figure-level numerical CSVs beside their canonical
figures. No external dataset is required for the default checks.

| Dataset | Produced by | Used by |
|---|---|---|
| `figures/appendix_numerics_fresh/appD1_lambda1_deltaOmega_M_wide_metrics.csv` | `code/figure-reproduction/Codes/make_appendix_figD1_lambda1_deltaOmega_M.py` | Appendix D.1 diagnostics |
| `figures/appendix_numerics_fresh/appD2_lambda1_magnetization_fft_peaks.csv` | `code/figure-reproduction/Codes/make_appendix_figD2_lambda1_magnetization_fft.py` | Main Fig. 2 and Appendix Fig. D.2 frequency checks |
| `figures/appendix_numerics_fresh/appD3_lambda1_zero_aligned_windows.csv` | `code/figure-reproduction/Codes/make_appendix_figD3_lambda1_zero_aligned_windows.py` | Appendix Fig. D.3, local-window table, and D.4 extraction |
| `figures/appendix_numerics_fresh/appD4_lambda1_window_p1_damping.csv` | `code/figure-reproduction/Codes/make_appendix_figD4_lambda1_window_damping.py` | Main Fig. 3 and Appendix Fig. D.4 |
| `figures/appendix_numerics_fresh/appD4_lambda1_window_p1_fit_summary.csv` | same D.4 script | D.5 effective-mass extraction |
| `figures/appendix_numerics_fresh/appD5_lambda1_effective_mass_summary.csv` | `code/figure-reproduction/Codes/make_appendix_figD5_lambda1_effective_mass.py` | Main Fig. 4, Appendix Fig. D.5, and fitted-mass table |
| `figures/spin_counting_diagnostics/electron_LL_near_fermi_summary_dense_public.csv` | staged electron LL diagnostic pipeline | electron chemical-potential/nearest-level-spacing figure |

The main-figure wrapper reads D.4 and D.5 directly from these canonical
locations. Superseded duplicate processed inputs are excluded. Large
exploratory sweeps, caches, backups, and private session materials are not part
of the publication.

For the headline numerical check, run the effective-mass command in
`code/figure-reproduction/README.md`; it reports the six fitted values and the
three approximately order-of-magnitude ratios directly from the staged D.5
CSV.
