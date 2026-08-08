# Lifshitz--Kosevich Theory of Anomalous Landau Levels in Topological Flat Bands

Chao-Xing Liu  
Department of Physics and Center for Theory of Emergent Quantum Matter, The Pennsylvania State University

This paper develops a Lifshitz--Kosevich theory for anomalous Landau levels in topological flat bands and shows how their thermal damping can probe quantum geometry.

Status: real-publication next-version release candidate; final validation and freeze pending  
Staging created: 2026-07-03  
Chat history: excluded by author request  
Public repository URL: https://github.com/lcxlight/lk-anomalous-landau-levels  
Release tag: v1.1.0  
License: MIT for code; CC BY 4.0 for all non-code publication content

## Talk to this paper

This staging tree is the real-publication next-version candidate for the Agentic Publication Protocol. Open this folder in an AI coding agent that reads `AGENTS.md` to ask questions about the paper, inspect equations, check staged data, or reproduce figures.

This `v1.1.0` candidate incorporates the current manuscript and appendix sources, including the APP repository availability note and the fixed-density effective-mass clarification in Appendix C.3. It becomes a verified APP publication only after the author approves the final validation, the `v1.1.0` tag and GitHub Release are published, and the release manifest is verified.

## Paper

- Main manuscript: `paper/manuscript.tex` and `paper/manuscript.pdf`
- Appendix: `paper/appendix.tex` and `paper/appendix.pdf`
- Bibliography: `paper/refs.bib`

## Layout

- `paper/`: canonical publication manuscript and appendix sources, bibliography files, and current PDFs.
- `figures/`: figures referenced by the manuscript and appendix.
- `code/figure-reproduction/`: focused scripts for regenerating main and appendix figures.
- `data/`: location and policy for staged figure-level numerical evidence.
- `environment/`: software and execution notes.
- `supplementary/`: reproduction report, deeper equation audit, Wolfram checks, and non-chat audit artifacts.
- `AGENTS.md`: root paper-agent instructions for reader agents.
- `CLAUDE.md`: delegates Claude Code to `AGENTS.md`.

## Reproducing results

Use `environment/README.md` for setup notes and `code/figure-reproduction/README.md` for the figure map, commands, quick checks, inputs, and expected outputs. The staged reproduction report is in `supplementary/reproduction/reproduction-report.md`.

## Validation

The current staging-root paper-agent smoke test is recorded in `supplementary/paper-agent-test.md`.
The current full local APP validation report is recorded in `supplementary/validation-report.md`.

## Citation

```bibtex
@misc{liu2026lk_anomalous_flatbands,
  title = {Lifshitz--Kosevich Theory of Anomalous Landau Levels in Topological Flat Bands},
  author = {Liu, Chao-Xing},
  year = {2026},
  url = {https://github.com/lcxlight/lk-anomalous-landau-levels/releases/tag/v1.1.0},
  note = {APP publication, version 1.1.0}
}
```

## License

See `LICENSE` for the approved component-specific terms: MIT for code and CC BY 4.0 for all non-code publication content.

## Required Before Public Release

- Complete the final full APP validation and author freeze approval for `v1.1.0`.
- Publish and verify the immutable tag, GitHub Release, and `APP_PUBLICATION.json` asset.
