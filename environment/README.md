# Environment

## Tested platform

The July 2026 release candidate was tested on Windows 11 with:

- Python 3.12.7;
- NumPy 1.26.4;
- SciPy 1.13.1;
- Matplotlib 3.9.2;
- MiKTeX 25.12 / pdfTeX 4.23;
- latexmk 4.88;
- WolframScript 1.13.0 for the supplementary symbolic audit.

The canonical Python dependency specification is `environment/requirements.txt`.

## Setup from the repository root

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r environment\requirements.txt
```

Use `.\.venv\Scripts\python.exe` as the Python runner prefix in a newly
created environment. Installed environments and caches are excluded by
`.gitignore`.

## Checks and builds

```powershell
.\.venv\Scripts\python.exe -c "import numpy, scipy, matplotlib; print('dependencies available')"
.\.venv\Scripts\python.exe code\figure-reproduction\Codes\make_manuscript_figures.py

cd paper
latexmk -pdf manuscript.tex
latexmk -pdf appendix.tex
```

The Python dependency import and main-figure wrapper passed with the tested
versions above. Both paper builds passed. WolframScript is optional for reading
and reproducing the Python figures, but is required to rerun the supplementary
symbolic audit. Wolfram Language installation and licensing are managed outside
this repository.

Some full Appendix D reruns are path-sensitive on Windows. Use a short local
checkout if the repository is otherwise nested under a deeply synchronized
directory.
