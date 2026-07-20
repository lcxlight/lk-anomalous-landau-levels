param(
    [string]$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path,
    [string]$PythonExe = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$required = @(
    "AGENTS.md",
    "paper/manuscript.tex",
    "paper/appendix.tex",
    "code/figure-reproduction/Codes/make_manuscript_figures.py",
    "figures/appendix_numerics_fresh/appD4_lambda1_window_p1_damping.csv",
    "figures/appendix_numerics_fresh/appD5_lambda1_effective_mass_summary.csv"
)

Push-Location $Root
try {
    foreach ($path in $required) {
        if (-not (Test-Path -LiteralPath $path)) {
            throw "Missing required staged artifact: $path"
        }
    }

    & $PythonExe -c "import numpy, scipy, matplotlib; print('Python dependencies: PASS')"
    if ($LASTEXITCODE -ne 0) { throw "Python dependency check failed" }

    Push-Location "code/figure-reproduction"
    try {
        & $PythonExe "Codes/make_manuscript_figures.py"
        if ($LASTEXITCODE -ne 0) { throw "Main-figure reproduction failed" }
    }
    finally {
        Pop-Location
    }

    if (Get-Command wolframscript -ErrorAction SilentlyContinue) {
        wolframscript -file "supplementary/reproduction/check_deeper_equation_audit.wls"
        if ($LASTEXITCODE -ne 0) { throw "Wolfram equation audit failed" }
    }
    else {
        Write-Warning "wolframscript not found; symbolic audit was not rerun"
    }

    $massRows = Import-Csv "figures/appendix_numerics_fresh/appD5_lambda1_effective_mass_summary.csv" |
        Where-Object { $_.rho -eq "-0.15" -and $_.plotted -eq "True" } |
        Select-Object rho,window,Bbar,m_eff_fit,m_eff_theory
    if ($massRows.Count -ne 3) {
        throw "Expected three plotted rho=-0.15 effective-mass rows"
    }
    $massRows | Format-Table -AutoSize
}
finally {
    Pop-Location
}
