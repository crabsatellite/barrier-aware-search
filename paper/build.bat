@echo off
setlocal
cd /d "%~dp0"

echo === Building ECJ version ===
pdflatex -interaction=nonstopmode covariance_drift_ecj.tex >nul 2>&1
bibtex covariance_drift_ecj >nul 2>&1
pdflatex -interaction=nonstopmode covariance_drift_ecj.tex >nul 2>&1
pdflatex -interaction=nonstopmode covariance_drift_ecj.tex

echo === Building JMLR version (blind) ===
pdflatex -interaction=nonstopmode -jobname=Anonymous_Covariance_Drift_2026 covariance_drift.tex >nul 2>&1
bibtex Anonymous_Covariance_Drift_2026 >nul 2>&1
pdflatex -interaction=nonstopmode -jobname=Anonymous_Covariance_Drift_2026 covariance_drift.tex >nul 2>&1
pdflatex -interaction=nonstopmode -jobname=Anonymous_Covariance_Drift_2026 covariance_drift.tex

echo === Building JMLR version (public) ===
pdflatex -interaction=nonstopmode -jobname=Li_Covariance_Drift_2026 "\def\publicmode{}\input{covariance_drift}" >nul 2>&1
bibtex Li_Covariance_Drift_2026 >nul 2>&1
pdflatex -interaction=nonstopmode -jobname=Li_Covariance_Drift_2026 "\def\publicmode{}\input{covariance_drift}" >nul 2>&1
pdflatex -interaction=nonstopmode -jobname=Li_Covariance_Drift_2026 "\def\publicmode{}\input{covariance_drift}"

echo === Cleaning auxiliary files ===
del /q *.aux *.bbl *.blg *.log *.out *.toc 2>nul

echo === Done ===
echo   covariance_drift_ecj.pdf  (ECJ submission)
echo   Anonymous_Covariance_Drift_2026.pdf  (JMLR blind)
echo   Li_Covariance_Drift_2026.pdf  (JMLR public)
