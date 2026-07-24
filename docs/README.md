# CSC3109 final report

This directory contains the LaTeX source, bibliography, figures, and supporting
notes for the CSC3109 Machine Learning group project.

## Main files

- `final-report.tex` - integrated final report.
- `mybib.bib` - bibliography used by the report.
- `figures/` - EDA samples, model curves, confusion matrices, and deployment
  evidence.
- `IMPLEMENTATION_SUMMARY.md` - verified implementation and deployment status.
- `titlepage.tex`, `includes.tex`, `notation.tex` - shared report formatting.

## Build

Compile from this directory with a LaTeX distribution that provides `pdflatex`,
BibTeX, and `latexmk`:

```powershell
Set-Location docs
latexmk -pdf final-report.tex
```

The final verification should include:

1. a successful build with resolved citations and references;
2. a page count within the coursework maximum;
3. visual inspection of every rendered page;
4. confirmation that the deployment screenshot and all model figures are
   legible.

Generated auxiliary files are build outputs. The submission PDF should follow
the team-number naming convention in the coursework specification.
