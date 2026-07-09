# Texas A&M Beamer Template

This template started from the presentation styling in `/home/vela/proj/arl/arl_fellowship_presentation.tex`, then was refit into a reusable Texas A\&M-themed deck with slide-specific content removed.

## Figures
**`TCAM.png`**

- TCAM output: <path>
- ParaView state file: <path>
- Manual step: open `something.py`, load <dataset.csv>, export png


## Files

- `arl_beamer_template.tex`: starter deck
- `beamerthemeARLProposal.sty`: reusable Beamer theme

## Usage

Compile from this directory with a standard LaTeX workflow, for example:

```bash
pdflatex arl_beamer_template.tex
```

or:

```bash
latexmk -pdf arl_beamer_template.tex
```

## What was preserved from the source deck

- `Madrid` + `seahorse` Beamer base
- top headline banner with short title and author
- frame-number footline
- block, alert, and timeline slide patterns
- helper emphasis macros: `\stage{}`, `\deliverable{}`, `\risk{}`

## What changed

- Replaced the source palette with a Texas A\&M maroon, charcoal, tan, and gold palette
- Updated the headline prefix and sample institute metadata for Texas A\&M
- Removed presentation-specific slides, references, and images
- Replaced external title art with a self-contained TikZ graphic
- Added reusable example slides for a thesis slide, comparison slide, visual slide, and timeline

## Notes

The template was compiled here with `tectonic` to produce `arl_beamer_template.pdf`.
