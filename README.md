# Integrated multi-omics synaptic, mitochondrial, and lipid remodeling in postoperative cognitive dysfunction

This repository contains the analysis code associated with the manuscript revision. It separates the code supplied with the original project from analyses added during revision.

## Repository structure

- `historical_code/` contains byte-for-byte copies of the supplied historical R/R Markdown code and R session histories. Files were only reorganized or renamed; their contents were not edited.
- `supplemental_analysis/` contains newly added code used for the revision analyses and standalone Figure 6 panels.

The provenance of every historical file is documented in `HISTORICAL_CODE_SHA256.txt`. Identical SHA-256 hashes verify that the repository copies have the same contents as the supplied files.

## Historical code

The recoverable historical analysis code includes spatial-proteomics differential/enrichment analysis, serum-proteomics differential/enrichment analysis, cell deconvolution, tissue-origin analysis, data merging, the RStudio project file, and two R session histories. The original scripts retain their original relative paths, object names, statistical choices, and comments.

Historical scripts were not rewritten to improve portability or to align them retrospectively with the revision analyses.

## Supplemental revision analysis

The supplemental Python workflow produces six standalone Figure 6 panels and their panel-level CSV files. Statistical results are reported as `P` values; the figures and tables do not use the qualifier `raw`.

Set the input and output roots before running:

```powershell
$env:HANJUNGANG_DATA_ROOT = "D:\path\to\HanJungang"
$env:HANJUNGANG_OUTPUT_ROOT = "D:\path\to\revision-output"
python supplemental_analysis/rebuild_revision_panels_v3.py
```

If the environment variables are omitted, the workflow uses `study_data/` and `revision_output/` under the current working directory.

Install the Python dependencies with:

```bash
python -m pip install -r supplemental_analysis/requirements.txt
```

## Data availability and repository scope

No raw mass-spectrometry files, processed study matrices, participant- or animal-level data, manuscripts, reviewer documents, generated figures, or generated tables are committed. The analysis code reads the study files locally and writes results outside the input data root.
