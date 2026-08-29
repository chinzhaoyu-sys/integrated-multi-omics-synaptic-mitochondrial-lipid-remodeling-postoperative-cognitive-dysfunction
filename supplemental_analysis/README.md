# Supplemental revision analysis

`rebuild_revision_panels_v3.py` is newly added analysis code. It is not part of the historical code supplied with the project.

The workflow reads the existing project matrices, checks the Control/AS sample mapping, calculates the Figure 6 statistics, and exports each panel as an individual PDF and 300-dpi PNG with a corresponding CSV file. The analysis reports `P` values and retains the statistical tests specified for each panel.

Expected inputs under `HANJUNGANG_DATA_ROOT` are the original project paths used by the workflow, including:

- `workflow/data/matrix/matrix_spatial_proteome.csv`
- `workflow/data/matrix/matrix_serum_proteome.csv`
- `workflow/data/matrix/matrix_serum_metabolic.csv`
- `workflow/data/matrix/matrix_serum_metabolic_original.csv`
- `workflow/data/sampleInfo/sampleInfo_spatial.csv`
- `workflow/data/sampleInfo/sampleInfo_serum.csv`
- `WOSP22099_report/2-Input/protein_Samplematrix_imputeNA_delOutlier.csv`
- `WOSP22099_report/2-Input/WOSP22099_sampleinfo.xlsx`

The workflow never writes into the input root. Outputs are created under `HANJUNGANG_OUTPUT_ROOT`.
