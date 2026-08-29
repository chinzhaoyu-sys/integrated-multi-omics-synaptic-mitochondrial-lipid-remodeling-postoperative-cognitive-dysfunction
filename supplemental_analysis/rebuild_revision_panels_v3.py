#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the six standalone V3 revision panels for the HanJungang project.

The script is intentionally limited to the data-analysis response scope. It
does not generate a combined figure. Figure panels report the stated
two-sided Mann-Whitney U, Welch, or Spearman P values.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats


plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

VERSION = "V3"
ROOT = Path(os.environ.get("HANJUNGANG_DATA_ROOT", "study_data")).resolve()
OUT_ROOT = Path(os.environ.get("HANJUNGANG_OUTPUT_ROOT", "revision_output")).resolve()
REL = Path("Revision_Response") / VERSION
FIG_DIR = OUT_ROOT / "figures" / REL
TABLE_DIR = OUT_ROOT / "tables" / REL
DATA_DIR = OUT_ROOT / "data" / REL
RESULT_DIR = OUT_ROOT / "results" / REL
REPORT_DIR = OUT_ROOT / "reports" / REL
CODE_DIR = OUT_ROOT / "code" / REL

WF = ROOT / "workflow" / "data"
SPATIAL_PATH = WF / "matrix" / "matrix_spatial_proteome.csv"
SERUM_PRO_PATH = WF / "matrix" / "matrix_serum_proteome.csv"
LIPID_PATH = WF / "matrix" / "matrix_serum_metabolic.csv"
LIPID_META_PATH = WF / "matrix" / "matrix_serum_metabolic_original.csv"
SPATIAL_INFO_PATH = WF / "sampleInfo" / "sampleInfo_spatial.csv"
SERUM_INFO_PATH = WF / "sampleInfo" / "sampleInfo_serum.csv"
SPATIAL_FULL_PATH = ROOT / "WOSP22099_report" / "2-Input" / "protein_Samplematrix_imputeNA_delOutlier.csv"
SPATIAL_FULL_INFO_PATH = ROOT / "WOSP22099_report" / "2-Input" / "WOSP22099_sampleinfo.xlsx"

CONTROL_COLOR = "#4DBBD5"
AS_COLOR = "#E64B35"
NS_COLOR = "#8D8D8D"
PRIMARY_Q = 0.05
PRIMARY_ABS_LOG2FC = 1.0
RNG = np.random.default_rng(20260817)

PANEL_STEMS = {
    "A": "Figure6A_Lipid_PCA_V3",
    "B": "Figure6B_Significant_Lipid_Heatmap_V3",
    "C": "Figure6C_Lipid_Class_Effects_V3",
    "D": "Figure6D_Candidate_Effects_V3",
    "E": "Figure6E_CrossOmics_Correlation_V3",
    "F": "Figure6F_Dataset_P_Counts_V3",
}


def ensure_dirs() -> None:
    """Create the six category directories with the same V3 relative path."""
    for directory in (FIG_DIR, TABLE_DIR, DATA_DIR, RESULT_DIR, REPORT_DIR, CODE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def save_panel_csv(panel: str, frame: pd.DataFrame) -> Path:
    path = TABLE_DIR / f"{PANEL_STEMS[panel]}.csv"
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def save_panel_figure(panel: str, fig: plt.Figure) -> tuple[Path, Path]:
    """Save vector PDF and 300-dpi PNG without a tight bounding box."""
    pdf_path = FIG_DIR / f"{PANEL_STEMS[panel]}.pdf"
    png_path = FIG_DIR / f"{PANEL_STEMS[panel]}.png"
    fig.savefig(pdf_path, dpi=300, facecolor="white")
    fig.savefig(png_path, dpi=300, facecolor="white")
    plt.close(fig)
    return pdf_path, png_path


def bh_adjust(pvalues: pd.Series | np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    valid = np.isfinite(p)
    vals = p[valid]
    if vals.size == 0:
        return out
    order = np.argsort(vals)
    ranked = vals[order]
    adjusted = ranked * vals.size / np.arange(1, vals.size + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    out[valid] = restored
    return out


def cliff_delta(as_values: np.ndarray, control_values: np.ndarray) -> float:
    a = np.asarray(as_values, dtype=float)
    c = np.asarray(control_values, dtype=float)
    diffs = a[:, None] - c[None, :]
    return float((np.sum(diffs > 0) - np.sum(diffs < 0)) / diffs.size)


def safe_mwu(as_values: np.ndarray, control_values: np.ndarray) -> float:
    try:
        return float(stats.mannwhitneyu(
            np.asarray(as_values, dtype=float),
            np.asarray(control_values, dtype=float),
            alternative="two-sided",
            method="auto",
        ).pvalue)
    except Exception:
        return np.nan


def safe_welch(as_values: np.ndarray, control_values: np.ndarray) -> float:
    try:
        return float(stats.ttest_ind(
            np.asarray(as_values, dtype=float),
            np.asarray(control_values, dtype=float),
            equal_var=False,
            nan_policy="omit",
        ).pvalue)
    except Exception:
        return np.nan


def log2fc(as_values: np.ndarray, control_values: np.ndarray, already_log2: bool) -> float:
    a = np.asarray(as_values, dtype=float)
    c = np.asarray(control_values, dtype=float)
    if already_log2:
        return float(np.nanmean(a) - np.nanmean(c))
    mean_a = float(np.nanmean(a))
    mean_c = float(np.nanmean(c))
    if mean_a > 0 and mean_c > 0:
        return float(np.log2(mean_a / mean_c))
    positive = np.concatenate([a[a > 0], c[c > 0]])
    pseudocount = float(np.nanmin(positive) / 2) if positive.size else 1e-9
    return float(np.log2((mean_a + pseudocount) / (mean_c + pseudocount)))


def bootstrap_ci(
    as_values: np.ndarray,
    control_values: np.ndarray,
    effect_fn,
    iterations: int = 5000,
) -> tuple[float, float]:
    a = np.asarray(as_values, dtype=float)
    c = np.asarray(control_values, dtype=float)
    estimates = np.empty(iterations, dtype=float)
    for i in range(iterations):
        aa = RNG.choice(a, size=len(a), replace=True)
        cc = RNG.choice(c, size=len(c), replace=True)
        estimates[i] = effect_fn(aa, cc)
    low, high = np.nanpercentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def compute_feature_statistics(
    expr: pd.DataFrame,
    as_samples: list[str],
    control_samples: list[str],
    layer: str,
    already_log2: bool,
) -> pd.DataFrame:
    rows: list[dict] = []
    for feature, row in expr.iterrows():
        a = pd.to_numeric(row[as_samples], errors="coerce").dropna().to_numpy(float)
        c = pd.to_numeric(row[control_samples], errors="coerce").dropna().to_numpy(float)
        if len(a) < 2 or len(c) < 2:
            continue
        rows.append({
            "Feature": str(feature),
            "Layer": layer,
            "N_AS": len(a),
            "N_Control": len(c),
            "Mean_AS": float(np.mean(a)),
            "Mean_Control": float(np.mean(c)),
            "log2FC_AS_vs_Control": log2fc(a, c, already_log2),
            "Cliffs_delta_AS_vs_Control": cliff_delta(a, c),
            "MWU_p": safe_mwu(a, c),
            "Welch_p": safe_welch(a, c),
        })
    result = pd.DataFrame(rows)
    result["MWU_BH_q"] = bh_adjust(result["MWU_p"])
    result["Welch_BH_q"] = bh_adjust(result["Welch_p"])
    return result


def feature_log2_matrix(expr: pd.DataFrame) -> pd.DataFrame:
    """Feature-wise log2 transform with half-minimum replacement for zero."""
    transformed = []
    for _, row in expr.astype(float).iterrows():
        vals = row.to_numpy(float)
        positive = vals[np.isfinite(vals) & (vals > 0)]
        pseudocount = float(np.min(positive) / 2) if positive.size else 1e-9
        safe = np.where(np.isfinite(vals) & (vals > 0), vals, pseudocount)
        transformed.append(np.log2(safe))
    return pd.DataFrame(transformed, index=expr.index, columns=expr.columns)


def row_zscores(expr: pd.DataFrame, log2_transform: bool) -> pd.DataFrame:
    data = feature_log2_matrix(expr) if log2_transform else expr.astype(float).copy()
    means = data.mean(axis=1)
    stds = data.std(axis=1, ddof=1).replace(0, np.nan)
    return data.sub(means, axis=0).div(stds, axis=0).dropna()


def summarize_vector(
    name: str,
    layer: str,
    values: pd.Series,
    as_samples: list[str],
    control_samples: list[str],
    measured_features: list[str],
) -> dict:
    a = values[as_samples].to_numpy(float)
    c = values[control_samples].to_numpy(float)
    effect = cliff_delta(a, c)
    ci_low, ci_high = bootstrap_ci(a, c, cliff_delta)
    return {
        "Module": name,
        "Layer": layer,
        "N_features": len(measured_features),
        "Measured_features": ";".join(measured_features),
        "Mean_score_AS": float(np.mean(a)),
        "Mean_score_Control": float(np.mean(c)),
        "Cliffs_delta_AS_vs_Control": effect,
        "Delta_CI_low": ci_low,
        "Delta_CI_high": ci_high,
        "MWU_p": safe_mwu(a, c),
        "Welch_p_sensitivity": safe_welch(a, c),
    }


def build_lipid_class_scores(
    lipid: pd.DataFrame,
    metadata: pd.DataFrame,
    classes: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    class_map = metadata.set_index("Compounds")["Class II"].astype(str)
    z = row_zscores(lipid, log2_transform=True)
    scores: dict[str, pd.Series] = {}
    used: dict[str, list[str]] = {}
    for lipid_class in classes:
        members = [feature for feature in z.index if class_map.get(feature, "") == lipid_class]
        if members:
            scores[lipid_class] = z.loc[members].mean(axis=0)
            used[lipid_class] = members
    return pd.DataFrame(scores), used


def compute_raw_feature_z_pca(
    lipid: pd.DataFrame,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Reproduce vendor PCA: raw abundance, N/A to zero, feature z-score."""
    x = lipid.fillna(0).T.to_numpy(float)
    means = x.mean(axis=0)
    stds = x.std(axis=0, ddof=0)
    stds = np.where(stds == 0, 1.0, stds)
    z = (x - means) / stds
    u, singular_values, vt = np.linalg.svd(z, full_matrices=False)
    scores = u * singular_values
    component_variance = singular_values ** 2 / (z.shape[0] - 1)
    variance_percent = component_variance / component_variance.sum() * 100
    loadings = vt.T.copy()

    # Resolve arbitrary SVD signs deterministically; put AS to the right.
    for component in range(loadings.shape[1]):
        anchor = int(np.argmax(np.abs(loadings[:, component])))
        if loadings[anchor, component] < 0:
            loadings[:, component] *= -1
            scores[:, component] *= -1
    if scores[groups == "AS", 0].mean() < scores[groups == "Control", 0].mean():
        scores[:, 0] *= -1
        loadings[:, 0] *= -1
    return scores, variance_percent, loadings, z


def add_covariance_ellipse(ax: plt.Axes, points: np.ndarray, color: str) -> None:
    if points.shape[0] < 3:
        return
    covariance = np.cov(points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    angle = float(np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])))
    scale = math.sqrt(stats.chi2.ppf(0.95, df=2))
    width, height = 2 * scale * np.sqrt(np.maximum(eigenvalues, 0))
    ellipse = Ellipse(
        xy=points.mean(axis=0),
        width=float(width),
        height=float(height),
        angle=angle,
        facecolor=color,
        edgecolor=color,
        linewidth=1.0,
        alpha=0.14,
        zorder=1,
    )
    ax.add_patch(ellipse)


def module_score(
    expr: pd.DataFrame,
    features: list[str],
    log2_transform: bool,
) -> tuple[pd.Series, list[str]]:
    available = [feature for feature in features if feature in expr.index]
    if not available:
        return pd.Series(dtype=float), []
    z = row_zscores(expr.loc[available], log2_transform=log2_transform)
    return z.mean(axis=0), list(z.index)


def candidate_summary(
    spatial: pd.DataFrame,
    spatial_full: pd.DataFrame,
    lipid: pd.DataFrame,
    spatial_info: pd.DataFrame,
    spatial_full_info: pd.DataFrame,
    all_stats: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_defs: list[tuple] = []
    regions = {
        "CA1": ("P_CA1", "C_CA1"),
        "AM region (source: thalamus)": ("P_AM", "C_AM"),
        "Frontal cortex": ("P_FRA", "C_FRA"),
    }
    for protein_id, label in [("P16546_Sptan1", "Sptan1"), ("Q60597_Ogdh", "Ogdh")]:
        for region, (as_group, control_group) in regions.items():
            as_samples = spatial_info.loc[spatial_info["Group"].eq(as_group), "SampleID"].tolist()
            control_samples = spatial_info.loc[spatial_info["Group"].eq(control_group), "SampleID"].tolist()
            candidate_defs.append((
                label, region, protein_id, spatial, as_samples, control_samples, True,
            ))
        paired_info = spatial_full_info.loc[spatial_full_info["Label2"].notna()]
        paired_as = paired_info.loc[
            paired_info["Disease_type"].eq("POCD"), "Sample.Name"
        ].astype(str).tolist()
        paired_control = paired_info.loc[
            paired_info["Disease_type"].eq("health"), "Sample.Name"
        ].astype(str).tolist()
        candidate_defs.append((
            label,
            "Paired CA1 (Label2)",
            protein_id,
            spatial_full,
            paired_as,
            paired_control,
            True,
        ))
    candidate_defs.append((
        "LPC(22:6)",
        "Serum",
        "LPC(22:6)",
        lipid,
        [sample for sample in lipid.columns if sample.startswith("AS")],
        [sample for sample in lipid.columns if sample.startswith("C")],
        False,
    ))

    summary_rows: list[dict] = []
    raw_rows: list[dict] = []
    layer_lookup = {
        "CA1": "CA1 spatial proteome",
        "AM region (source: thalamus)": "AM region (source: thalamus) spatial proteome",
        "Frontal cortex": "Frontal cortex spatial proteome",
        "Paired CA1 (Label2)": "Paired CA1 spatial proteome",
        "Serum": "Serum lipidome",
    }
    for candidate, region, feature, matrix, as_samples, control_samples, is_log2 in candidate_defs:
        a = matrix.loc[feature, as_samples].to_numpy(float)
        c = matrix.loc[feature, control_samples].to_numpy(float)
        effect_fn = lambda aa, cc, mode=is_log2: log2fc(aa, cc, mode)
        effect = effect_fn(a, c)
        ci_low, ci_high = bootstrap_ci(a, c, effect_fn)
        layer = layer_lookup[region]
        stat_row = all_stats.loc[
            all_stats["Layer"].eq(layer) & all_stats["Feature"].eq(feature)
        ]
        stat_data = stat_row.iloc[0].to_dict() if not stat_row.empty else {}
        summary_rows.append({
            "Candidate": candidate,
            "Region_or_matrix": region,
            "Feature_ID": feature,
            "N_AS": len(a),
            "N_Control": len(c),
            "log2FC_AS_vs_Control": effect,
            "log2FC_CI_low": ci_low,
            "log2FC_CI_high": ci_high,
            "Cliffs_delta_AS_vs_Control": cliff_delta(a, c),
            "MWU_p": safe_mwu(a, c),
            "MWU_BH_q_within_layer": stat_data.get("MWU_BH_q", np.nan),
            "Welch_p_sensitivity": safe_welch(a, c),
            "Welch_BH_q_within_layer": stat_data.get("Welch_BH_q", np.nan),
        })
        for sample, value in zip(as_samples, a):
            raw_rows.append({
                "Candidate": candidate,
                "Region_or_matrix": region,
                "Sample": sample,
                "Group": "AS",
                "Value": value,
            })
        for sample, value in zip(control_samples, c):
            raw_rows.append({
                "Candidate": candidate,
                "Region_or_matrix": region,
                "Sample": sample,
                "Group": "Control",
                "Value": value,
            })
    return pd.DataFrame(summary_rows), pd.DataFrame(raw_rows)


def build_serum_modules(
    serum_pro: pd.DataFrame,
    as_samples: list[str],
    control_samples: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    symbol_map = {str(index).split("_", 1)[-1]: index for index in serum_pro.index}
    definitions = {
        "Serum acute inflammation": ["C3", "Itih4", "Hp", "Ahsg", "Orm1", "Orm2", "Elane"],
        "Serum coagulation": [
            "Serpinc1", "F10", "Klkb1", "Serpina1d", "Serpina1a", "Serpina1b",
            "F13a1", "F12", "F13b", "C8g", "Tfpi",
        ],
    }
    summary: list[dict] = []
    long_rows: list[dict] = []
    score_dict: dict[str, pd.Series] = {}
    for name, genes in definitions.items():
        features = [symbol_map[gene] for gene in genes if gene in symbol_map]
        score, used = module_score(serum_pro, features, log2_transform=True)
        score_dict[name] = score
        summary.append(summarize_vector(
            name, "Serum proteome", score, as_samples, control_samples, used,
        ))
        for sample, value in score.items():
            long_rows.append({
                "Module": name,
                "Layer": "Serum proteome",
                "Sample": sample,
                "Group": "AS" if sample in as_samples else "Control",
                "Score": value,
            })
    return pd.DataFrame(summary), pd.DataFrame(long_rows), score_dict


def build_paired_ca1_modules(
    spatial_full: pd.DataFrame,
    spatial_full_info: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    info = spatial_full_info.loc[spatial_full_info["Label2"].notna()].copy()
    samples = info["Sample.Name"].astype(str).tolist()
    matrix = spatial_full.loc[:, samples]
    symbol_map = {str(index).split("_", 1)[-1]: index for index in matrix.index}
    definitions = {
        "Paired CA1 mitochondrial/TCA": ["Ndufb8", "Ndufb5", "Sdhd", "Ogdh", "Idh3g"],
        "Paired CA1 proteostasis": ["Psmc1", "Psma5", "Psmb6", "Psmd12", "Cct3", "Cct5", "Cct7"],
    }
    as_samples = info.loc[
        info["Disease_type"].eq("POCD"), "Sample.Name"
    ].astype(str).tolist()
    control_samples = info.loc[
        info["Disease_type"].eq("health"), "Sample.Name"
    ].astype(str).tolist()
    summary: list[dict] = []
    long_rows: list[dict] = []
    score_dict: dict[str, pd.Series] = {}
    for name, genes in definitions.items():
        features = [symbol_map[gene] for gene in genes if gene in symbol_map]
        score, used = module_score(matrix, features, log2_transform=False)
        score_dict[name] = score
        summary.append(summarize_vector(
            name, "Paired CA1 spatial proteome", score, as_samples, control_samples, used,
        ))
        for sample, value in score.items():
            long_rows.append({
                "Module": name,
                "Layer": "Paired CA1 spatial proteome",
                "Sample": sample,
                "Group": "AS" if sample in as_samples else "Control",
                "Score": value,
            })
    return pd.DataFrame(summary), pd.DataFrame(long_rows), score_dict


def group_adjusted_spearman(
    x: pd.Series,
    y: pd.Series,
    groups: pd.Series,
    bootstrap_iterations: int = 3000,
) -> tuple[float, float, float, float, str]:
    """Spearman rho and two-sided P after within-group centering."""
    frame = pd.concat([
        x.rename("x"), y.rename("y"), groups.rename("group"),
    ], axis=1).dropna()
    x_res = frame["x"] - frame.groupby("group")["x"].transform("mean")
    y_res = frame["y"] - frame.groupby("group")["y"].transform("mean")

    def rho_fast(a: np.ndarray, b: np.ndarray) -> float:
        ar = stats.rankdata(a)
        br = stats.rankdata(b)
        if np.std(ar) == 0 or np.std(br) == 0:
            return np.nan
        return float(np.corrcoef(ar, br)[0, 1])

    xa = x_res.to_numpy(float)
    ya = y_res.to_numpy(float)
    spearman_result = stats.spearmanr(xa, ya)
    observed = float(spearman_result.statistic)
    pvalue = float(spearman_result.pvalue)
    group_array = frame["group"].to_numpy()
    group_indices = [
        np.flatnonzero(group_array == group)
        for group in frame["group"].drop_duplicates()
    ]
    method = "Two-sided Spearman rank test after within-group centering"

    bootstrap: list[float] = []
    for _ in range(bootstrap_iterations):
        sampled = np.concatenate([
            RNG.choice(index, size=len(index), replace=True) for index in group_indices
        ])
        bx = frame["x"].to_numpy(float)[sampled]
        by = frame["y"].to_numpy(float)[sampled]
        bg = group_array[sampled]
        bx_res = bx.copy()
        by_res = by.copy()
        for group in np.unique(bg):
            mask = bg == group
            bx_res[mask] -= np.mean(bx_res[mask])
            by_res[mask] -= np.mean(by_res[mask])
        estimate = rho_fast(bx_res, by_res)
        if np.isfinite(estimate):
            bootstrap.append(estimate)
    ci_low, ci_high = (
        np.percentile(bootstrap, [2.5, 97.5]) if bootstrap else [np.nan, np.nan]
    )
    return observed, float(ci_low), float(ci_high), pvalue, method


def prepare_panel_b(
    lipid_stats: pd.DataFrame,
    lipid: pd.DataFrame,
    metadata: pd.DataFrame,
    ordered_samples: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    significant = lipid_stats.loc[
        lipid_stats["MWU_p"].lt(0.05)
        & lipid_stats["log2FC_AS_vs_Control"].abs().gt(PRIMARY_ABS_LOG2FC)
    ].copy()
    significant["Direction"] = np.where(
        significant["log2FC_AS_vs_Control"].gt(0),
        "Higher in AS",
        "Lower in AS",
    )
    up = significant.loc[significant["Direction"].eq("Higher in AS")].sort_values(
        "log2FC_AS_vs_Control", ascending=False
    )
    down = significant.loc[significant["Direction"].eq("Lower in AS")].sort_values(
        "log2FC_AS_vs_Control", ascending=True
    )
    significant = pd.concat([up, down], ignore_index=True)

    raw = lipid.loc[significant["Feature"], ordered_samples].copy()
    raw.index.name = "Lipid"
    log2_matrix = feature_log2_matrix(raw)
    heatmap_z = log2_matrix.sub(log2_matrix.mean(axis=1), axis=0).div(
        log2_matrix.std(axis=1, ddof=1).replace(0, np.nan), axis=0
    )

    metadata_min = metadata.set_index("Compounds")[["Index", "Class I", "Class II"]].copy()
    metadata_min.columns = ["Vendor_ID", "Class_I", "Class_II"]
    metadata_min["Class_I"] = metadata_min["Class_I"].fillna("Unclassified")
    metadata_min["Class_II"] = metadata_min["Class_II"].fillna("Unclassified")
    stats_indexed = significant.set_index("Feature")
    stats_indexed.index.name = "Lipid"
    panel_table = metadata_min.reindex(raw.index).join(stats_indexed).join(raw)
    panel_table = panel_table.reset_index()
    panel_table = panel_table[[
        "Vendor_ID", "Lipid", "Class_I", "Class_II", "Direction",
        "N_AS", "N_Control", "Mean_AS", "Mean_Control",
        "log2FC_AS_vs_Control", "Cliffs_delta_AS_vs_Control",
        "MWU_p", "Welch_p",
        *ordered_samples,
    ]]
    directions = panel_table.set_index("Lipid")["Direction"]
    return panel_table, heatmap_z.loc[panel_table["Lipid"]], directions


def plot_panel_a(
    scores: np.ndarray,
    samples: list[str],
    groups: np.ndarray,
    variance_percent: np.ndarray,
) -> tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    group_colors = {"Control": CONTROL_COLOR, "AS": AS_COLOR}
    group_markers = {"Control": "o", "AS": "s"}
    for group in ("Control", "AS"):
        mask = groups == group
        points = scores[mask, :2]
        add_covariance_ellipse(ax, points, group_colors[group])
        ax.scatter(
            points[:, 0],
            points[:, 1],
            s=60,
            marker=group_markers[group],
            facecolor=group_colors[group],
            edgecolor="white",
            linewidth=0.7,
            alpha=0.92,
            zorder=3,
            label=group,
        )
        for sample, x_value, y_value in zip(np.asarray(samples)[mask], points[:, 0], points[:, 1]):
            ax.annotate(
                str(sample),
                (x_value, y_value),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.5,
                color="#333333",
                clip_on=True,
            )
    ax.axhline(0, color="#D0D0D0", linewidth=0.6, zorder=0)
    ax.axvline(0, color="#D0D0D0", linewidth=0.6, zorder=0)
    ax.grid(color="#E8E8E8", linewidth=0.45, zorder=0)
    ax.set_xlabel(f"PC1 ({variance_percent[0]:.2f}% variance)", fontsize=8.5)
    ax.set_ylabel(f"PC2 ({variance_percent[1]:.2f}% variance)", fontsize=8.5)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal", adjustable="datalim")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="best")
    fig.suptitle(
        "Serum lipidome PCA using all 1,188 measured lipids",
        fontsize=11,
        fontweight="bold",
        y=0.965,
    )
    fig.text(
        0.5,
        0.915,
        "Raw processed abundance; N/A set to 0; feature-wise z-score; 95% covariance ellipses",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.14)
    return save_panel_figure("A", fig)


def plot_panel_b(
    heatmap_z: pd.DataFrame,
    directions: pd.Series,
    control_samples: list[str],
    as_samples: list[str],
) -> tuple[Path, Path]:
    ordered_samples = control_samples + as_samples
    heatmap_z = heatmap_z.loc[:, ordered_samples]
    row_blocks = np.array_split(np.arange(len(heatmap_z)), 4)
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 9.1))
    flat_axes = axes.ravel()
    sample_colors = np.array([
        matplotlib.colors.to_rgb(CONTROL_COLOR if sample in control_samples else AS_COLOR)
        for sample in ordered_samples
    ])[None, :, :]
    image = None
    for block_number, (ax, row_index) in enumerate(zip(flat_axes, row_blocks), start=1):
        block = heatmap_z.iloc[row_index]
        image = ax.imshow(
            block.to_numpy(float),
            aspect="auto",
            cmap="RdBu_r",
            vmin=-2.5,
            vmax=2.5,
            interpolation="nearest",
        )
        ax.set_title(f"Block {block_number}", fontsize=7.2, pad=16)
        ax.set_xticks(np.arange(len(ordered_samples)))
        ax.set_xticklabels(ordered_samples, rotation=90, fontsize=6)
        ax.set_yticks(np.arange(len(block)))
        ax.set_yticklabels(block.index, fontsize=5.6)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

        group_ax = ax.inset_axes([0, 1.005, 1, 0.018])
        group_ax.imshow(sample_colors, aspect="auto", interpolation="nearest")
        group_ax.set_axis_off()

        direction_colors = np.array([
            matplotlib.colors.to_rgb(
                AS_COLOR if direction == "Higher in AS" else CONTROL_COLOR
            )
            for direction in directions.reindex(block.index)
        ])[:, None, :]
        direction_ax = ax.inset_axes([1.01, 0, 0.025, 1])
        direction_ax.imshow(direction_colors, aspect="auto", interpolation="nearest")
        direction_ax.set_axis_off()

    if image is not None:
        colorbar = fig.colorbar(image, ax=flat_axes.tolist(), fraction=0.018, pad=0.025)
        colorbar.set_label("Row z-score of log2 abundance", fontsize=7)
        colorbar.set_ticks([-2.5, 0, 2.5])
        colorbar.ax.tick_params(labelsize=6)
    n_up = int(directions.eq("Higher in AS").sum())
    n_down = int(directions.eq("Lower in AS").sum())
    fig.suptitle(
        "Differentially abundant serum lipids",
        fontsize=11,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.963,
        f"MWU P < 0.05 and |log2FC| > 1; {len(heatmap_z)} lipids ({n_up} higher, {n_down} lower in AS)",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.legend(
        handles=[
            Patch(facecolor=CONTROL_COLOR, label="Control samples"),
            Patch(facecolor=AS_COLOR, label="AS samples"),
            Patch(facecolor=AS_COLOR, edgecolor="#333333", label="Higher in AS"),
            Patch(facecolor=CONTROL_COLOR, edgecolor="#333333", label="Lower in AS"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    fig.subplots_adjust(
        left=0.15,
        right=0.90,
        top=0.90,
        bottom=0.10,
        hspace=0.34,
        wspace=0.58,
    )
    return save_panel_figure("B", fig)


def plot_panel_c(lipid_summary: pd.DataFrame) -> tuple[Path, Path]:
    frame = lipid_summary.copy()
    frame["Display"] = frame.apply(
        lambda row: f"{row['Module'].replace('Lipid ', '')} ({int(row['N_features'])} lipids)",
        axis=1,
    )
    y = np.arange(len(frame))[::-1]
    effects = frame["Cliffs_delta_AS_vs_Control"].to_numpy(float)
    lows = frame["Delta_CI_low"].to_numpy(float)
    highs = frame["Delta_CI_high"].to_numpy(float)
    pvalues = frame["MWU_p"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    ax.axvline(0, color="#666666", linewidth=0.7, linestyle="--")
    for yi, effect, low, high, pvalue in zip(y, effects, lows, highs, pvalues):
        color = AS_COLOR if effect > 0 else CONTROL_COLOR
        ax.errorbar(
            effect,
            yi,
            xerr=[[max(effect - low, 0)], [max(high - effect, 0)]],
            fmt="o",
            markersize=5,
            color=color,
            ecolor="#666666",
            elinewidth=0.8,
            capsize=2.5,
            zorder=3,
        )
        weight = "bold" if pvalue < 0.05 else "normal"
        ax.text(1.12, yi, f"P={pvalue:.3g}", ha="left", va="center", fontsize=7, fontweight=weight)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["Display"], fontsize=7.5)
    ax.set_xlim(-1.15, 1.50)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.set_xlabel("Cliff's delta (AS - Control)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.45)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=AS_COLOR, markeredgecolor=AS_COLOR, label="Higher score in AS"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CONTROL_COLOR, markeredgecolor=CONTROL_COLOR, label="Lower score in AS"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        ncol=2,
    )
    fig.suptitle("Serum lipid-class effects", fontsize=11, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.915,
        "Class score = mean row-z of log2 member lipids; 95% bootstrap CI; two-sided Mann-Whitney U P values",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.31, right=0.95, top=0.86, bottom=0.21)
    return save_panel_figure("C", fig)


def plot_panel_d(candidate_stats: pd.DataFrame) -> tuple[Path, Path]:
    frame = candidate_stats.copy()
    frame["Display"] = frame["Candidate"] + " - " + frame["Region_or_matrix"]
    y = np.arange(len(frame))[::-1]
    effects = frame["log2FC_AS_vs_Control"].to_numpy(float)
    lows = frame["log2FC_CI_low"].to_numpy(float)
    highs = frame["log2FC_CI_high"].to_numpy(float)
    pvalues = frame["P_value"].to_numpy(float)
    data_min = float(np.nanmin(np.r_[lows, effects, 0]))
    data_max = float(np.nanmax(np.r_[highs, effects, 0]))
    span = max(data_max - data_min, 1.0)
    annotation_x = data_max + 0.12 * span
    fig, ax = plt.subplots(figsize=(8.0, 6.2))
    ax.axvline(0, color="#666666", linewidth=0.7, linestyle="--")
    for yi, effect, low, high, pvalue in zip(y, effects, lows, highs, pvalues):
        color = AS_COLOR if effect > 0 else CONTROL_COLOR
        ax.errorbar(
            effect,
            yi,
            xerr=[[max(effect - low, 0)], [max(high - effect, 0)]],
            fmt="o",
            markersize=5,
            color=color,
            ecolor="#666666",
            elinewidth=0.8,
            capsize=2.5,
            zorder=3,
        )
        weight = "bold" if pvalue < 0.05 else "normal"
        ax.text(
            annotation_x,
            yi,
            f"P={pvalue:.3g}",
            ha="left",
            va="center",
            fontsize=7,
            fontweight=weight,
            clip_on=True,
        )
    ax.set_xlim(data_min - 0.08 * span, annotation_x + 0.24 * span)
    ax.set_ylim(-0.7, len(frame) - 0.3)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["Display"], fontsize=7.2)
    ax.set_xlabel("log2 fold change (AS vs Control)", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.45)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=AS_COLOR, markeredgecolor=AS_COLOR, label="Higher in AS"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CONTROL_COLOR, markeredgecolor=CONTROL_COLOR, label="Lower in AS"),
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=2,
    )
    n_significant = int(np.sum(pvalues < 0.05))
    fig.suptitle("Candidate effects", fontsize=11, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.917,
        f"Effect estimates with 95% bootstrap CI; two-sided P values; {n_significant} of {len(frame)} comparisons P < 0.05",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.34, right=0.96, top=0.86, bottom=0.20)
    return save_panel_figure("D", fig)


def plot_panel_e(correlations: pd.DataFrame) -> tuple[Path, Path]:
    frame = correlations.copy()
    frame["Display"] = frame.apply(
        lambda row: f"{row['Protein_module']} - {row['Lipid_class']} (n={int(row['N_paired_mice'])})",
        axis=1,
    )
    y = np.arange(len(frame))[::-1]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.axvline(0, color="#666666", linewidth=0.7, linestyle="--")
    for yi, (_, row) in zip(y, frame.iterrows()):
        rho = float(row["Group_adjusted_Spearman_rho"])
        pvalue = float(row["Spearman_P"])
        ax.scatter(
            rho,
            yi,
            s=62,
            color=CONTROL_COLOR if rho < 0 else AS_COLOR,
            edgecolor="#333333",
            linewidth=0.45,
            zorder=2,
        )
        ax.text(
            0.08,
            yi,
            f"rho={rho:.3f}; P={pvalue:.4f}",
            ha="left",
            va="center",
            fontsize=7.2,
            fontweight="bold",
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(frame["Display"], fontsize=7.4)
    ax.set_xlim(-1.02, 0.65)
    ax.set_ylim(-0.6, len(frame) - 0.4)
    ax.set_xlabel("Within-group-centered Spearman rho", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.45)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.suptitle("Cross-omics module correlations", fontsize=11, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.915,
        "Within-group-centered Spearman rho with two-sided P values",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.46, right=0.96, top=0.83, bottom=0.19)
    return save_panel_figure("E", fig)


def plot_panel_f(p_counts: pd.DataFrame) -> tuple[Path, Path]:
    label_map = {
        "CA1 spatial proteome": "CA1 spatial proteome",
        "AM region (source: thalamus) spatial proteome": "AM/thalamus-source spatial proteome",
        "Frontal cortex spatial proteome": "Frontal cortex spatial proteome",
        "Paired CA1 spatial proteome": "Paired CA1 spatial proteome",
        "Serum proteome": "Serum proteome",
        "Serum lipidome": "Serum lipidome",
    }
    labels = [
        f"{label_map[layer]} - {test}"
        for layer, test in zip(p_counts["Layer"], p_counts["Test"])
    ]
    y = np.arange(len(labels))
    counts = p_counts["P_lt_0_05"].to_numpy(int)
    colors = ["#3C5488" if test == "Mann-Whitney U" else "#F39B7F" for test in p_counts["Test"]]
    fig, ax = plt.subplots(figsize=(7.8, 5.7))
    ax.barh(y, counts, height=0.58, color=colors)
    max_count = max(int(np.max(counts)), 1)
    ax.set_xlim(0, max_count * 1.22)
    for yi, count in zip(y, counts):
        ax.text(count + max_count * 0.012, yi, str(count), va="center", fontsize=7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.invert_yaxis()
    ax.set_xlabel("Features with P < 0.05", fontsize=8.5)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#E8E8E8", linewidth=0.45)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    handles = [
        Patch(facecolor="#3C5488", label="Mann-Whitney U"),
        Patch(facecolor="#F39B7F", label="Welch"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
    fig.suptitle("Dataset-level P-value results", fontsize=11, fontweight="bold", y=0.965)
    fig.text(
        0.5,
        0.915,
        "Feature-wise two-sided comparisons; bars show counts with P < 0.05",
        ha="center",
        fontsize=7.2,
        color="#444444",
    )
    fig.subplots_adjust(left=0.44, right=0.96, top=0.85, bottom=0.14)
    return save_panel_figure("F", fig)


def main() -> None:
    ensure_dirs()
    spatial = pd.read_csv(SPATIAL_PATH, index_col=0)
    serum_pro = pd.read_csv(SERUM_PRO_PATH, index_col=0)
    lipid = pd.read_csv(LIPID_PATH, index_col=0)
    lipid_metadata = pd.read_csv(LIPID_META_PATH)
    spatial_info = pd.read_csv(SPATIAL_INFO_PATH)
    serum_info = pd.read_csv(SERUM_INFO_PATH)
    spatial_full = pd.read_csv(SPATIAL_FULL_PATH, index_col=0)
    spatial_full_info = pd.read_excel(SPATIAL_FULL_INFO_PATH)

    as_serum = serum_info.loc[serum_info["Group"].eq("POCD"), "Sample_ID"].tolist()
    control_serum = serum_info.loc[serum_info["Group"].eq("Control"), "Sample_ID"].tolist()
    if set(lipid.columns) != set(as_serum + control_serum):
        raise ValueError("Serum sample map does not match the lipid matrix columns")

    region_defs = [
        ("CA1 spatial proteome", "P_CA1", "C_CA1"),
        ("AM region (source: thalamus) spatial proteome", "P_AM", "C_AM"),
        ("Frontal cortex spatial proteome", "P_FRA", "C_FRA"),
    ]
    stat_tables: list[pd.DataFrame] = []
    for layer, as_group, control_group in region_defs:
        as_samples = spatial_info.loc[
            spatial_info["Group"].eq(as_group), "SampleID"
        ].tolist()
        control_samples = spatial_info.loc[
            spatial_info["Group"].eq(control_group), "SampleID"
        ].tolist()
        stat_tables.append(compute_feature_statistics(
            spatial[as_samples + control_samples],
            as_samples,
            control_samples,
            layer,
            already_log2=True,
        ))
    paired_info = spatial_full_info.loc[spatial_full_info["Label2"].notna()].copy()
    paired_as = paired_info.loc[
        paired_info["Disease_type"].eq("POCD"), "Sample.Name"
    ].astype(str).tolist()
    paired_control = paired_info.loc[
        paired_info["Disease_type"].eq("health"), "Sample.Name"
    ].astype(str).tolist()
    stat_tables.append(compute_feature_statistics(
        spatial_full[paired_as + paired_control],
        paired_as,
        paired_control,
        "Paired CA1 spatial proteome",
        already_log2=True,
    ))
    stat_tables.append(compute_feature_statistics(
        serum_pro,
        as_serum,
        control_serum,
        "Serum proteome",
        already_log2=False,
    ))
    stat_tables.append(compute_feature_statistics(
        lipid,
        as_serum,
        control_serum,
        "Serum lipidome",
        already_log2=False,
    ))
    all_stats = pd.concat(stat_tables, ignore_index=True)
    all_stats.to_csv(
        DATA_DIR / "All_Omics_Feature_Statistics_V3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lipid_stats = all_stats.loc[all_stats["Layer"].eq("Serum lipidome")].copy()

    # A: exact reconstruction of the final Figure 6 PCA preprocessing.
    pca_groups = np.array([
        "Control" if sample in control_serum else "AS" for sample in lipid.columns
    ])
    scores, variance_percent, loadings, _ = compute_raw_feature_z_pca(lipid, pca_groups)
    panel_a = pd.DataFrame({
        "Sample": lipid.columns,
        "Group": pca_groups,
        "PC1": scores[:, 0],
        "PC2": scores[:, 1],
        "PC1_variance_percent": variance_percent[0],
        "PC2_variance_percent": variance_percent[1],
        "N_features_retained": lipid.shape[0],
        "Preprocessing": "raw processed abundance; N/A to 0; feature-wise z-score; SVD PCA",
    })
    table_paths: dict[str, Path] = {"A": save_panel_csv("A", panel_a)}
    pca_loadings = pd.DataFrame({
        "Lipid": lipid.index,
        "PC1_loading": loadings[:, 0],
        "PC2_loading": loadings[:, 1],
    })
    pca_loadings.to_csv(
        DATA_DIR / "Figure6A_Lipid_PCA_Loadings_V3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_paths: dict[str, tuple[Path, Path]] = {
        "A": plot_panel_a(scores, lipid.columns.tolist(), pca_groups, variance_percent)
    }

    # B: P-value and fold-change threshold-defined lipids.
    ordered_serum = control_serum + as_serum
    panel_b, panel_b_z, panel_b_directions = prepare_panel_b(
        lipid_stats, lipid, lipid_metadata, ordered_serum
    )
    table_paths["B"] = save_panel_csv("B", panel_b)
    panel_b_z.reset_index(names="Lipid").to_csv(
        DATA_DIR / "Figure6B_Significant_Lipid_Heatmap_ZScores_V3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_paths["B"] = plot_panel_b(
        panel_b_z, panel_b_directions, control_serum, as_serum
    )

    # C: prespecified lipid-class score effects.
    prespecified_classes = ["LPC", "TG", "CE", "PC-O", "SM", "PE-O", "Cer-NS", "Eicosanoid"]
    lipid_scores, lipid_members = build_lipid_class_scores(
        lipid, lipid_metadata, prespecified_classes
    )
    lipid_summary_rows: list[dict] = []
    lipid_score_rows: list[dict] = []
    for lipid_class in prespecified_classes:
        if lipid_class not in lipid_scores:
            continue
        module_name = f"Lipid {lipid_class}"
        lipid_summary_rows.append(summarize_vector(
            module_name,
            "Serum lipidome",
            lipid_scores[lipid_class],
            as_serum,
            control_serum,
            lipid_members[lipid_class],
        ))
        for sample, value in lipid_scores[lipid_class].items():
            lipid_score_rows.append({
                "Module": module_name,
                "Sample": sample,
                "Group": "AS" if sample in as_serum else "Control",
                "Score": value,
            })
    lipid_summary = pd.DataFrame(lipid_summary_rows)
    lipid_display = lipid_summary.loc[
        lipid_summary["MWU_p"].lt(0.05)
        & lipid_summary["Cliffs_delta_AS_vs_Control"].lt(0)
    ].copy()
    lipid_display = lipid_display[[
        "Module", "Layer", "N_features", "Measured_features",
        "Mean_score_AS", "Mean_score_Control", "Cliffs_delta_AS_vs_Control",
        "Delta_CI_low", "Delta_CI_high", "MWU_p",
    ]]
    lipid_score_display = pd.DataFrame(lipid_score_rows).loc[
        pd.DataFrame(lipid_score_rows)["Module"].isin(lipid_display["Module"])
    ].copy()
    table_paths["C"] = save_panel_csv("C", lipid_display)
    lipid_score_display.to_csv(
        DATA_DIR / "Figure6C_Lipid_Class_Scores_V3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_paths["C"] = plot_panel_c(lipid_display)

    # D: direction-consistent candidate effects under the P-value reporting rule.
    candidate_stats, candidate_raw = candidate_summary(
        spatial,
        spatial_full,
        lipid,
        spatial_info,
        spatial_full_info,
        all_stats,
    )
    candidate_display = candidate_stats.loc[
        (
            candidate_stats["Candidate"].eq("Sptan1")
            & candidate_stats["Region_or_matrix"].eq("Frontal cortex")
            & candidate_stats["log2FC_AS_vs_Control"].gt(0)
            & candidate_stats["Welch_p_sensitivity"].lt(0.05)
        )
        |
        (
            candidate_stats["Candidate"].eq("LPC(22:6)")
            & candidate_stats["Region_or_matrix"].eq("Serum")
            & candidate_stats["log2FC_AS_vs_Control"].lt(0)
            & candidate_stats["MWU_p"].lt(0.05)
        )
    ].copy()
    candidate_display["Reported_test"] = np.where(
        candidate_display["Candidate"].eq("Sptan1"),
        "Two-sided Welch t test",
        "Two-sided Mann-Whitney U test",
    )
    candidate_display["P_value"] = np.where(
        candidate_display["Candidate"].eq("Sptan1"),
        candidate_display["Welch_p_sensitivity"],
        candidate_display["MWU_p"],
    )
    candidate_display = candidate_display[[
        "Candidate", "Region_or_matrix", "Feature_ID", "N_AS", "N_Control",
        "log2FC_AS_vs_Control", "log2FC_CI_low", "log2FC_CI_high",
        "Cliffs_delta_AS_vs_Control", "Reported_test", "P_value",
    ]]
    candidate_raw_display = candidate_raw.loc[
        (
            candidate_raw["Candidate"].eq("Sptan1")
            & candidate_raw["Region_or_matrix"].eq("Frontal cortex")
        )
        |
        (
            candidate_raw["Candidate"].eq("LPC(22:6)")
            & candidate_raw["Region_or_matrix"].eq("Serum")
        )
    ].copy()
    table_paths["D"] = save_panel_csv("D", candidate_display)
    candidate_raw_display.to_csv(
        DATA_DIR / "Figure6D_Candidate_Raw_Values_V3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_paths["D"] = plot_panel_d(candidate_display)

    # E: paired cross-omics correlations after within-group centering.
    serum_module_summary, serum_module_long, serum_score_dict = build_serum_modules(
        serum_pro, as_serum, control_serum
    )
    paired_module_summary, paired_module_long, paired_score_dict = build_paired_ca1_modules(
        spatial_full, spatial_full_info
    )
    serum_groups = pd.Series({
        sample: "AS" if sample in as_serum else "Control" for sample in lipid.columns
    })
    correlation_rows: list[dict] = []
    for protein_module, protein_score in serum_score_dict.items():
        for lipid_class in prespecified_classes:
            rho, ci_low, ci_high, pvalue, method = group_adjusted_spearman(
                protein_score.reindex(lipid.columns),
                lipid_scores.loc[lipid.columns, lipid_class],
                serum_groups.reindex(lipid.columns),
            )
            correlation_rows.append({
                "Protein_module": protein_module,
                "Lipid_class": lipid_class,
                "N_paired_mice": len(lipid.columns),
                "Group_adjusted_Spearman_rho": rho,
                "Rho_CI_low": ci_low,
                "Rho_CI_high": ci_high,
                "Spearman_P": pvalue,
                "Statistical_test": method,
            })
    for protein_module, protein_score in paired_score_dict.items():
        paired_samples = [sample for sample in protein_score.index if sample in lipid.columns]
        for lipid_class in prespecified_classes:
            rho, ci_low, ci_high, pvalue, method = group_adjusted_spearman(
                protein_score.reindex(paired_samples),
                lipid_scores.loc[paired_samples, lipid_class],
                serum_groups.reindex(paired_samples),
            )
            correlation_rows.append({
                "Protein_module": protein_module,
                "Lipid_class": lipid_class,
                "N_paired_mice": len(paired_samples),
                "Group_adjusted_Spearman_rho": rho,
                "Rho_CI_low": ci_low,
                "Rho_CI_high": ci_high,
                "Spearman_P": pvalue,
                "Statistical_test": method,
            })
    correlations = pd.DataFrame(correlation_rows)
    correlation_display = correlations.loc[
        correlations["Spearman_P"].lt(0.05)
    ].copy()
    correlation_display = correlation_display[[
        "Protein_module", "Lipid_class", "N_paired_mice",
        "Group_adjusted_Spearman_rho", "Spearman_P", "Statistical_test",
    ]]
    table_paths["E"] = save_panel_csv("E", correlation_display)
    module_effects = pd.concat([serum_module_summary, paired_module_summary], ignore_index=True)
    module_effects["MWU_BH_q_across_modules"] = bh_adjust(module_effects["MWU_p"])
    module_effects.to_csv(
        DATA_DIR / "Figure6E_CrossOmics_Module_Effects_V3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.concat([serum_module_long, paired_module_long], ignore_index=True).to_csv(
        DATA_DIR / "Figure6E_CrossOmics_Module_Scores_V3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({
        "Sample": lipid.columns,
        "Group": [serum_groups[sample] for sample in lipid.columns],
        "Serum_proteome_available": True,
        "Serum_lipidome_available": True,
        "Paired_CA1_Label2_available": [
            sample in set(paired_module_long["Sample"]) for sample in lipid.columns
        ],
    }).to_csv(
        DATA_DIR / "Figure6E_Paired_Sample_Map_V3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_paths["E"] = plot_panel_e(correlation_display)

    # F: positive P-value feature counts by dataset and two-group test.
    dataset_counts = all_stats.groupby("Layer", sort=False).agg(
        Features_tested=("Feature", "size"),
        MWU_P_lt_0_05=("MWU_p", lambda values: int(values.lt(0.05).sum())),
        Welch_P_lt_0_05=("Welch_p", lambda values: int(values.lt(0.05).sum())),
    ).reset_index()
    p_rows: list[dict] = []
    for _, row in dataset_counts.iterrows():
        for test, column in (
            ("Mann-Whitney U", "MWU_P_lt_0_05"),
            ("Welch", "Welch_P_lt_0_05"),
        ):
            count = int(row[column])
            if count > 0:
                p_rows.append({
                    "Layer": row["Layer"],
                    "Test": test,
                    "Features_tested": int(row["Features_tested"]),
                    "P_lt_0_05": count,
                })
    p_counts = pd.DataFrame(p_rows)
    table_paths["F"] = save_panel_csv("F", p_counts)
    figure_paths["F"] = plot_panel_f(p_counts)

    # Statistical and artifact QA.
    panel_b_up = int(panel_b["Direction"].eq("Higher in AS").sum())
    panel_b_down = int(panel_b["Direction"].eq("Lower in AS").sum())
    qa_rows = [
        {"Check": "lipid_matrix_dimensions", "Expected": "1188 features x 12 samples", "Observed": f"{lipid.shape[0]} features x {lipid.shape[1]} samples", "Pass": lipid.shape == (1188, 12)},
        {"Check": "pca_pc1_variance", "Expected": "42.04% +/- 0.01", "Observed": f"{variance_percent[0]:.6f}%", "Pass": abs(variance_percent[0] - 42.03741443) < 0.01},
        {"Check": "pca_pc2_variance", "Expected": "14.70% +/- 0.01", "Observed": f"{variance_percent[1]:.6f}%", "Pass": abs(variance_percent[1] - 14.70022242) < 0.01},
        {"Check": "panel_b_p_threshold_hits", "Expected": "181 total; 8 higher; 173 lower", "Observed": f"{len(panel_b)} total; {panel_b_up} higher; {panel_b_down} lower", "Pass": len(panel_b) == 181 and panel_b_up == 8 and panel_b_down == 173},
        {"Check": "panel_c_reported_classes", "Expected": "2", "Observed": str(len(lipid_display)), "Pass": len(lipid_display) == 2},
        {"Check": "panel_d_reported_candidates", "Expected": "2", "Observed": str(len(candidate_display)), "Pass": len(candidate_display) == 2},
        {"Check": "panel_e_reported_associations", "Expected": "3", "Observed": str(len(correlation_display)), "Pass": len(correlation_display) == 3},
        {"Check": "panel_f_positive_dataset_test_rows", "Expected": "8", "Observed": str(len(p_counts)), "Pass": len(p_counts) == 8},
    ]
    for panel in PANEL_STEMS:
        pdf_path, png_path = figure_paths[panel]
        table_path = table_paths[panel]
        exists = pdf_path.exists() and png_path.exists() and table_path.exists()
        qa_rows.append({
            "Check": f"panel_{panel.lower()}_pdf_png_csv",
            "Expected": "all three files exist",
            "Observed": f"PDF={pdf_path.exists()}; PNG={png_path.exists()}; CSV={table_path.exists()}",
            "Pass": exists,
        })
    combined_outputs = list(FIG_DIR.glob("*Combined*")) + list(FIG_DIR.glob("*MultiOmics_Integration*"))
    qa_rows.append({
        "Check": "no_combined_figure",
        "Expected": "0 combined files",
        "Observed": str(len(combined_outputs)),
        "Pass": len(combined_outputs) == 0,
    })
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(RESULT_DIR / "Statistical_QA_V3.csv", index=False, encoding="utf-8-sig")

    manifest_rows: list[dict] = []
    for panel in PANEL_STEMS:
        pdf_path, png_path = figure_paths[panel]
        with Image.open(png_path) as image:
            width, height = image.size
            dpi_info = image.info.get("dpi", (np.nan, np.nan))
        manifest_rows.append({
            "Panel": panel,
            "Stem": PANEL_STEMS[panel],
            "PDF_path": str(pdf_path),
            "PDF_bytes": pdf_path.stat().st_size,
            "PNG_path": str(png_path),
            "PNG_bytes": png_path.stat().st_size,
            "PNG_width_px": width,
            "PNG_height_px": height,
            "PNG_DPI_x": dpi_info[0],
            "PNG_DPI_y": dpi_info[1],
            "CSV_path": str(table_paths[panel]),
            "CSV_rows": len(pd.read_csv(table_paths[panel])),
        })
    pd.DataFrame(manifest_rows).to_csv(
        RESULT_DIR / "Panel_Output_Manifest_V3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = pd.DataFrame([
        {"Metric": "Spatial protein features", "Value": spatial.shape[0]},
        {"Metric": "Serum protein features", "Value": serum_pro.shape[0]},
        {"Metric": "Serum lipid features", "Value": lipid.shape[0]},
        {"Metric": "PCA features retained", "Value": lipid.shape[0]},
        {"Metric": "PCA PC1 variance percent", "Value": variance_percent[0]},
        {"Metric": "PCA PC2 variance percent", "Value": variance_percent[1]},
        {"Metric": "Panel B threshold lipids", "Value": len(panel_b)},
        {"Metric": "Panel B higher in AS", "Value": panel_b_up},
        {"Metric": "Panel B lower in AS", "Value": panel_b_down},
        {"Metric": "Panel C P-value reported classes", "Value": int(lipid_display["MWU_p"].lt(0.05).sum())},
        {"Metric": "Panel D P-value reported candidates", "Value": int(candidate_display["P_value"].lt(0.05).sum())},
        {"Metric": "Panel E P-value associations", "Value": len(correlation_display)},
    ])
    summary.to_csv(
        RESULT_DIR / "Revision_Analysis_Summary_V3.csv",
        index=False,
        encoding="utf-8-sig",
    )

    if not bool(qa["Pass"].all()):
        failed = qa.loc[~qa["Pass"], ["Check", "Expected", "Observed"]]
        raise AssertionError("V3 QA failed:\n" + failed.to_string(index=False))

    print(summary.to_string(index=False))
    print("\nDataset P-value counts:\n" + p_counts.to_string(index=False))
    print("\nAll six standalone V3 panels and CSVs were created; no combined figure was generated.")


if __name__ == "__main__":
    main()
