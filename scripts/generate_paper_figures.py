"""Regenerate the los-paper.tex run-derived figures/tables from results folders.

`text/tex/los-paper.tex` (sibling `text/` repo) is the SoftwareX companion
paper for `los_estimator`. Several of its figures and two of its tables are
numeric outputs of a `los_estimator` pipeline run, currently pasted into the
manuscript by hand. This script regenerates them from one or more results
folders and collects them in a staging output directory, ready for a manual
(reviewed) copy into `text/tex/`. It does not write into `text/` itself.

Usage:
    python -m scripts.generate_paper_figures \\
        --real-run <path to real-data DIVI/RKI results folder> \\
        --synthetic-run <path to synthetic_example.py results folder> \\
        --out <staging output folder> \\
        [--animation-frame-window <window index>] \\
        [--windows 3 17 28] [--distros gaussian exponential cauchy linear compartmental]

`--real-run`, `--synthetic-run`, and `--out` all default to the current
author-confirmed case-study run (see `DEFAULT_REAL_RUN` /
`DEFAULT_SYNTHETIC_RUN` / `DEFAULT_OUT` below), so this file can also just be
run directly -- e.g. hit "Run" on it in an IDE -- with no arguments at all.
To point it at a new data version, either pass `--real-run`/`--synthetic-run`
on the command line, or edit those three constants.

Output filenames follow the order the corresponding `\\begin{figure}` blocks
actually appear in `los-paper.tex` (i.e. the number each will compile to),
NOT the order this script happens to generate them in:

    figure_2.pdf -- fig:training               (figure_2/figure_2.py, via --real-run)
    figure_3.png -- fig:error                   (--real-run, needs UQ enabled, 2x2 grid of example outputs)
    figure_4.png -- fig:animation_frame         (--real-run)
    figure_6.png -- (not referenced in los-paper.tex; kept for reuse if a future figure needs it)

`fig:coverage_over_time` (the former standalone `figure_4.png`) is no longer
its own figure -- it is `figure_3.png`'s panel D. `fig:synthetic_kernel_recovery`
(the former `figure_6.png`) was likewise dropped from `los-paper.tex`
(2026-09-01, redundant with `tab:synthetic-recovery`'s numbers); this script
still generates `figure_6.png` since nothing currently reuses that code path,
but it is not copied into `text/tex/` any more.

`tab_coverage.csv` (tab:coverage) also comes from `--real-run`;
`tab_synthetic_recovery.csv` (tab:synthetic-recovery) from `--synthetic-run`.
Either run may be omitted; the figures/tables that need it are then skipped
with a printed reason, not silently produced with wrong data.

See `text/claude_prompts/generate-paper-figures.md` for the full inventory
this script implements, including why each figure/table maps to that
filename.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_CODE_ROOT = Path(__file__).resolve().parent.parent
# Every saved run's run_configurations.toml stores output_config.base as the
# relative path "./results", resolved against the process's cwd at
# LosEstimationRun.load_run() time -- not against the run folder or
# _CODE_ROOT. Without pinning cwd here, this script only works when launched
# with cwd already at code/ (e.g. `cd code && python -m
# scripts.generate_paper_figures`); an IDE's "Run" button often uses the
# file's own directory or the project root instead, which would silently
# resolve results/.../metrics against the wrong place. Pinning cwd up front
# makes every default and every run's relative paths resolve consistently
# regardless of how this script was launched. Captured *before* the chdir so
# a caller's own relative --real-run/--synthetic-run/--out overrides can
# still be resolved against the directory they actually ran this from,
# rather than silently against code/ -- see `_resolve_from_original_cwd`.
_ORIGINAL_CWD = Path.cwd()
os.chdir(_CODE_ROOT)


def _resolve_from_original_cwd(path_str: str) -> str:
    """Resolve a possibly-relative CLI path against the caller's original cwd.

    `os.chdir(_CODE_ROOT)` above means a relative path parsed by argparse
    would otherwise silently resolve against code/ instead of wherever the
    user actually ran this script from.
    """
    p = Path(path_str)
    return str(p if p.is_absolute() else (_ORIGINAL_CWD / p))


sys.path.insert(0, str(_CODE_ROOT))
sys.path.insert(0, str(_CODE_ROOT / "figure_2"))

from los_estimator.estimation_run import LosEstimationRun  # noqa: E402
from los_estimator.fitting.distributions import Distributions  # noqa: E402
from los_estimator.fitting.fit_results import MultiSeriesFitResults  # noqa: E402
from los_estimator.visualization import DeconvolutionPlots  # noqa: E402

import figure_2 as figure_2_module  # noqa: E402

# The Configuration section of los-paper.tex describes the real-data case
# study as a comparison of exactly these five distributions. A `--real-run`
# results folder may have been fit with more (e.g. a fuller multi-distribution
# sweep used for other purposes), so figure_3/figure_4/tab_coverage.csv -- the
# outputs that directly correspond to that five-way comparison -- filter down
# to this set rather than showing whatever superset the run happened to fit.
CASE_STUDY_DISTROS = ["gaussian", "exponential", "cauchy", "t", "linear"]


def _filter_fit_results(all_fit_results: MultiSeriesFitResults, distros) -> MultiSeriesFitResults:
    """Subset a baked `MultiSeriesFitResults` down to `distros`, re-baked.

    `bake()` recomputes every derived array/summary from the container's
    current keys, so this is the supported way to narrow one down -- see
    `MultiSeriesFitResults.bake` in `los_estimator/fitting/fit_results.py`.
    """
    missing = [d for d in distros if d not in all_fit_results]
    if missing:
        raise KeyError(f"Run has no fit results for {missing} (has: {list(all_fit_results.keys())}).")
    filtered = MultiSeriesFitResults()
    for d in distros:
        filtered[d] = all_fit_results[d]
    return filtered.bake()


# The lognormal kernel `examples/generate_synthetic_data.py` convolves the
# synthetic admissions curve with (as of the current script). If that
# generator's parameters ever change, this constant must be updated to match
# -- there is no shared single source of truth between the two scripts, and
# `tab_synthetic_recovery.csv` would otherwise silently compare against a
# stale ground truth. See `examples/generate_synthetic_data.py`,
# `generate_and_save_synthetic_data()`.
SYNTHETIC_TRUE_SIGMA = 1.2
SYNTHETIC_TRUE_MU = 3.0

DEFAULT_WINDOWS_TO_SHOW = figure_2_module.DEFAULT_WINDOWS_TO_SHOW
DEFAULT_DISTROS_TO_SHOW = figure_2_module.DEFAULT_DISTROS_TO_SHOW

# The author-confirmed definitive case-study run (2026-08-31) -- see
# `text/claude_prompts/generate-paper-figures.md`. Resolved relative to
# `_CODE_ROOT`, not the working directory, so running this file directly from
# an IDE (whatever its cwd) resolves the same paths as `python -m
# scripts.generate_paper_figures` from `code/`. Override with --real-run /
# --synthetic-run / --out for a new data version; these three are the only
# things you need to change to re-point this script.
DEFAULT_REAL_RUN = (
    _CODE_ROOT
    / "results"
    / "260831_1307_dev_step7_train102_test21_fit_admissions_mse_reuse_last_parametrization_iterative_kernel_fit"
)
# Not author-confirmed like DEFAULT_REAL_RUN -- picked because it's the
# closest same-day synthetic run and its tab_synthetic_recovery.csv numbers
# matched the manuscript's pre-existing table within noise. Re-point this if
# the author names a different synthetic run.
DEFAULT_SYNTHETIC_RUN = _CODE_ROOT / "examples" / "results" / "260902_0735_dev_step7_train120_test21_fit_admissions_mse"

DEFAULT_OUT = _CODE_ROOT / "results" / "_paper_figures_staging"


def _staging_output_config(out_dir: Path) -> SimpleNamespace:
    """A minimal stand-in for `OutputFolderConfig` that routes every
    visualizer's save path (figures/metrics/animation) to one staging folder,
    instead of writing back into the source run's own results folder.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    p = str(out_dir)
    return SimpleNamespace(figures=p, metrics=p, animation=p, model_data=p, results=p)


def _load_run(path) -> LosEstimationRun:
    run = LosEstimationRun.load_run(path)
    if run.series_data is None or run.all_fit_results is None:
        raise ValueError(
            f"'{path}' does not look like a completed los_estimator run "
            "(missing model_data/series_data.pkl or all_fit_results.pkl)."
        )
    # Headless, force-save regardless of what the original run's TOML said.
    run.visualization_config.show_figures = False
    run.visualization_config.save_figures = True
    return run


def generate_figure_2(real_run, out_dir: Path, windows, distros, kernel_window, written, skipped):
    """`fig:training` -> `figure_2.pdf`."""
    if real_run is None:
        skipped.append("figure_2 (fig:training): skipped, no --real-run given")
        return
    fig = figure_2_module.build_figure(
        real_run, windows_to_show=windows, distros_to_show=distros, kernel_window=kernel_window
    )
    figure_2_module.save_figure(fig, out_dir, basename="figure_2", html=False, pdf=True)
    written.append(str(out_dir / "figure_2.pdf"))


def _plot_mse_mae_boxplots(metrics_test_csv: Path, distros, dst_path: Path):
    """Panel A source image: one axes, Test MSE and Test MAE boxplots side by side per
    distribution, outliers omitted -- MSE on the left y-scale, MAE on a twinned right
    y-scale (their magnitudes differ too much to share one axis).

    Reads both metrics directly from `metrics_test.csv` (written for every
    run regardless of `model_config.error_fun`, see
    `los_estimator.evaluation.Evaluator.save_result`) rather than reusing
    `MultiSeriesFitResults.test_errors_by_distro` (which only ever holds
    whatever single metric the run was optimized against) -- this way panel A
    shows MSE and MAE for a run even if it wasn't an MSE-optimized run.
    """
    from matplotlib.patches import Patch

    df = pd.read_csv(metrics_test_csv)
    df = df[df["distribution"].isin(distros)]

    n = len(distros)
    positions = np.arange(n) + 1
    width = 0.32
    edge_color = "black"

    fig, ax_mse = plt.subplots(figsize=(12, 3.5))
    ax_mae = ax_mse.twinx()

    mse_data = [df[(df["metric"] == "mse") & (df["distribution"] == d)]["value"].values for d in distros]
    mae_data = [df[(df["metric"] == "mae") & (df["distribution"] == d)]["value"].values for d in distros]

    # MSE vs MAE distinguished by hatch pattern (not color) plus their fixed
    # left/right offset -- color alone isn't accessible to colorblind readers.
    box_style = dict(
        patch_artist=True,
        showfliers=False,
        widths=width,
        medianprops=dict(color=edge_color),
        whiskerprops=dict(color=edge_color),
        capprops=dict(color=edge_color),
    )
    ax_mse.boxplot(
        mse_data,
        positions=positions - width / 2,
        boxprops=dict(facecolor="white", edgecolor=edge_color),
        **box_style,
    )
    ax_mae.boxplot(
        mae_data,
        positions=positions + width / 2,
        boxprops=dict(facecolor="white", edgecolor=edge_color, hatch="///"),
        **box_style,
    )

    ax_mse.set_xlim(0.5, n + 0.5)
    ax_mse.set_xticks(positions)
    ax_mse.set_xticklabels([d.capitalize() for d in distros], rotation=0, fontsize=18)
    ax_mse.set_ylabel("MSE")
    ax_mae.set_ylabel("MAE")
    # MSE spans hundreds of thousands here; a fixed power-of-ten scaling
    # factor (shown as the axis offset text) keeps tick labels short instead
    # of printing every digit.
    ax_mse.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax_mse.set_title("Test Error")
    ax_mse.grid(True, axis="y", alpha=0.3)
    ax_mse.legend(
        handles=[
            Patch(facecolor="white", edgecolor=edge_color, label="MSE"),
            Patch(facecolor="white", edgecolor=edge_color, hatch="///", label="MAE"),
        ],
        loc="upper right",
    )

    fig.tight_layout()
    fig.savefig(dst_path, bbox_inches="tight")
    plt.close(fig)


# Font/line sizes tuned so panels stay legible once printed at the paper's
# full text width (~5.5in per stacked panel) -- see
# `text/claude_prompts/build-figure3-output-grid.md`. Default matplotlib
# sizes (~10pt) were sized for on-screen viewing of a 2x2 grid and become
# illegible once each panel is shrunk further to fit a printed page.
FIGURE_3_RC = {
    "font.size": 15,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "lines.linewidth": 2.2,
    "lines.markersize": 6,
}


def generate_figure_3_output_grid(
    run: LosEstimationRun, out_dir: Path, written, distros=CASE_STUDY_DISTROS, kernel_distro="exponential"
):
    """`fig:error` -> `figure_3.png`: stacked (single-column) grid of example visualization-module outputs.

    Composites four already-rendered outputs into one labeled (A-D) stack,
    rather than a single MSE boxplot, so the figure demonstrates the range of
    output types the software produces (this is a SoftwareX software-
    description paper, not the results paper) instead of one result-style
    plot that overlaps with `tab:coverage`. See
    `text/claude_prompts/build-figure3-output-grid.md` for the rationale.

    Stacked single-column (not 2x2) so each panel prints at the paper's full
    text width instead of half of it -- at 2x2, each panel printed under 3in
    wide and axis/legend/tick text sized for on-screen viewing became
    illegible on paper. `FIGURE_3_RC` additionally scales up font/line sizes
    for the same reason.

    Panel A: Test MSE and Test MAE boxplots on one axes (MSE left scale, MAE
        right scale, twinned), outliers omitted (`_plot_mse_mae_boxplots`) --
        kept because the manuscript prose immediately after the figure
        refers to "boxplots of MSE", and the prose already separately claims
        MAE is computed alongside MSE.
    Panel B: `coverage_over_time.png`, coverage subplot only
        (`CoveragePlots.plot_coverage_over_time(include_band_width=False)`)
        -- empirical UQ-band coverage per rolling window. The "mean band
        width over time" subplot this used to include is dropped: it added a
        second dense subplot that didn't fit legibly at print size. This
        panel used to be the standalone `figure_4.png`/`fig:coverage_over_time`
        -- folded in here instead of kept as its own figure, since it is now
        shown at this grid's panel B; `los-paper.tex` must drop the old
        standalone occurrence when this changes. Requires
        `run.uncertainty_config.enabled = True`; raises rather than silently
        producing a 3-panel grid if UQ was off for this run.
    Panel C: `prediction_error_<kernel_distro>_fit.png`
        (`DeconvolutionPlots.show_error_windows`) -- one distribution's
        predictions and errors across all rolling windows.
    Panel D: `rolling_kernels_<kernel_distro>.png`
        (`DeconvolutionPlots.superimpose_kernels`) -- the same distribution's
        fitted kernels across rolling windows, y-limits auto-scaled to the
        actual kernel range instead of a fixed (-0.005, 0.3).

    Panel C drops its title entirely (`include_title=False`): with panel D
    directly below showing its own "All Rolling Exponential Kernels" title,
    a repeated "Exponential Distribution" line added nothing. Panel D omits
    the run-name subtitle its underlying method prints by default
    (`include_run_name=False`) -- an internal run-folder name (e.g.
    `260831_1307_dev_step7_..._iterative_kernel_fit`) is not informative in a
    print figure and only ate space needed for legible text.

    `kernel_distro` (default "exponential", the distribution the manuscript
    prose already singles out as a strong performer) drives both panel C and
    D, so they show the same distribution's kernels and predictions. All
    panels are filtered to `distros` (default: `CASE_STUDY_DISTROS`, the same
    five-distribution comparison `tab_coverage.csv` uses). PNG, not PDF --
    `file_format="pdf"` is reserved for `figure_2` (see `generate_figure_2`).
    """
    if kernel_distro not in distros:
        raise KeyError(f"kernel_distro={kernel_distro!r} not in distros={distros!r}.")
    if not run.uncertainty_config.enabled:
        raise ValueError(
            "figure_3 (fig:error) panel B needs coverage_over_time.png, which needs "
            "uncertainty_config.enabled = true on the real-data run."
        )
    filtered = _filter_fit_results(run.all_fit_results, distros)
    scratch = out_dir / "_scratch_figure_3"
    scratch.mkdir(parents=True, exist_ok=True)
    plots = DeconvolutionPlots(
        filtered,
        run.series_data,
        run.model_config,
        run.visualization_config,
        run.visualization_context,
        _staging_output_config(scratch),
    )

    from los_estimator.visualization import CoveragePlots

    metrics_test_path = Path(run.output_config.metrics) / "metrics_test.csv"
    if not metrics_test_path.exists():
        raise FileNotFoundError(f"Expected {metrics_test_path} (per-window test metrics) not found.")

    coverage_detail_path = Path(run.output_config.metrics) / "ci_coverage_detail.csv"
    if not coverage_detail_path.exists():
        raise FileNotFoundError(
            f"'{run.output_config.metrics}' has uncertainty_config.enabled = true but no "
            f"ci_coverage_detail.csv -- the run does not look complete."
        )
    detail = pd.read_csv(coverage_detail_path)
    detail = detail[detail["distribution"].isin(distros)]
    cov_plots = CoveragePlots(detail, run.visualization_config, _staging_output_config(scratch))

    with plt.rc_context(FIGURE_3_RC):
        _plot_mse_mae_boxplots(metrics_test_path, distros, scratch / "test_error_mse_mae.png")
        # Same month/year ticks (and the same [1::2] thinning) panel C's
        # prediction/error plot uses (DeconvolutionPlots._ax_plot_prediction_error_window),
        # so the two time-axis panels read consistently instead of B showing
        # raw window numbers while C shows calendar dates.
        cov_plots.plot_coverage_over_time(
            include_band_width=False,
            xtick_pos=run.visualization_context.xtick_pos[1::2],
            xtick_label=run.visualization_context.xtick_label[1::2],
        )
        plots.show_error_windows(kernel_distro, include_title=False, include_xlabel=False)
        plots.superimpose_kernels(
            kernel_distro, include_run_name=False, figsize=(10, 3.3), ylabel_fontsize=12
        )

    if not (scratch / "coverage_over_time.png").exists():
        raise FileNotFoundError(
            "CoveragePlots did not produce coverage_over_time.png -- every window's coverage "
            "may be NaN (no windowed UQ band at all)."
        )

    panel_files = [
        ("A", scratch / "test_error_mse_mae.png"),
        ("B", scratch / "coverage_over_time.png"),
        ("C", scratch / f"prediction_error_{kernel_distro}_fit.png"),
        ("D", scratch / f"rolling_kernels_{kernel_distro}.png"),
    ]
    for label, path in panel_files:
        if not path.exists():
            raise FileNotFoundError(f"Panel {label} ({path}) was not produced.")

    # Composited with PIL rather than matplotlib.image.imread: the source
    # panels are already ~300 dpi PNGs (several thousand px wide each), and
    # loading four of those as matplotlib's float32 RGBA arrays for imshow()
    # needs several hundred MB and can exceed memory in constrained
    # environments. PIL keeps everything as uint8 and lets each panel be
    # downsized to its target cell size before compositing, not after.
    from PIL import Image, ImageDraw, ImageFont

    panel_width = 1600
    margin = 8
    label_box = 56

    panels = []
    for label, path in panel_files:
        img = Image.open(path).convert("RGB")
        w, h = img.size
        panel_height = round(panel_width * h / w)
        img = img.resize((panel_width, panel_height), Image.LANCZOS)
        panels.append((label, img))

    # Stacked single column, one panel per row -- each panel then prints at
    # the paper's full text width instead of half of it (see docstring).
    # Small margin between panels: most of each panel image is already
    # whitespace from matplotlib's own layout, so a small compositor margin
    # keeps the stack from getting needlessly tall. The A/B/C/D label is
    # layered directly on each panel's top-left corner (not in its own
    # whitespace strip) to avoid adding height; panel D's ylabel font is
    # shrunk at the source (see `superimpose_kernels(ylabel_fontsize=...)`
    # below) so it doesn't collide with the label box on that panel's now
    # much shorter aspect ratio.
    canvas_w = panel_width + 2 * margin
    canvas_h = sum(img.height for _, img in panels) + (len(panels) + 1) * margin
    canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arialbd.ttf", 40)
    except OSError:
        font = ImageFont.load_default()

    y = margin
    for label, img in panels:
        x = margin
        canvas.paste(img, (x, y))
        draw.rectangle([x, y, x + label_box, y + label_box], fill="white")
        draw.text((x + 12, y + 4), label, fill="black", font=font)
        y += img.height + margin

    dst = out_dir / "figure_3.png"
    canvas.save(dst)
    written.append(str(dst))
    del canvas, panels
    shutil.rmtree(scratch, ignore_errors=True)


def generate_figure_4_animation_frame(run: LosEstimationRun, out_dir: Path, window_id: Optional[int], written, skipped):
    """`fig:animation_frame` -> `figure_4.png`: one rolling-window animation frame.

    Named `figure_4` (not `figure_5`) because `fig:coverage_over_time` (the
    former `figure_4.png`) was folded into `figure_3.png`'s panel D and
    dropped as a standalone figure -- `fig:animation_frame` is now the
    document's 4th figure, and filenames follow document order (see the
    module docstring).

    Kept as a high-resolution PNG rather than forced through PDF -- see the
    docstring on `DeconvolutionAnimator.save_n_show_animation_frame` in
    `los_estimator/visualization/animators.py` for why. Rendered directly
    with the animator's plotting primitives rather than
    `animate_fit_deconvolution()`/`save_n_show_animation_frame()`, because
    those write into `visualization_context.animation_folder` (the *original*
    run's animation folder, restored from the pickle, which this script
    intentionally does not touch) and because rendering all windows just to
    keep one frame would be wasteful.
    """
    if window_id is None:
        skipped.append(
            "figure_4 (fig:animation_frame): skipped, no --animation-frame-window given "
            "(ask the author which rolling window is 'peak epidemic conditions')"
        )
        return
    # Local import: DeconvolutionAnimator pulls in imageio/tqdm plumbing this
    # script does not otherwise need.
    from los_estimator.visualization import DeconvolutionAnimator
    from los_estimator.config import AnimationConfig

    animator = DeconvolutionAnimator(
        all_fit_results=run.all_fit_results,
        series_data=run.series_data,
        model_config=run.model_config,
        visualization_config=run.visualization_config,
        visualization_context=run.visualization_context,
        animation_config=AnimationConfig(show_figures=False, save_figures=False),
        output_folder_config=_staging_output_config(out_dir / "_scratch_figure_4"),
    )
    n_windows = len(run.series_data.window_infos)
    if not (0 <= window_id < n_windows):
        raise ValueError(f"--animation-frame-window {window_id} out of range (run has {n_windows} windows).")

    fig, ax_main, ax_kernel, ax_err_train, ax_err_test = animator._get_subplots()
    animator._plot_ax_main(ax_main, window_id)
    animator._plot_ax_kernel(ax_kernel, window_id)
    animator._plot_ax_errors(ax_err_train, ax_err_test, window_id)
    import matplotlib.pyplot as plt

    plt.suptitle(
        f"{run.model_config.run_name.replace('_', ' ')}\n\nDeconvolution Training Process",
        fontsize=16,
    )
    plt.tight_layout()

    dst = out_dir / "figure_4.png"
    fig.savefig(dst, bbox_inches="tight", dpi=max(300, run.visualization_config.savefig_dpi))
    plt.close(fig)
    shutil.rmtree(out_dir / "_scratch_figure_4", ignore_errors=True)
    written.append(str(dst))


def generate_figure_6_synthetic_kernel_recovery(run: LosEstimationRun, out_dir: Path, written, skipped):
    """`fig:synthetic_kernel_recovery` -> `figure_6.png`: rolling lognormal kernels vs. true kernel.

    No longer referenced in `los-paper.tex` (dropped 2026-09-01, redundant
    with `tab:synthetic-recovery`'s numbers) -- kept as a generator in case a
    future figure needs it, but not copied into `text/tex/`. PNG, not PDF --
    see `generate_figure_3_output_grid`.
    """
    if run.visualization_context.real_los is None:
        skipped.append(
            "figure_6 (fig:synthetic_kernel_recovery): skipped, synthetic run has no real_los "
            "(true generating kernel) set"
        )
        return
    if "lognorm" not in run.all_fit_results:
        skipped.append("figure_6 (fig:synthetic_kernel_recovery): skipped, synthetic run has no 'lognorm' fit results")
        return
    scratch = out_dir / "_scratch_figure_6"
    plots = DeconvolutionPlots(
        run.all_fit_results,
        run.series_data,
        run.model_config,
        run.visualization_config,
        run.visualization_context,
        _staging_output_config(scratch),
    )
    plots.superimpose_kernels("lognorm")
    src = scratch / "rolling_kernels_lognorm.png"
    dst = out_dir / "figure_6.png"
    shutil.copyfile(src, dst)
    shutil.rmtree(scratch, ignore_errors=True)
    written.append(str(dst))


def generate_tab_coverage(run: LosEstimationRun, out_dir: Path, written, skipped, distros=CASE_STUDY_DISTROS):
    """`tab:coverage`: per-distribution band rate / effective coverage, pooled.

    Filtered to `distros` (default: the five the Configuration section
    describes) -- see `generate_figure_3_output_grid`.
    """
    if not run.uncertainty_config.enabled:
        skipped.append("tab_coverage.csv: skipped, real run has uncertainty_config.enabled = false")
        return
    src = Path(run.output_config.metrics) / "ci_coverage.csv"
    if not src.exists():
        raise FileNotFoundError(f"Expected {src} (real-data UQ run summary) not found.")
    summary = pd.read_csv(src)
    summary = summary[summary["distribution"].isin(distros)]
    missing = set(distros) - set(summary["distribution"])
    if missing:
        raise KeyError(f"'{src}' has no coverage row for {sorted(missing)}.")
    table = summary[["distribution", "band_rate", "mean_coverage_effective"]].copy()
    table["band_rate_pct"] = (table["band_rate"] * 100).round(1)
    table["coverage_effective_pct"] = (table["mean_coverage_effective"] * 100).round(1)
    table = table[["distribution", "band_rate_pct", "coverage_effective_pct"]]
    dst = out_dir / "tab_coverage.csv"
    table.to_csv(dst, index=False)
    written.append(str(dst))


def _mean_los_from_kernel(kernel: np.ndarray) -> float:
    """Mean length of stay implied by a discharge-probability kernel."""
    days = np.arange(len(kernel))
    return float(np.sum(kernel * days))


def generate_tab_synthetic_recovery(run: LosEstimationRun, out_dir: Path, written, skipped):
    """`tab:synthetic-recovery`: true vs. fitted lognormal kernel parameters."""
    if "lognorm" not in run.all_fit_results:
        skipped.append("tab_synthetic_recovery.csv: skipped, synthetic run has no 'lognorm' fit results")
        return
    fit_results = run.all_fit_results["lognorm"].fit_results
    params = np.array([fr.distro_params for fr in fit_results])  # (n_windows, 2): sigma, mu
    kernels = np.array([fr.kernel for fr in fit_results])

    fitted_sigma = float(np.mean(params[:, 0]))
    fitted_mu = float(np.mean(params[:, 1]))
    fitted_mean_los = float(np.mean([_mean_los_from_kernel(k) for k in kernels]))

    true_kernel = Distributions.generate_kernel(
        "lognorm", [SYNTHETIC_TRUE_SIGMA, SYNTHETIC_TRUE_MU], kernel_size=run.model_config.kernel_width
    )
    true_mean_los = _mean_los_from_kernel(np.asarray(true_kernel))

    def rel_diff_pct(true, fitted):
        return abs(fitted - true) / abs(true) * 100.0

    rows = [
        {
            "parameter": "shape_sigma",
            "true_value": SYNTHETIC_TRUE_SIGMA,
            "fitted_mean": round(fitted_sigma, 4),
            "relative_diff_pct": round(rel_diff_pct(SYNTHETIC_TRUE_SIGMA, fitted_sigma), 4),
        },
        {
            "parameter": "log_scale_location_mu",
            "true_value": round(SYNTHETIC_TRUE_MU, 5),
            "fitted_mean": round(fitted_mu, 5),
            "relative_diff_pct": round(rel_diff_pct(SYNTHETIC_TRUE_MU, fitted_mu), 4),
        },
        {
            "parameter": "mean_length_of_stay_days",
            "true_value": round(true_mean_los, 4),
            "fitted_mean": round(fitted_mean_los, 4),
            "relative_diff_pct": round(rel_diff_pct(true_mean_los, fitted_mean_los), 4),
        },
    ]
    dst = out_dir / "tab_synthetic_recovery.csv"
    pd.DataFrame(rows).to_csv(dst, index=False)
    written.append(str(dst))


def run(
    real_run: Optional[str],
    synthetic_run: Optional[str],
    out: str,
    animation_frame_window: Optional[int] = None,
    windows=DEFAULT_WINDOWS_TO_SHOW,
    distros=DEFAULT_DISTROS_TO_SHOW,
    kernel_window: Optional[int] = None,
    case_study_distros=CASE_STUDY_DISTROS,
):
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list = []
    skipped: list = []

    generate_figure_2(real_run, out_dir, windows, distros, kernel_window, written, skipped)

    if real_run is not None:
        real = _load_run(real_run)
        generate_figure_3_output_grid(real, out_dir, written, distros=case_study_distros)
        generate_figure_4_animation_frame(real, out_dir, animation_frame_window, written, skipped)
        generate_tab_coverage(real, out_dir, written, skipped, distros=case_study_distros)
    else:
        skipped.append("figure_3 (fig:error): skipped, no --real-run given")
        skipped.append("figure_4 (fig:animation_frame): skipped, no --real-run given")
        skipped.append("tab_coverage.csv: skipped, no --real-run given")

    if synthetic_run is not None:
        synthetic = _load_run(synthetic_run)
        generate_figure_6_synthetic_kernel_recovery(synthetic, out_dir, written, skipped)
        generate_tab_synthetic_recovery(synthetic, out_dir, written, skipped)
    else:
        skipped.append("figure_6 (fig:synthetic_kernel_recovery): skipped, no --synthetic-run given")
        skipped.append("tab_synthetic_recovery.csv: skipped, no --synthetic-run given")

    print("\n=== generate_paper_figures summary ===")
    print(f"Output folder: {out_dir}")
    print("Written:")
    for f in written:
        print(f"  - {f}")
    print("Skipped:")
    for s in skipped:
        print(f"  - {s}")
    return written, skipped


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--real-run",
        default=str(DEFAULT_REAL_RUN),
        help="Real-data (DIVI/RKI) results folder (default: the author-confirmed case-study run)",
    )
    parser.add_argument(
        "--synthetic-run",
        default=str(DEFAULT_SYNTHETIC_RUN),
        help="synthetic_example.py results folder (default: unconfirmed best-guess pairing, see module docstring)",
    )
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="Staging output folder")
    parser.add_argument(
        "--animation-frame-window",
        type=int,
        default=4,
        help=(
            "Window index for figure_4/fig:animation_frame ('peak epidemic conditions'). Default 4 "
            "is the author's answer -- beginning of December (2021) -- for the run used when this "
            "default was set; re-verify against the definitive case-study run's own "
            "window-to-date mapping once available, since the window/date correspondence "
            "depends on that run's step size and start date."
        ),
    )
    parser.add_argument("--windows", type=int, nargs="+", default=list(DEFAULT_WINDOWS_TO_SHOW))
    parser.add_argument("--distros", nargs="+", default=list(DEFAULT_DISTROS_TO_SHOW), help="figure_2 only")
    parser.add_argument("--kernel-window", type=int, default=None)
    parser.add_argument(
        "--case-study-distros",
        nargs="+",
        default=list(CASE_STUDY_DISTROS),
        help="figure_3/figure_4/tab_coverage.csv only -- the Configuration section's five",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    run(
        real_run=_resolve_from_original_cwd(args.real_run) if args.real_run else None,
        synthetic_run=_resolve_from_original_cwd(args.synthetic_run) if args.synthetic_run else None,
        out=_resolve_from_original_cwd(args.out),
        animation_frame_window=args.animation_frame_window,
        windows=args.windows,
        distros=args.distros,
        kernel_window=args.kernel_window,
        case_study_distros=args.case_study_distros,
    )


if __name__ == "__main__":
    main()
