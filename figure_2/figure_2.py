# %%
# %load_ext autoreload
# %autoreload 2
"""Builds the "training and prediction" plotly figure used as `figure_2.pdf`
in the SoftwareX paper (`fig:training` in `text/tex/los-paper.tex`).

Loads a completed `los_estimator` run via `LosEstimationRun.load_run()` and
plots real ICU occupancy, a few rolling-window train/test fits, and the
kernels fitted at one window, for a chosen subset of distributions.

Originally a notebook-style script with hardcoded run paths (`a`/`b`) and
hardcoded `windows_to_show`/`distros_to_show`. Refactored into an importable
`build_figure()` plus a CLI so `code/scripts/generate_paper_figures.py` (and
any other caller) can point it at any results folder without editing this
file.

CLI usage:
    python figure_2.py --run <results_folder> --windows 3 17 28 \
        --distros gaussian exponential cauchy linear compartmental \
        --out <output_dir>
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from los_estimator.estimation_run import LosEstimationRun
from los_estimator.fitting.distributions import Distributions

DEFAULT_WINDOWS_TO_SHOW = (3, 17, 28)
DEFAULT_DISTROS_TO_SHOW = ["gaussian", "exponential", "cauchy", "linear", "compartmental"]


def build_figure(
    run_path,
    windows_to_show: Sequence[int] = DEFAULT_WINDOWS_TO_SHOW,
    distros_to_show: Sequence[str] = DEFAULT_DISTROS_TO_SHOW,
    kernel_window: Optional[int] = None,
) -> go.Figure:
    """Build the training/prediction plotly figure for one results folder.

    Args:
        run_path: Path to a `los_estimator` results folder (as accepted by
            `LosEstimationRun.load_run`).
        windows_to_show: Window indices whose train/test fits are overlaid
            on the main occupancy plot.
        distros_to_show: Which fitted distributions to include, both in the
            main plot's legend and the per-distribution kernel subplots.
        kernel_window: Window index used for the "Estimated Kernels at
            position n=<kernel_window>" subplots. Defaults to
            `windows_to_show[0]` if not given, matching the original script's
            behavior (it reused the first shown window for the kernel plots).

    Returns:
        go.Figure: The assembled plotly figure. Caller is responsible for
        writing it out (`fig.write_html` / `fig.write_image`).
    """
    run_path = Path(run_path)
    estimator = LosEstimationRun.load_run(run_path)
    estimator.evaluate()

    windows_to_show = tuple(windows_to_show)
    distros_to_show = list(distros_to_show)
    if kernel_window is None:
        kernel_window = windows_to_show[0]

    data_package = estimator.evaluator.window_data_package
    vc = estimator.visualization_context
    distros = list(estimator.all_fit_results.keys())

    r_kernel = 3
    h1 = 0.345
    wide_plot = [
        {
            "colspan": 2,
        },
        None,
    ]
    fig = make_subplots(
        rows=4,
        cols=2,
        specs=[
            wide_plot,
            wide_plot,
            [{}, {}],
            [{}, {}],
        ],
        row_heights=[0.5, 0, 0.15, 0.15],
        subplot_titles=["Training and Prediction", ""] + [d for d in distros if d not in ["t", "compartmental"]],
        vertical_spacing=0.05,
    )
    fig.add_annotation(
        text=f"Estimated Kernels at position n={kernel_window}",
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.41,
        showarrow=False,
        font=dict(size=16),
    )

    ###############################################################################
    #################  Real Data  #################################################
    ###############################################################################

    x_full = np.arange(len(data_package.series_data.y_full))
    fig.add_trace(
        go.Scatter(
            x=x_full,
            y=data_package.series_data.y_full,
            mode="lines",
            line=dict(color="black"),
            name="Real ICU Occupancy",
            opacity=0.8,
            showlegend=True,
            legendgroup="real",
            legendrank=1,
        )
    )
    train_legend_shown = False

    ###############################################################################
    #################  Predictions  ###############################################
    ###############################################################################
    first_window = True
    for i_window in windows_to_show:
        for i_distro, (distro, fit_res) in enumerate(list(estimator.all_fit_results.items())):
            if distro not in distros_to_show:
                continue
            # ensure we don't repeat legend entries for many windows
            d = data_package.data[i_distro][i_window]
            fit_result = estimator.all_fit_results[distro].fit_results[i_window]

            _, y_pred_train, _, y_pred_test, x_train, x_test, w = d

            fig.add_trace(
                go.Scatter(
                    x=x_train,
                    y=y_pred_train,
                    mode="lines",
                    line=dict(color="orange"),
                    name="Training Fits",
                    legendgroup=f"traces",
                    showlegend=not train_legend_shown and first_window,
                    legendrank=2,
                ),
                col=1,
                row=1,
            )
            train_legend_shown = True

            fig.add_trace(
                go.Scatter(
                    x=x_test,
                    y=y_pred_test,
                    mode="lines",
                    legendgroup=f"traces",
                    legendgrouptitle_text="Models",
                    name=f"{distro.capitalize()} Models",
                    showlegend=first_window,
                    legendrank=2,
                )
            )
        first_window = False

    ###############################################################################
    #################  Rolling Windows  ###########################################
    ###############################################################################
    for i_window in windows_to_show:
        y = h1 + 0.155  # ensure we don't repeat legend entries for many windows
        d = data_package.data[0][i_window]

        _, y_pred_train, _, y_pred_test, x_train, x_test, w = d
        for x in (x_train[0], x_train[-1], x_test[-1]):
            # draw a vertical line that spans (and slightly extends) the plotting area so it appears
            # under axis ticks/labels, and put it below data traces
            fig.add_shape(
                type="line",
                x0=x,
                x1=x,
                xref="x",
                y0=y,
                y1=1,
                yref="paper",
                line=dict(color="gray", dash="dash"),
                opacity=0.8,
                legendrank=3,
            )
        fig.add_shape(
            type="rect",
            x0=x_train[0],
            x1=x_train[-1],
            xref="x",
            y0=y,
            y1=1,
            yref="paper",
            fillcolor="orange",
            opacity=0.2,
            line=dict(width=0),
            layer="below",
        )
        fig.add_shape(
            type="rect",
            x0=x_test[0],
            x1=x_test[-1],
            xref="x",
            y0=y,
            y1=1,
            yref="paper",
            fillcolor="blue",
            opacity=0.2,
            line=dict(width=0),
            layer="below",
        )
        # Add text annotations for train and test regions
        fig.add_annotation(
            text=f"Rolling window {i_window}",
            x=(x_train[0] + x_test[-1]) / 2,
            y=y + 0.105,
            yref="paper",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.8)",
        )
        fig.add_annotation(
            text="Train",
            x=(x_train[0] + x_train[-1]) / 2,
            y=y + 0.03,
            yref="paper",
            showarrow=False,
        )
        fig.add_annotation(
            text="Pred",
            x=(x_test[0] + x_test[-1]) / 2,
            y=y + 0.03,
            yref="paper",
            showarrow=False,
        )
        y_below = y - 0.012
        n = "n"
        li = [
            (f"t{n}-Δtrain", x_train[0]),
            (f"t{n}", x_test[0]),
            (f"t{n}+Δpred", x_test[-1]),
        ]
        for text, x in li:
            fig.add_annotation(
                text=text,
                x=x,
                y=y_below,
                yref="paper",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="lightgray",
            )
    ###############################################################################
    #################  LoS Distros  ###############################################
    ###############################################################################

    w = 2
    dd = list(estimator.all_fit_results.items())
    dd = [x for x in dd if x[0] not in ["t", "compartmental"]]
    dd = [x for x in dd if x[0] in distros_to_show]
    for i_distro, (distro, fit_res) in enumerate(dd):
        d = data_package.data[i_distro][kernel_window]
        fit_result = estimator.all_fit_results[distro].fit_results[kernel_window]
        kernel = fit_result.kernel
        x_kernel = np.arange(len(kernel))
        fig.add_trace(
            go.Scatter(
                x=x_kernel,
                y=kernel,
                mode="lines",
                showlegend=False,
            ),
            row=r_kernel + i_distro // w,
            col=1 + i_distro % w,
        )
        param_str = ""
        if distro != "compartmental":
            param_str = Distributions.to_string(distro, fit_result.distro_params).replace(",", "<br>")
            fig.add_annotation(
                text=param_str,
                x=60,
                y=0.051,
                xanchor="right",
                yanchor="top",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="lightgray",
                borderwidth=1,
                borderpad=4,
                font=dict(size=9),
                row=r_kernel + i_distro // w,
                col=1 + i_distro % w,
                align="right",  # Align text to the right
            )

    for i in range(r_kernel, r_kernel + 2):
        for j in range(1, 3):
            fig.update_yaxes(
                col=j,
                row=i,
                gridwidth=1,
                gridcolor="lightgray",
                range=(-0.005, 0.051),
                showline=True,
                linewidth=1,
                linecolor="lightgray",
                mirror=True,
            )
        fig.update_xaxes(
            col=j,
            row=i,
            gridwidth=1,
            gridcolor="lightgray",
            range=(0, 60),
            showline=True,
            linewidth=1,
            linecolor="lightgray",
            mirror=True,
        )

    fig.add_annotation(
        text="Discharge Probability",
        xref="paper",
        yref="paper",
        x=-0.075,
        y=0.02,
        showarrow=False,
        font=dict(size=14),
        textangle=-90,
    )

    for i in range(1, 4):
        fig.update_xaxes(
            col=i,
            row=r_kernel + 1,
            title_text="Days since admission",
            title_standoff=0,
        )

    # Add borders to all kernel subplots
    fig.update_xaxes(
        row=r_kernel,
        showline=True,
        linewidth=1,
        linecolor="lightgray",
        mirror=True,
    )
    fig.update_xaxes(
        row=r_kernel + 1,
        showline=True,
        linewidth=1,
        linecolor="lightgray",
        mirror=True,
    )
    fig.update_yaxes(
        row=r_kernel,
        showline=True,
        linewidth=1,
        linecolor="lightgray",
        mirror=True,
    )
    fig.update_yaxes(
        row=r_kernel + 1,
        showline=True,
        linewidth=1,
        linecolor="lightgray",
        mirror=True,
    )

    # Add border to the main plot
    fig.update_xaxes(
        row=1,
        col=1,
        showline=True,
        linewidth=1,
        linecolor="lightgray",
        mirror=True,
    )
    fig.update_yaxes(
        row=1,
        col=1,
        showline=True,
        linewidth=1,
        linecolor="lightgray",
        mirror=True,
    )

    ###############################################################################
    #################  General stuff  #############################################
    ###############################################################################

    # place legend at top-right of the first (top) subplot
    fig.update_layout(
        legend=dict(
            x=0.99,
            y=0.99,
            xanchor="right",
            yanchor="top",
            bordercolor="lightgray",
            borderwidth=1,
        )
    )

    xaxis = dict(
        range=(50, 399),
        tickmode="array",
        tickvals=vc.xtick_pos[1:],
        ticktext=[element.replace("\n", " ") for element in vc.xtick_label[1:]],
        gridcolor="lightgray",
    )

    fig.update_layout(
        height=700,
        width=900,
        xaxis=xaxis,
        yaxis=dict(range=[-100, 5200], title="ICU occupancy", title_standoff=0),
        title=dict(
            text="Rolling Training and Prediction of ICU Occupancy with Different LoS Distributions",
            y=0.93,
            x=0.5,
            xanchor="center",
        ),
        template="plotly_white",
    )

    fig.update_xaxes(col=1, row=1, **xaxis)

    return fig


def save_figure(fig: go.Figure, output_dir, basename: str = "figure_2", html: bool = True, pdf: bool = True) -> None:
    """Write `fig` to `<output_dir>/<basename>.html` and/or `.pdf`."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if html:
        html_path = output_dir / f"{basename}.html"
        print(f"Saving figure as html: {html_path}")
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
    if pdf:
        pdf_path = output_dir / f"{basename}.pdf"
        print(f"Saving figure as pdf: {pdf_path}")
        fig.write_image(str(pdf_path))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="Path to a los_estimator results folder to load")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOWS_TO_SHOW),
        help=f"Window indices to overlay on the main plot (default: {DEFAULT_WINDOWS_TO_SHOW})",
    )
    parser.add_argument(
        "--distros",
        nargs="+",
        default=list(DEFAULT_DISTROS_TO_SHOW),
        help=f"Distributions to include (default: {DEFAULT_DISTROS_TO_SHOW})",
    )
    parser.add_argument(
        "--kernel-window",
        type=int,
        default=None,
        help="Window index used for the kernel subplots (default: first --windows value)",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent),
        help="Output directory for figure_2.html/figure_2.pdf (default: this script's folder)",
    )
    parser.add_argument("--show", action="store_true", help="Also open the figure interactively (fig.show())")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    fig = build_figure(
        run_path=args.run,
        windows_to_show=args.windows,
        distros_to_show=args.distros,
        kernel_window=args.kernel_window,
    )
    save_figure(fig, args.out)
    if args.show:
        fig.show()
    print("done")


if __name__ == "__main__":
    main()
