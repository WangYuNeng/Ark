# Visualze loss during optimization with weights-and-bias downloaded data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["text.usetex"] = True


# Remove the columns ends with _MAX or _MIN and remove Step column
def remove_redudant_columns(run_csv: pd.DataFrame) -> pd.DataFrame:
    run_csv = run_csv.loc[:, ~run_csv.columns.str.endswith("_MAX")]
    run_csv = run_csv.loc[:, ~run_csv.columns.str.endswith("_MIN")]
    run_csv = run_csv.loc[:, ~run_csv.columns.str.endswith("Step")]
    return run_csv


markers = ["o", "v", "D", "^", "s", "<", ">", "p", "P", "*", "X", "d"]
colors = ["b", "g", "tab:orange", "c", "y", "m", "r", "k", "w"]


def plot_runs(
    run_names: list[str],
    ylabel: str,
    save_path: str = None,
    method: str = "mean",
    window_size: int = 1,
    baseline_values: list[float] = None,
    markevery: int = 4,
    ylog_scale: bool = False,
    figsize: tuple[int, int] = (5, 4),
    label_fontsize: int = 24,
    tick_fontsize: int = 22,
    cmp_final_x: int = 10,
):
    """Visualize the optimization process of multiple runs

    Args:
        run_names (list[str]): file paths to the runs
        ylabel (str): y-axis label
        save_path (str, optional): location to save the plot. Defaults to None.
        method (str, optional): plot mean or min within a run_file. Defaults to "mean".
        window_size (int, optional): running average window size. Defaults to 1.
        baseline_values (list[float], optional): Show horizontal baseline. Defaults to None.
        markevery (int, optional): line marker showing period. Defaults to 4.
        ylog_scale (bool, optional): whether to make y-axis log scale. Defaults to False.
        figsize (tuple[int, int], optional): figure size. Defaults to (5, 4).
        label_fontsize (int, optional): font size of the xy-axes labels. Defaults to 16.
        tick_fontsize (int, optional): font size of the xy-axes ticks. Defaults to 14.
        cmp_final_x (int, optional): If use min, how many runs to average before comparison. Defaults to 10.
    """
    plt.figure(figsize=figsize)  # Set a larger figure size
    for i, (run_name) in enumerate(run_names):
        run = remove_redudant_columns(pd.read_csv(run_name))
        color, marker = colors[i], markers[i]
        if method == "mean":
            data_point = run.mean(axis=1)
        if method == "min":
            # use the run with the lowest final test loss (average of last 10 values)
            final_x_losses = run.iloc[-cmp_final_x:, :].mean(axis=0)
            min_loss_idx = final_x_losses.argmin()
            data_point = run.iloc[:, min_loss_idx]
            print(
                f"{run_name} best run name: {run.columns[min_loss_idx]}, loss ~= {final_x_losses.min()}"
            )

        steps = np.arange(0, len(data_point))

        if baseline_values and baseline_values[i]:
            # Plot the baseline with a dashed line
            plt.axhline(
                baseline_values[i],
                xmin=0.02,
                xmax=0.98,
                linestyle="--",
                color=color,
                linewidth=2,
            )

        if window_size > 1:
            # Plot the smoothened curve solid line and markers
            # Plot the original curve as dimmed line
            smooth_data_point = np.convolve(
                data_point, np.ones(window_size) / window_size, mode="valid"
            )

            # Pad the smooth points with first data point to match the length
            pad = np.ones(window_size - 1) * data_point[0]
            smooth_data_point = np.concatenate([pad, smooth_data_point])
            plt.plot(
                steps,
                smooth_data_point,
                marker=marker,
                color=color,
                markevery=markevery,
                linewidth=2,
                markersize=8,
            )
            plt.plot(steps, data_point, alpha=0.5, linewidth=1, color=color)
        else:
            plt.plot(
                steps,
                data_point,
                marker=marker,
                color=color,
                markevery=markevery,
                linewidth=2,
                markersize=8,
            )
        i += 1

    # Enhancing aesthetics
    plt.xlabel("Steps", fontsize=label_fontsize)  # Larger label font size
    plt.ylabel(ylabel, fontsize=label_fontsize)  # Larger label font size
    if ylog_scale:
        plt.yscale("log")

    # Adjust tick size
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    # Add a grid with a finer style
    plt.grid(True, which="both", linestyle="--", linewidth=0.7, alpha=0.7)

    # Modify legend
    # plt.legend(loc='best', fontsize=16, frameon=True)

    # Adjust layout to prevent cutting off labels
    plt.tight_layout()

    # Show the plot
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
