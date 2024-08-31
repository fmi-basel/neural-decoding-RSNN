import matplotlib.pyplot as plt
import seaborn as sns

import torch
import numpy as np
from scipy.signal import find_peaks

import stork
from stork.plotting import add_xscalebar

# prevent matplotlib from spamming the console
import logging

logging.getLogger("matplotlib").setLevel(logging.ERROR)

def plot_training(
    results,
    nb_epochs,
    names=[
        "loss",
        "r2",
    ],
    save_path=None,
):
    fig, ax = plt.subplots(
        2,
        len(names),
        figsize=(7, 3),
        dpi=150,
        sharex=True,
        sharey="col",
        constrained_layout=True,
    )

    for i, n in enumerate(names):

        for j, s in enumerate(["x", "y"]):
            if "loss" in n:
                ax[0][i].plot(
                    results["train_{}".format(n)], color="black", label="train"
                )
                ax[0][i].plot(
                    results["val_{}".format(n)], color="black", alpha=0.5, label="valid"
                )

                ax[0][i].scatter(
                    nb_epochs,
                    results["test_{}".format(n)],
                    color="crimson",
                )
                ax[0][i].set_ylabel(n)
            else:
                ax[j][i].plot(
                    results["train_{}{}".format(n, s)], color="black", label="train"
                )
                ax[j][i].plot(
                    results["val_{}{}".format(n, s)],
                    color="black",
                    alpha=0.5,
                    label="valid",
                )

                ax[j][i].scatter(
                    nb_epochs,
                    results["test_{}{}".format(n, s)],
                    color="crimson",
                    label="test",
                )

                ax[j][i].set_ylabel("{} {}".format(n, s))
            
            if "r2" in n:
                ax[j][i].set_ylim(0, 1)

            ax[1][i].set_xlabel("Epochs")

    ax[0][-1].legend()
    ax[1][0].axis("off")
    ax[0][0].set_xlabel("Epochs")

    sns.despine()

    if save_path is not None:
        fig.savefig(save_path, dpi=250)
    return fig, ax


def plot_cumulative_mse(model, dataset, n_samples=5, save_path=None):

    fig, ax = plt.subplots(
        4, n_samples, figsize=(2 * n_samples, 3), dpi=250, sharex=True, sharey="row"
    )
    preds = model.predict(dataset)
    targets = dataset.labels

    for s in range(5):
        curr_preds = preds[s]
        curr_targets = targets[s]

        for i, idx in enumerate([0, 2]):
            ax[idx][s].hlines(0, len(curr_preds), 0, color="silver")
            ax[idx][s].plot(
                curr_targets[:, i], c="crimson", label="target", alpha=0.75, lw=1
            )
            ax[idx][s].plot(
                curr_preds[:, i], c="k", label="prediction", alpha=0.5, lw=1
            )

            # plot cumulative mse loss
            cs_se = np.cumsum((curr_targets[:, i] - curr_preds[:, i]) ** 2)
            ax[idx + 1][s].plot(cs_se / cs_se[-1], c="k")

            ax[0][s].set_title(f"$v_x$, cs_se = {cs_se[-1].item():.04f}")
            ax[2][s].set_title(f"$v_y$, cs_se = {cs_se[-1].item():.04f}")

            # Compute the absolute values of the segment
            abs_segment = torch.abs(curr_targets[:, i])

            # Find peaks in the absolute values

            peaks, _ = find_peaks(abs_segment.numpy())

            # threshold the peaks
            peaks = peaks[abs_segment.numpy()[peaks] > 0.25]

            # Plot the peaks
            ax[idx][s].vlines(peaks, -1.5, 1.5, color="silver", alpha=0.5)
            ax[idx + 1][s].vlines(peaks, 0, 1, color="silver", alpha=0.5)

            ax[idx][s].set_ylim(-1.5, 1.5)
            ax[idx + 1][s].set_ylim(0, 1)

        ax[0][0].set_ylabel(f"$v_x$")
        ax[1][0].set_ylabel("Cum.\nSE x")
        ax[2][0].set_ylabel(f"$v_y$")
        ax[3][0].set_ylabel("Cum.\nSE y")

    ax[0][-1].legend()

    sns.despine()
    if save_path is not None:
        fig.savefig(save_path, dpi=250)
    return fig, ax


def turn_axis_off(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)


def plot_activity_CST(
    model,
    data,
    nb_samples=5,
    figsize=(10, 5),
    dpi=250,
    marker=".",
    point_size=5,
    point_alpha=1,
    pos=(0, -1),
    off=(0, -0.05),
    turn_ro_axis_off=True,
):
    #print("plotting CST snapshot")

    # Run model once and get activities
    scores = model.evaluate(data)

    inp = model.input_group.get_flattened_out_sequence().detach().cpu().numpy()
    hidden_groups = model.groups[1:-1]
    hid_activity = [
        g.get_flattened_out_sequence().detach().cpu().numpy() for g in hidden_groups
    ]
    out_group = model.out.detach().cpu().numpy()
    labels = [l for d, l in data]

    nb_groups = len(hidden_groups)
    nb_total_units = (
        np.sum([g.nb_units for g in hidden_groups]) + model.input_group.nb_units
    )
    hr = (
        [4 * model.input_group.nb_units / nb_total_units]
        + [4 * g.nb_units / nb_total_units for g in hidden_groups]
        + [0.5, 0.5]
    )
    hr = list(reversed(hr))  # since we are plotting from bottom to top

    fig, ax = plt.subplots(
        nb_groups + 3,
        nb_samples,
        figsize=figsize,
        dpi=dpi,
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": hr},
    )

    for i in range(nb_samples):
        # plot input spikes
        ax[-1][i].scatter(
            np.where(inp[i])[0],
            np.where(inp[i])[1],
            s=point_size / 2,
            marker=marker,
            color="k",
            alpha=point_alpha,
        )

        turn_axis_off(ax[-1][i])

        # plot hidden layer spikes
        for g in range(nb_groups):
            ax[-(2 + g)][i].scatter(
                np.where(hid_activity[g][i])[0],
                np.where(hid_activity[g][i])[1],
                s=point_size / 2,
                marker=marker,
                color="k",
                alpha=point_alpha,
            )
            turn_axis_off(ax[-(2 + g)][i])

            ax[-(2 + g)][0].set_ylabel(
                hidden_groups[g].name
                if hidden_groups[g].name is not None
                else "Hid. %i" % g
            )

        for line_index, ro_line in enumerate(np.transpose(out_group[i])):
            ax[line_index][i].plot(
                labels[i][:, line_index],
                color="crimson",
                label="label"
            )
            ax[line_index][i].plot(ro_line, color="k", alpha=0.5, label="ro")
            if turn_ro_axis_off:
                turn_axis_off(ax[0][i])
                turn_axis_off(ax[1][i])
            
        ax[0][-1].legend()

    dur_50 = 50e-3 / model.time_step
    # print(dur_10)
    add_xscalebar(ax[-1][0], dur_50, label="50ms", pos=pos, off=off, fontsize=8)

    ax[-1][0].set_ylabel("Input")
    ax[0][0].set_ylabel(f"$v_X$")
    ax[1][0].set_ylabel(f"$v_Y$")
    #plt.tight_layout()
    
    return fig, ax

def plot_activity_snapshot(model, data, save_path=None):

    fig, ax = plot_activity_CST(
        model,
        data=data,
        figsize=(10, 5),
        dpi=250,
        pos=(0, 0),
        off=(0.0, -0.05),
        turn_ro_axis_off=True,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=250)

    return fig, ax
