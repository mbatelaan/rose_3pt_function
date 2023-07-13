import numpy as np
from pathlib import Path

import pickle
import csv
import scipy.optimize as syopt
import matplotlib.pyplot as plt

from plot_utils import save_plot
from formatting import err_brackets
from analysis.evxptreaders import evxptdata
from analysis.bootstrap import bootstrap
from analysis import stats
from analysis import fitfunc
import plot_functions as plots

from gevpanalysis.util import read_config
from threept_analysis.twoexpfitting import get_Qsquared_values

_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}

_colors = [
    "#377eb8",
    "#4daf4a",
    "#f781bf",
    "#a65628",
    "#ff7f00",
    "#984ea3",
    "#999999",
    "#e41a1c",
    "#dede00",
]

_fmts = ["s", "^", "o", "p", "x", "v", "P", ",", "*", "."]
# _fmts = ["s", "p", "x", "^", "o", "v", "P", ",", "*", "."]


def read_pickle(filename, nboot=200, nbin=1):
    """Get the data from the pickle file and output a bootstrapped numpy array.

    The output is a numpy matrix with:
    axis=0: bootstraps
    axis=2: time axis
    axis=3: real & imaginary parts
    """
    with open(filename, "rb") as file_in:
        data = pickle.load(file_in)
    bsdata = bootstrap(data, config_ax=0, nboot=nboot, nbin=nbin)
    return bsdata


def plot_ratio_fit(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(16, 9))
    for icorr, corr in enumerate(ratios):
        plot_time2 = time - (src_snk_times[icorr]) / 2
        ydata = np.average(corr, axis=0)
        yerror = np.std(corr, axis=0)
        plot_x_values = (
            np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
            - (src_snk_times[icorr]) / 2
        )
        tau_values = np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
        t_values = np.array([src_snk_times[icorr]] * len(tau_values))

        step_indices = [
            0,
            src_snk_times[0] + 1 - (2 * delta_t),
            src_snk_times[0] + 1 + src_snk_times[1] + 1 - (4 * delta_t),
            src_snk_times[0]
            + 1
            + src_snk_times[1]
            + 1
            + src_snk_times[2]
            + 1
            - (6 * delta_t),
        ]

        axarr[icorr].errorbar(
            plot_time2[1 : src_snk_times[icorr]],
            ydata[1 : src_snk_times[icorr]],
            yerror[1 : src_snk_times[icorr]],
            capsize=4,
            elinewidth=1,
            color=_colors[icorr],
            fmt=_fmts[icorr],
            label=labels[icorr],
        )
        axarr[icorr].plot(
            plot_x_values,
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            color=_colors[3],
        )
        axarr[icorr].fill_between(
            plot_x_values,
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            )
            - np.std(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            )
            + np.std(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            alpha=0.3,
            linewidth=0,
            color=_colors[3],
        )
        axarr[icorr].axhline(
            np.average(fit_param_boot[:, 0]),
            color=_colors[5],
            label=rf"fit = {err_brackets(np.average(fit_param_boot[:, 0]), np.std(fit_param_boot[:, 0]))}",
        )

        plot_time3 = np.array([-20, 20])
        axarr[icorr].fill_between(
            plot_time3,
            [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            alpha=0.3,
            linewidth=0,
            color=_colors[5],
        )

        # axarr[icorr].grid(True)
        axarr[icorr].legend(fontsize=15, loc="upper left")
        axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(
            r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
        )
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(-src_snk_times[-1] - 1, src_snk_times[-1] + 1)
        axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)

    f.suptitle(
        rf"{plotparam[3]} 3-point function ratio with $\hat{{\mathcal{{O}}}}=${plotparam[1]}, $\Gamma = ${plotparam[2]}, $\vec{{q}}\, ={plotparam[0][1:]}$ with two-state fit $\chi^2_{{\mathrm{{dof}}}}={redchisq:.2f}$"
    )
    savefile = plotdir / Path(f"{title}_full.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit_comp_paper(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_data,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t=10$",
        r"$t=13$",
        r"$t=16$",
    ]

    # f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 5))
    # f, axarr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(8, 5))
    f, axarr = plt.subplots(1, 4, sharex=False, sharey=True, figsize=(8, 5))
    f.subplots_adjust(wspace=0, bottom=0.2)
    for icorr, corr in enumerate(ratios):
        plot_time2 = time - (src_snk_times[icorr]) / 2
        ydata = np.average(corr, axis=0)
        yerror = np.std(corr, axis=0)
        plot_x_values = (
            np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
            - (src_snk_times[icorr]) / 2
        )
        tau_values = np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
        t_values = np.array([src_snk_times[icorr]] * len(tau_values))

        step_indices = [
            0,
            src_snk_times[0] + 1 - (2 * delta_t),
            src_snk_times[0] + 1 + src_snk_times[1] + 1 - (4 * delta_t),
            src_snk_times[0]
            + 1
            + src_snk_times[1]
            + 1
            + src_snk_times[2]
            + 1
            - (6 * delta_t),
        ]

        axarr[icorr].errorbar(
            plot_time2[1 : src_snk_times[icorr]],
            ydata[1 : src_snk_times[icorr]],
            yerror[1 : src_snk_times[icorr]],
            capsize=4,
            elinewidth=1,
            color=_colors[icorr],
            fmt=_fmts[icorr],
            # label=labels[icorr],
        )
        axarr[icorr].plot(
            plot_x_values,
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            color=_colors[3],
        )
        axarr[icorr].fill_between(
            plot_x_values,
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            )
            - np.std(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            )
            + np.std(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            alpha=0.3,
            linewidth=0,
            color=_colors[3],
        )
        axarr[icorr].axhline(
            np.average(fit_param_boot[:, 0]),
            color=_colors[5],
            # label=rf"fit = {err_brackets(np.average(fit_param_boot[:, 0]), np.std(fit_param_boot[:, 0]))}",
        )

        plot_time3 = np.array([-20, 20])
        axarr[icorr].fill_between(
            plot_time3,
            [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            alpha=0.3,
            linewidth=0,
            color=_colors[5],
        )

        axarr[icorr].set_title(labels[icorr])
        axarr[icorr].set_xlabel(r"$\tau-t/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(r"$R(\vec{p}\, ; t, \tau)$", labelpad=5, fontsize=18)
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)
        axarr[icorr].set_xlim(-10, 10)

    efftime = np.arange(63)
    axarr[3].errorbar(
        efftime[:20],
        np.average(FH_data["deltaE_eff"], axis=0)[:20],
        np.std(FH_data["deltaE_eff"], axis=0)[:20],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
        # label=labels[icorr],
    )
    axarr[3].plot(
        # FH_data["ratio_t_range"],
        plot_time3,
        [np.average(FH_data["deltaE_fit"])]
        * len(plot_time3),  # * len(FH_data["ratio_t_range"]),
        color=_colors[3],
    )
    axarr[3].fill_between(
        # FH_data["ratio_t_range"],
        plot_time3,
        [np.average(FH_data["deltaE_fit"]) - np.std(FH_data["deltaE_fit"])]
        * len(plot_time3),
        # * len(FH_data["ratio_t_range"]),
        [np.average(FH_data["deltaE_fit"]) + np.std(FH_data["deltaE_fit"])]
        * len(plot_time3),
        # * len(FH_data["ratio_t_range"]),
        alpha=0.3,
        linewidth=0,
        color=_colors[7],
    )

    # axarr[3].axhline(
    #     np.average(FH_data["FH_matrix_element"]),
    #     color=_colors[6],
    # )
    # axarr[3].fill_between(
    #     plot_time3,
    #     [
    #         np.average(FH_data["FH_matrix_element"])
    #         - np.std(FH_data["FH_matrix_element"])
    #     ]
    #     * len(plot_time3),
    #     [
    #         np.average(FH_data["FH_matrix_element"])
    #         + np.std(FH_data["FH_matrix_element"])
    #     ]
    #     * len(plot_time3),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[6],
    # )

    axarr[3].set_xlabel(r"$t$", labelpad=14, fontsize=18)
    axarr[3].set_title(r"\textrm{FH}")
    axarr[3].set_xlim(-1, 15)
    axarr[3].set_ylim(0.4, 1)
    axarr[3].set_xticks([0, 5, 10])
    axarr[0].set_xticks([-5, 0, 5])
    axarr[1].set_xticks([-5, 0, 5])
    axarr[2].set_xticks([-5, 0, 5])

    savefile = plotdir / Path(f"{title}.pdf")
    savefile2 = plotdir / Path(f"{title}.png")
    savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.savefig(savefile2, dpi=500)
    plt.savefig(savefile3, dpi=100)
    # plt.ylim(1.104, 1.181)
    # plt.savefig(savefile_ylim)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit_comp_paper_separate(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_data,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t=10$",
        r"$t=13$",
        r"$t=16$",
    ]

    # Fit a constant to the ratios
    t10_ratio_data = ratios[0][:, delta_t:-delta_t]
    t10_const_fit = np.average(t10_ratio_data, axis=1)

    t13_ratio_data = ratios[1][:, delta_t:-delta_t]
    fitparam_t13 = stats.fit_bootstrap(
        fitfunc.constant,
        [1],
        np.arange(len(t13_ratio_data[0])),
        t13_ratio_data,
        bounds=None,
        time=False,
        fullcov=False,
    )
    t13_const_fit = fitparam_t13["param"]
    print(f"{np.average(t13_const_fit)=}")

    t16_ratio_data = ratios[2][:, delta_t:-delta_t]
    print(f"\n{np.shape(t16_ratio_data)=}\n")
    print(f"\n{np.average(t16_ratio_data, axis=0)=}\n")
    fitparam_t16 = stats.fit_bootstrap(
        fitfunc.constant,
        np.array([1]),
        np.arange(len(t16_ratio_data[0])),
        t16_ratio_data,
        bounds=None,
        time=False,
    )
    t16_const_fit = fitparam_t16["param"]
    print(f"{np.average(t16_const_fit)=}")

    # f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 5))
    # f, axarr = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(8, 5))
    f, axarr = plt.subplots(1, 4, sharex=False, sharey=True, figsize=(8, 5))
    f.subplots_adjust(wspace=0, bottom=0.2)
    for icorr, corr in enumerate(ratios):
        plot_time2 = time - (src_snk_times[icorr]) / 2
        ydata = np.average(corr, axis=0)
        yerror = np.std(corr, axis=0)
        plot_x_values = (
            np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
            - (src_snk_times[icorr]) / 2
        )
        tau_values = np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
        t_values = np.array([src_snk_times[icorr]] * len(tau_values))

        step_indices = [
            0,
            src_snk_times[0] + 1 - (2 * delta_t),
            src_snk_times[0] + 1 + src_snk_times[1] + 1 - (4 * delta_t),
            src_snk_times[0]
            + 1
            + src_snk_times[1]
            + 1
            + src_snk_times[2]
            + 1
            - (6 * delta_t),
        ]

        axarr[icorr].errorbar(
            plot_time2[1 : src_snk_times[icorr]],
            ydata[1 : src_snk_times[icorr]],
            yerror[1 : src_snk_times[icorr]],
            capsize=4,
            elinewidth=1,
            color=_colors[icorr],
            fmt=_fmts[icorr],
            # label=labels[icorr],
        )
        axarr[icorr].plot(
            plot_x_values,
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            color=_colors[3],
        )
        axarr[icorr].fill_between(
            plot_x_values,
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            )
            - np.std(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            np.average(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            )
            + np.std(
                ratio_fit[:, step_indices[icorr] : step_indices[icorr + 1]], axis=0
            ),
            alpha=0.3,
            linewidth=0,
            color=_colors[3],
        )
        axarr[icorr].axhline(
            np.average(fit_param_boot[:, 0]),
            color=_colors[5],
            # label=rf"fit = {err_brackets(np.average(fit_param_boot[:, 0]), np.std(fit_param_boot[:, 0]))}",
        )

        plot_time3 = np.array([-20, 20])
        axarr[icorr].fill_between(
            plot_time3,
            [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
            * len(plot_time3),
            alpha=0.3,
            linewidth=0,
            color=_colors[5],
        )

        axarr[icorr].set_title(labels[icorr])
        axarr[icorr].set_xlabel(r"$\tau-t/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(r"$R(\vec{p}\, ; t, \tau)$", labelpad=5, fontsize=18)
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)
        axarr[icorr].set_xlim(-10, 10)

    # Plot the fit results on the second subplot
    axarr[3].set_yticks([])
    axarr[3].errorbar(
        0,
        np.average(t10_const_fit),
        np.std(t10_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
    )
    axarr[3].errorbar(
        1,
        np.average(t13_const_fit),
        np.std(t13_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
    )
    axarr[3].errorbar(
        2,
        np.average(t16_const_fit),
        np.std(t16_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
    )
    axarr[3].errorbar(
        3,
        np.average(fit_param_boot[:, 0]),
        np.std(fit_param_boot[:, 0]),
        capsize=4,
        elinewidth=1,
        color=_colors[5],
        fmt=_fmts[4],
    )
    axarr[3].errorbar(
        4,
        np.average(FH_data["deltaE_fit"]),
        np.std(FH_data["deltaE_fit"]),
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
    )
    axarr[3].set_xticks(
        [0, 1, 2, 3, 4],
    )
    axarr[3].set_xticklabels(
        [labels[0], labels[1], labels[2], r"2-exp", r"FH"],
        fontsize="x-small",
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axarr[3].tick_params(axis="x", which="minor", length=0)
    axarr[3].set_xlim(-0.8, 4.8)
    axarr[3].set_ylim(0.6, 0.9)

    # efftime = np.arange(63)
    # axarr[3].errorbar(
    #     efftime[:20],
    #     np.average(FH_data["deltaE_eff"], axis=0)[:20],
    #     np.std(FH_data["deltaE_eff"], axis=0)[:20],
    #     capsize=4,
    #     elinewidth=1,
    #     color=_colors[3],
    #     fmt=_fmts[3],
    #     # label=labels[icorr],
    # )
    # axarr[3].plot(
    #     # FH_data["ratio_t_range"],
    #     plot_time3,
    #     [np.average(FH_data["deltaE_fit"])]
    #     * len(plot_time3),  # * len(FH_data["ratio_t_range"]),
    #     color=_colors[3],
    # )
    # axarr[3].fill_between(
    #     # FH_data["ratio_t_range"],
    #     plot_time3,
    #     [np.average(FH_data["deltaE_fit"]) - np.std(FH_data["deltaE_fit"])]
    #     * len(plot_time3),
    #     # * len(FH_data["ratio_t_range"]),
    #     [np.average(FH_data["deltaE_fit"]) + np.std(FH_data["deltaE_fit"])]
    #     * len(plot_time3),
    #     # * len(FH_data["ratio_t_range"]),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[7],
    # )

    # axarr[3].axhline(
    #     np.average(FH_data["FH_matrix_element"]),
    #     color=_colors[6],
    # )
    # axarr[3].fill_between(
    #     plot_time3,
    #     [
    #         np.average(FH_data["FH_matrix_element"])
    #         - np.std(FH_data["FH_matrix_element"])
    #     ]
    #     * len(plot_time3),
    #     [
    #         np.average(FH_data["FH_matrix_element"])
    #         + np.std(FH_data["FH_matrix_element"])
    #     ]
    #     * len(plot_time3),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[6],
    # )

    # axarr[3].set_xlabel(r"$t$", labelpad=14, fontsize=18)
    # axarr[3].set_title(r"\textrm{FH}")
    # axarr[3].set_xlim(-1, 15)
    # axarr[3].set_ylim(0.4, 1.1)
    # axarr[3].set_xticks([0, 5, 10])

    axarr[0].set_xticks([-5, 0, 5])
    axarr[1].set_xticks([-5, 0, 5])
    axarr[2].set_xticks([-5, 0, 5])

    savefile = plotdir / Path(f"{title}_separate.pdf")
    savefile2 = plotdir / Path(f"{title}.png")
    savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.savefig(savefile2, dpi=500)
    # plt.savefig(savefile3, dpi=100)
    # plt.ylim(1.104, 1.181)
    # plt.savefig(savefile_ylim)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit_comp_paper_2(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_data,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(1, 1, figsize=(8, 5))
    # f, axarr = plt.subplots(1, 2, figsize=(8, 5), gridspec_kw={"width_ratios": [3, 1]})
    ydata0 = np.average(ratios[0], axis=0)
    yerror0 = np.std(ratios[0], axis=0)
    axarr.errorbar(
        time[1 : src_snk_times[0]],
        ydata0[1 : src_snk_times[0]],
        yerror0[1 : src_snk_times[0]],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
        label=labels[0],
    )

    ydata1 = np.average(ratios[1], axis=0)
    yerror1 = np.std(ratios[1], axis=0)
    axarr.errorbar(
        time[1 : src_snk_times[1]] + 0.07,
        ydata1[1 : src_snk_times[1]],
        yerror1[1 : src_snk_times[1]],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        label=labels[1],
    )

    ydata2 = np.average(ratios[2], axis=0)
    yerror2 = np.std(ratios[2], axis=0)
    axarr.errorbar(
        time[1 : src_snk_times[2]] + 0.14,
        ydata2[1 : src_snk_times[2]],
        yerror2[1 : src_snk_times[2]],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
        label=labels[2],
    )

    # tau_values = np.arange(src_snk_times[icorr] + 1)[delta_t:-delta_t]
    # t_values = np.array([src_snk_times[icorr]] * len(tau_values))

    step_indices = [
        0,
        src_snk_times[0] + 1 - (2 * delta_t),
        src_snk_times[0] + 1 + src_snk_times[1] + 1 - (4 * delta_t),
        src_snk_times[0]
        + 1
        + src_snk_times[1]
        + 1
        + src_snk_times[2]
        + 1
        - (6 * delta_t),
    ]
    plot_x_values0 = np.arange(src_snk_times[0] + 1)[delta_t:-delta_t]
    axarr.plot(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        color=_colors[0],
    )
    plot_x_values1 = np.arange(src_snk_times[1] + 1)[delta_t:-delta_t]
    # print(ratio_fit[:, step_indices[1] : step_indices[2]])
    # print(np.average(ratio_fit[:, step_indices[1] : step_indices[1]], axis=0))
    axarr.plot(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        color=_colors[1],
    )
    plot_x_values2 = np.arange(src_snk_times[2] + 1)[delta_t:-delta_t]
    axarr.plot(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        color=_colors[2],
    )

    axarr.fill_between(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        - np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        + np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    axarr.fill_between(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        - np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        + np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    axarr.fill_between(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        - np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        + np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[2],
    )
    #     axarr[icorr].axhline(
    #         np.average(fit_param_boot[:, 0]),
    #         color=_colors[5],
    #         # label=rf"fit = {err_brackets(np.average(fit_param_boot[:, 0]), np.std(fit_param_boot[:, 0]))}",
    #     )

    plot_time3 = np.array([0, 20])
    # axarr[icorr].fill_between(
    #     plot_time3,
    #     [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
    #     * len(plot_time3),
    #     [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
    #     * len(plot_time3),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[5],
    # )

    # axarr[icorr].set_title(labels[icorr])
    # axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
    # axarr[icorr].set_ylabel(
    #     r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
    # )
    # axarr[icorr].label_outer()
    # # axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)
    # axarr[icorr].set_xlim(-10, 10)

    efftime = np.arange(63)
    axarr.errorbar(
        efftime[:20] + 0.21,
        np.average(FH_data["deltaE_eff"], axis=0)[:20],
        np.std(FH_data["deltaE_eff"], axis=0)[:20],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
        label=r"$\textrm{FH}$",
    )
    # axarr.axhline(
    #     np.average(FH_data["deltaE_fit"]),
    #     color=_colors[7],
    # )
    # axarr.fill_between(
    #     plot_time3,
    #     [np.average(FH_data["deltaE_fit"]) - np.std(FH_data["deltaE_fit"])]
    #     * len(plot_time3),
    #     [np.average(FH_data["deltaE_fit"]) + np.std(FH_data["deltaE_fit"])]
    #     * len(plot_time3),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[7],
    # )

    axarr.plot(
        FH_data["ratio_t_range"],
        [FH_data["FH_matrix_element"]] * len(FH_data["ratio_t_range"]),
        color=_colors[3],
    )
    axarr.fill_between(
        FH_data["ratio_t_range"],
        [FH_data["FH_matrix_element"] - FH_data["FH_matrix_element_err"]]
        * len(FH_data["ratio_t_range"]),
        [FH_data["FH_matrix_element"] + FH_data["FH_matrix_element_err"]]
        * len(FH_data["ratio_t_range"]),
        alpha=0.3,
        linewidth=0,
        color=_colors[3],
    )
    # axarr.set_title("")
    axarr.set_xlim(0, 18)
    axarr.set_ylim(0.3, 1.1)

    plt.legend(fontsize="x-small")
    savefile = plotdir / Path(f"{title}_one.pdf")
    savefile2 = plotdir / Path(f"{title}.png")
    savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.savefig(savefile2, dpi=500)
    # plt.savefig(savefile3, dpi=100)
    # plt.ylim(1.104, 1.181)
    # plt.savefig(savefile_ylim)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit_comp_paper_3(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_data,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(
        1, 2, figsize=(7, 5), sharey=False, gridspec_kw={"width_ratios": [3, 1]}
    )
    f.subplots_adjust(wspace=0, bottom=0.15)

    # Plot the 3pt fn ratios
    offset = 0.08
    ydata0 = np.average(ratios[0], axis=0)
    yerror0 = np.std(ratios[0], axis=0)
    axarr[0].errorbar(
        time[1 : src_snk_times[0]],
        ydata0[1 : src_snk_times[0]],
        yerror0[1 : src_snk_times[0]],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
        label=labels[0],
    )
    ydata1 = np.average(ratios[1], axis=0)
    yerror1 = np.std(ratios[1], axis=0)
    axarr[0].errorbar(
        time[1 : src_snk_times[1]] + offset,
        ydata1[1 : src_snk_times[1]],
        yerror1[1 : src_snk_times[1]],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        label=labels[1],
    )
    ydata2 = np.average(ratios[2], axis=0)
    yerror2 = np.std(ratios[2], axis=0)
    axarr[0].errorbar(
        time[1 : src_snk_times[2]] + offset * 2,
        ydata2[1 : src_snk_times[2]],
        yerror2[1 : src_snk_times[2]],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
        label=labels[2],
    )

    # set the indices which split the fit data into t10, t13, t16
    step_indices = [
        0,
        src_snk_times[0] + 1 - (2 * delta_t),
        src_snk_times[0] + 1 + src_snk_times[1] + 1 - (4 * delta_t),
        src_snk_times[0]
        + 1
        + src_snk_times[1]
        + 1
        + src_snk_times[2]
        + 1
        - (6 * delta_t),
    ]

    # Fit a constant to the ratios
    # t10_ratio_data = ratios[0][:, step_indices[0] : step_indices[1]]
    # t10_const_fit = np.average(t10_ratio_data, axis=1)
    # t13_ratio_data = ratios[1][:, step_indices[1] : step_indices[2]]
    # t13_const_fit = np.average(t13_ratio_data, axis=1)
    # t16_ratio_data = ratios[2][:, step_indices[2] : step_indices[3]]
    # t16_const_fit = np.average(t16_ratio_data, axis=1)
    t10_ratio_data = ratios[0][:, delta_t:-delta_t]
    t10_const_fit = np.average(t10_ratio_data, axis=1)
    t13_ratio_data = ratios[1][:, delta_t:-delta_t]
    fitparam_t13 = stats.fit_bootstrap(
        fitfunc.constant,
        [1],
        np.arange(len(t13_ratio_data[0])),
        t13_ratio_data,
        bounds=None,
        time=False,
        fullcov=False,
    )
    t13_const_fit = fitparam_t13["param"]
    print(f"{np.average(t13_const_fit)=}")

    t16_ratio_data = ratios[2][:, delta_t:-delta_t]
    print(f"\n{np.shape(t16_ratio_data)=}\n")
    print(f"\n{np.average(t16_ratio_data, axis=0)=}\n")
    fitparam_t16 = stats.fit_bootstrap(
        fitfunc.constant,
        np.array([1]),
        np.arange(len(t16_ratio_data[0])),
        t16_ratio_data,
        bounds=None,
        time=False,
    )
    t16_const_fit = fitparam_t16["param"]
    print(f"{np.average(t16_const_fit)=}")

    # plot the two-exp fit results
    plot_x_values0 = np.arange(src_snk_times[0] + 1)[delta_t:-delta_t]
    axarr[0].plot(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        color=_colors[0],
    )
    plot_x_values1 = np.arange(src_snk_times[1] + 1)[delta_t:-delta_t]
    axarr[0].plot(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        color=_colors[1],
    )
    plot_x_values2 = np.arange(src_snk_times[2] + 1)[delta_t:-delta_t]
    axarr[0].plot(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        color=_colors[2],
    )

    # Plot the two-exp fits to the ratio
    axarr[0].fill_between(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        - np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        + np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    axarr[0].fill_between(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        - np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        + np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    axarr[0].fill_between(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        - np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        + np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[2],
    )

    # plot the Feynman-Hellmann data
    efftime = np.arange(63)
    axarr[0].errorbar(
        efftime[:20] + offset * 3,
        np.average(FH_data["deltaE_eff"], axis=0)[:20],
        np.std(FH_data["deltaE_eff"], axis=0)[:20],
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
        label=r"$\textrm{FH}$",
    )

    # axarr.axhline(
    #     np.average(FH_data["deltaE_fit"]),
    #     color=_colors[7],
    # )
    # axarr.fill_between(
    #     plot_time3,
    #     [np.average(FH_data["deltaE_fit"]) - np.std(FH_data["deltaE_fit"])]
    #     * len(plot_time3),
    #     [np.average(FH_data["deltaE_fit"]) + np.std(FH_data["deltaE_fit"])]
    #     * len(plot_time3),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[7],
    # )

    # plot the fit to the  Feynman-Hellmann data
    axarr[0].plot(
        FH_data["ratio_t_range"][:-2] + offset * 3,
        [np.average(FH_data["deltaE_fit"])] * len(FH_data["ratio_t_range"][:-2]),
        color=_colors[3],
    )
    axarr[0].fill_between(
        FH_data["ratio_t_range"][:-2] + offset * 3,
        # [FH_data["deltaE_fit"] - FH_data["deltaE_fit_err"]]
        [np.average(FH_data["deltaE_fit"]) - np.std(FH_data["deltaE_fit"])]
        * len(FH_data["ratio_t_range"][:-2]),
        [np.average(FH_data["deltaE_fit"]) + np.std(FH_data["deltaE_fit"])]
        # [FH_data["deltaE_fit"] + FH_data["deltaE_fit_err"]]
        * len(FH_data["ratio_t_range"][:-2]),
        alpha=0.3,
        linewidth=0,
        color=_colors[3],
    )

    # # plot the fit to the  Feynman-Hellmann data
    # axarr[0].plot(
    #     FH_data["ratio_t_range"][:-2] + offset * 3,
    #     [np.average(FH_data["FH_matrix_element"])] * len(FH_data["ratio_t_range"][:-2]),
    #     color=_colors[3],
    # )
    # axarr[0].fill_between(
    #     FH_data["ratio_t_range"][:-2] + offset * 3,
    #     # [FH_data["FH_matrix_element"] - FH_data["FH_matrix_element_err"]]
    #     [
    #         np.average(FH_data["FH_matrix_element"])
    #         - np.std(FH_data["FH_matrix_element"])
    #     ]
    #     * len(FH_data["ratio_t_range"][:-2]),
    #     [
    #         np.average(FH_data["FH_matrix_element"])
    #         + np.std(FH_data["FH_matrix_element"])
    #     ]
    #     # [FH_data["FH_matrix_element"] + FH_data["FH_matrix_element_err"]]
    #     * len(FH_data["ratio_t_range"][:-2]),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[3],
    # )

    axarr[0].set_xlim(0, 16)
    axarr[0].set_ylim(0.2, 1.1)
    axarr[0].set_xlabel(r"$t$", labelpad=14, fontsize=18)
    axarr[0].set_ylabel(
        r"$R(\vec{q}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
    )
    axarr[0].label_outer()
    axarr[0].legend(fontsize="x-small")

    # Plot the fit results on the second subplot
    axarr[1].set_yticks([])
    axarr[1].errorbar(
        0,
        np.average(t10_const_fit),
        np.std(t10_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
    )
    axarr[1].errorbar(
        1,
        np.average(t13_const_fit),
        np.std(t13_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
    )
    axarr[1].errorbar(
        2,
        np.average(t16_const_fit),
        np.std(t16_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
    )
    axarr[1].errorbar(
        3,
        np.average(fit_param_boot[:, 0]),
        np.std(fit_param_boot[:, 0]),
        capsize=4,
        elinewidth=1,
        color=_colors[4],
        fmt=_fmts[4],
    )
    axarr[1].errorbar(
        4,
        np.average(FH_data["deltaE_fit"]),
        np.std(FH_data["deltaE_fit"]),
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
    )
    # axarr[1].errorbar(
    #     4,
    #     np.average(FH_data["FH_matrix_element"]),
    #     np.std(FH_data["FH_matrix_element"]),
    #     capsize=4,
    #     elinewidth=1,
    #     color=_colors[3],
    #     fmt=_fmts[3],
    # )
    axarr[1].set_xticks(
        [0, 1, 2, 3, 4],
    )
    axarr[1].set_xticklabels(
        [labels[0], labels[1], labels[2], r"2-exp", r"FH"],
        fontsize="x-small",
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axarr[1].tick_params(axis="x", which="minor", length=0)
    axarr[1].set_xlim(-0.8, 4.8)
    axarr[1].set_ylim(0.2, 1.1)

    savefile = plotdir / Path(f"{title}_two.pdf")
    savefile2 = plotdir / Path(f"{title}.png")
    savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    plt.savefig(savefile2, dpi=500)
    plt.savefig(savefile3, dpi=100)
    # plt.ylim(1.104, 1.181)
    # plt.savefig(savefile_ylim)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit_comp_paper_4(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_data,
    title="",
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(
        1, 2, figsize=(7, 5), sharey=False, gridspec_kw={"width_ratios": [3, 1]}
    )
    f.subplots_adjust(wspace=0, bottom=0.15)

    # Plot the 3pt fn ratios
    ydata0 = np.average(ratios[0], axis=0)
    yerror0 = np.std(ratios[0], axis=0)
    plot_time0 = time - (src_snk_times[0]) / 2
    axarr[0].errorbar(
        plot_time0[1 : src_snk_times[0]],
        ydata0[1 : src_snk_times[0]],
        yerror0[1 : src_snk_times[0]],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
        label=labels[0],
    )
    ydata1 = np.average(ratios[1], axis=0)
    yerror1 = np.std(ratios[1], axis=0)
    plot_time1 = time - (src_snk_times[1]) / 2
    axarr[0].errorbar(
        plot_time1[1 : src_snk_times[1]],
        ydata1[1 : src_snk_times[1]],
        yerror1[1 : src_snk_times[1]],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        label=labels[1],
    )
    ydata2 = np.average(ratios[2], axis=0)
    yerror2 = np.std(ratios[2], axis=0)
    plot_time2 = time - (src_snk_times[2]) / 2
    axarr[0].errorbar(
        plot_time2[1 : src_snk_times[2]],
        ydata2[1 : src_snk_times[2]],
        yerror2[1 : src_snk_times[2]],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
        label=labels[2],
    )

    # set the indices which split the fit data into t10, t13, t16
    step_indices = [
        0,
        src_snk_times[0] + 1 - (2 * delta_t),
        src_snk_times[0] + 1 + src_snk_times[1] + 1 - (4 * delta_t),
        src_snk_times[0]
        + 1
        + src_snk_times[1]
        + 1
        + src_snk_times[2]
        + 1
        - (6 * delta_t),
    ]

    # Fit a constant to the ratios
    t10_ratio_data = ratios[0][:, delta_t:-delta_t]
    t10_const_fit = np.average(t10_ratio_data, axis=1)

    t13_ratio_data = ratios[1][:, delta_t:-delta_t]
    fitparam_t13 = stats.fit_bootstrap(
        fitfunc.constant,
        [1],
        np.arange(len(t13_ratio_data[0])),
        t13_ratio_data,
        bounds=None,
        time=False,
        fullcov=False,
    )
    t13_const_fit = fitparam_t13["param"]
    print(f"{np.average(t13_const_fit)=}")

    t16_ratio_data = ratios[2][:, delta_t:-delta_t]
    print(f"\n{np.shape(t16_ratio_data)=}\n")
    print(f"\n{np.average(t16_ratio_data, axis=0)=}\n")
    fitparam_t16 = stats.fit_bootstrap(
        fitfunc.constant,
        np.array([1]),
        np.arange(len(t16_ratio_data[0])),
        t16_ratio_data,
        bounds=None,
        time=False,
    )
    t16_const_fit = fitparam_t16["param"]
    print(f"{np.average(t16_const_fit)=}")

    # plot the two-exp fit results
    plot_x_values0 = (
        np.arange(src_snk_times[0] + 1)[delta_t:-delta_t] - (src_snk_times[0]) / 2
    )
    axarr[0].plot(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        color=_colors[0],
    )

    plot_x_values1 = (
        np.arange(src_snk_times[1] + 1)[delta_t:-delta_t] - (src_snk_times[1]) / 2
    )
    axarr[0].plot(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        color=_colors[1],
    )
    plot_x_values2 = (
        np.arange(src_snk_times[2] + 1)[delta_t:-delta_t] - (src_snk_times[2]) / 2
    )
    axarr[0].plot(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        color=_colors[2],
    )

    # Plot the two-exp fits to the ratio
    axarr[0].fill_between(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        - np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0)
        + np.std(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[0],
    )
    axarr[0].fill_between(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        - np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0)
        + np.std(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[1],
    )
    axarr[0].fill_between(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        - np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0)
        + np.std(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        alpha=0.3,
        linewidth=0,
        color=_colors[2],
    )

    axarr[0].set_xlim(-8, 8)
    axarr[0].set_xlabel(r"$\tau - t_{\textrm{sep}}/2$", labelpad=14, fontsize=18)
    axarr[0].set_ylabel(
        r"$R(\vec{q}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
    )
    axarr[0].label_outer()
    axarr[0].legend(fontsize="x-small")

    # Plot the fit results on the second subplot
    axarr[1].set_yticks([])
    axarr[1].errorbar(
        0,
        np.average(t10_const_fit),
        np.std(t10_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
    )
    axarr[1].errorbar(
        1,
        np.average(t13_const_fit),
        np.std(t13_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
    )
    axarr[1].errorbar(
        2,
        np.average(t16_const_fit),
        np.std(t16_const_fit),
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
    )
    axarr[1].errorbar(
        3,
        np.average(fit_param_boot[:, 0]),
        np.std(fit_param_boot[:, 0]),
        capsize=4,
        elinewidth=1,
        color=_colors[4],
        fmt=_fmts[4],
    )
    axarr[1].errorbar(
        4,
        np.average(FH_data["deltaE_fit"]),
        np.std(FH_data["deltaE_fit"]),
        capsize=4,
        elinewidth=1,
        color=_colors[3],
        fmt=_fmts[3],
    )
    # axarr[1].errorbar(
    #     4,
    #     np.average(FH_data["FH_matrix_element"]),
    #     np.std(FH_data["FH_matrix_element"]),
    #     capsize=4,
    #     elinewidth=1,
    #     color=_colors[3],
    #     fmt=_fmts[3],
    # )
    axarr[1].set_xticks(
        [0, 1, 2, 3, 4],
    )
    axarr[1].set_xticklabels(
        [labels[0], labels[1], labels[2], r"2-exp", r"FH"],
        fontsize="x-small",
        rotation=45,
        ha="right",
        rotation_mode="anchor",
    )
    axarr[1].tick_params(axis="x", which="minor", length=0)
    axarr[1].set_xlim(-0.8, 4.8)

    axarr[0].set_ylim(0.6, 0.9)
    axarr[1].set_ylim(0.6, 0.9)

    savefile = plotdir / Path(f"{title}_three.pdf")
    savefile2 = plotdir / Path(f"{title}.png")
    savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.savefig(savefile2, dpi=500)
    # plt.savefig(savefile3, dpi=100)
    # plt.ylim(1.104, 1.181)
    # plt.savefig(savefile_ylim)
    # plt.show()
    plt.close()
    return


def make_full_ratio(threeptfn, twoptfn_sigma_real, twoptfn_neutron_real, src_snk_time):
    """Make the ratio of two-point and three-point functions which produces the plateau"""
    sqrt_factor = np.sqrt(
        (
            twoptfn_sigma_real[:, : src_snk_time + 1]
            * twoptfn_neutron_real[:, src_snk_time::-1]
        )
        / (
            twoptfn_neutron_real[:, : src_snk_time + 1]
            * twoptfn_sigma_real[:, src_snk_time::-1]
        )
    )
    prefactor_full = np.einsum(
        "ij,i->ij",
        sqrt_factor,
        np.sqrt(
            twoptfn_sigma_real[:, src_snk_time] / twoptfn_neutron_real[:, src_snk_time]
        )
        / twoptfn_sigma_real[:, src_snk_time],
    )
    ratio = np.einsum("ijk,ij->ijk", threeptfn[:, : src_snk_time + 1], prefactor_full)
    return ratio


def main():
    plt.style.use("./mystyle.txt")
    plt.rc("text.latex", preamble=r"\usepackage{physics}")
    plt.rcParams.update({"figure.autolayout": False})

    # --- directories ---
    latticedir = Path.home() / Path("Documents/PhD/lattice_results/EMFF_3pt_function/")
    resultsdir = Path.home() / Path("Dropbox/PhD/analysis_code/rose_3pt_function/")
    plotdir = resultsdir / Path("plots/")
    plotdir2 = plotdir / Path("twopoint/")
    datadir = resultsdir / Path("data/")
    plotdir.mkdir(parents=True, exist_ok=True)
    plotdir2.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # Operators as defined in chroma
    operators_chroma = [
        "gI",
        "g0",
        "g1",
        "g01",
        "g2",
        "g02",
        "g12",
        "g53",
        "g3",
        "g03",
        "g13",
        "g25",
        "g23",
        "g51",
        "g05",
        "g5",
    ]
    operators_tex_chroma = [
        "I",
        "\gamma_1",
        "\gamma_2",
        "\gamma_1\gamma_2",
        "\gamma_3",
        "\gamma_1\gamma_3",
        "\gamma_2\gamma_3",
        "\gamma_5\gamma_4",
        "\gamma_4",
        "\gamma_1\gamma_4",
        "\gamma_2\gamma_4",
        "\gamma_3\gamma_5",
        "\gamma_3\gamma_4",
        "\gamma_5\gamma_2",
        "\gamma_1\gamma_5",
        "\gamma_5",
    ]

    # # ======================================================================
    # # Read in the three point function data
    # operators_tex = ["$\gamma_4$"]
    # operators = ["g3"]
    # polarizations = ["UNPOL"]
    # momenta = ["p+1+0+0"]
    # delta_t_list = [5]
    # tmin_choice = [7]

    # # ======================================================================
    # GE quark 1
    corr_mom0_quark1 = [
        {
            "chroma_index": 8,
            "op_name": "Vector4",
            "quark_flavour": "U",
            "pol": "UNPOL",
            "momentum": "p+0+0+0",
            "source_mom": "p+0+0+0",
            "sink_mom": "p+0+0+0",
            "snk_mom": np.array([0, 0, 0]),
            "mom": np.array([0, 0, 0]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]
    corr_mom1_quark1 = [
        {
            "chroma_index": 8,
            "op_name": "Vector4",
            "quark_flavour": "U",
            "pol": "UNPOL",
            "momentum": "p+2+0+0",
            "source_mom": "p+1+0+0",
            "sink_mom": "p+1+0+0",
            "snk_mom": np.array([1, 0, 0]),
            "mom": np.array([2, 0, 0]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]
    corr_mom2_quark1 = [
        {
            "chroma_index": 8,
            "op_name": "Vector4",
            "quark_flavour": "U",
            "pol": "UNPOL",
            "momentum": "p+2+2+2",
            "source_mom": "p+1+1+1",
            "sink_mom": "p+1+1+1",
            "snk_mom": np.array([1, 1, 1]),
            "mom": np.array([2, 2, 2]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]
    corr_mom3_quark1 = [
        {
            "chroma_index": 8,
            "op_name": "Vector4",
            "quark_flavour": "U",
            "pol": "UNPOL",
            "momentum": "p+4+2+0",
            "source_mom": "p+2+1+0",
            "sink_mom": "p+2+1+0",
            "snk_mom": np.array([2, 1, 0]),
            "mom": np.array([4, 2, 0]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]
    # GE Quark 2
    corr_mom0_quark2 = [
        {
            "chroma_index": 8,
            "op_name": "Vector4",
            "quark_flavour": "D",
            "pol": "UNPOL",
            "momentum": "p+0+0+0",
            "source_mom": "p+0+0+0",
            "sink_mom": "p+0+0+0",
            "snk_mom": np.array([0, 0, 0]),
            "mom": np.array([0, 0, 0]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]
    corr_mom1_quark2 = [
        {
            "chroma_index": 8,
            "op_name": "Vector4",
            "quark_flavour": "D",
            "pol": "UNPOL",
            "momentum": "p+2+0+0",
            "source_mom": "p+1+0+0",
            "sink_mom": "p+1+0+0",
            "snk_mom": np.array([1, 0, 0]),
            "mom": np.array([2, 0, 0]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]
    corr_mom2_quark2 = [
        {
            "chroma_index": 8,
            "op_name": "Vector4",
            "quark_flavour": "D",
            "pol": "UNPOL",
            "momentum": "p+2+2+2",
            "source_mom": "p+1+1+1",
            "sink_mom": "p+1+1+1",
            "snk_mom": np.array([1, 1, 1]),
            "mom": np.array([2, 2, 2]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]
    corr_mom3_quark2 = [
        {
            "chroma_index": 8,
            "op_name": "Vector4",
            "quark_flavour": "D",
            "pol": "UNPOL",
            "momentum": "p+4+2+0",
            "source_mom": "p+2+1+0",
            "sink_mom": "p+2+1+0",
            "snk_mom": np.array([2, 1, 0]),
            "mom": np.array([4, 2, 0]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]

    # ======================================================================
    # GM Quark 1
    corr_mom1_quark1_GM = [
        {
            "chroma_index": 2,
            "op_name": "Vector2",
            "quark_flavour": "U",
            "pol": "POL",
            "momentum": "p+2+0+0",
            "source_mom": "p+1+0+0",
            "sink_mom": "p+1+0+0",
            "snk_mom": np.array([1, 0, 0]),
            "mom": np.array([2, 0, 0]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]
    corr_mom2_quark1_GM = [
        {
            "chroma_index": 2,
            "op_name": "Vector2",
            "quark_flavour": "U",
            "pol": "POL",
            "momentum": "p+2+2+2",
            "source_mom": "p+1+1+1",
            "sink_mom": "p+1+1+1",
            "snk_mom": np.array([1, 1, 1]),
            "mom": np.array([2, 2, 2]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]
    # GM Quark 2
    corr_mom1_quark2_GM = [
        {
            "chroma_index": 2,
            "op_name": "Vector2",
            "quark_flavour": "D",
            "pol": "POL",
            "momentum": "p+2+0+0",
            "source_mom": "p+1+0+0",
            "sink_mom": "p+1+0+0",
            "snk_mom": np.array([1, 0, 0]),
            "mom": np.array([2, 0, 0]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]
    corr_mom2_quark2_GM = [
        {
            "chroma_index": 2,
            "op_name": "Vector2",
            "quark_flavour": "D",
            "pol": "POL",
            "momentum": "p+2+2+2",
            "source_mom": "p+1+1+1",
            "sink_mom": "p+1+1+1",
            "snk_mom": np.array([1, 1, 1]),
            "mom": np.array([2, 2, 2]),
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
    ]

    # ======================================================================
    # Read the data from the FH analysis
    FH_datadir = Path.home() / Path(
        "Documents/PhD/analysis_results/effectiveFF/kp120900kp120900/data/"
    )
    datafile_ratios_g4_quark1 = FH_datadir / Path(
        "gamma4/eff_3pt_formfactors_quark1.pkl"
    )
    with open(datafile_ratios_g4_quark1, "rb") as file_in:
        ratios_g4_quark1 = pickle.load(file_in)
    datafile_ratios_g4_quark2 = FH_datadir / Path(
        "gamma4/eff_3pt_formfactors_quark2.pkl"
    )
    with open(datafile_ratios_g4_quark2, "rb") as file_in:
        ratios_g4_quark2 = pickle.load(file_in)
    datafile_ratios_g2_quark1 = FH_datadir / Path(
        "gamma2/eff_3pt_formfactors_quark1.pkl"
        # "gamma2/eff_3pt_ratios_quark1.pkl"
    )
    with open(datafile_ratios_g2_quark1, "rb") as file_in:
        ratios_g2_quark1 = pickle.load(file_in)
    datafile_ratios_g2_quark2 = FH_datadir / Path(
        "gamma2/eff_3pt_formfactors_quark2.pkl"
        # "gamma2/eff_3pt_ratios_quark2.pkl"
    )
    with open(datafile_ratios_g2_quark2, "rb") as file_in:
        ratios_g2_quark2 = pickle.load(file_in)

    # # ======================================================================
    # # Read fits
    # datafile_fits_quark1 = FH_datadir / Path("gamma4/3pt_formfactors_fit_quark1.pkl")
    # with open(datafile_fits_quark1, "rb") as file_in:
    #     fitlist_quark1 = pickle.load(file_in)
    # datafile_fits_quark2 = FH_datadir / Path("gamma4/3pt_formfactors_fit_quark2.pkl")
    # with open(datafile_fits_quark2, "rb") as file_in:
    #     fitlist_quark2 = pickle.load(file_in)

    # ======================================================================
    # Read fits from the completed FH analysis
    # For gamma4
    datafile_fits_quark1 = Path(
        "/home/mischa/Documents/PhD/analysis_code/feynhell_analysis/data/pickles/32x64/Feyn-Hell_b5p50kp120900kp120900/kp120900kp120900/formfactors/GEdbl.csv"
    )
    datafile_fits_quark2 = Path(
        "/home/mischa/Documents/PhD/analysis_code/feynhell_analysis/data/pickles/32x64/Feyn-Hell_b5p50kp120900kp120900/kp120900kp120900/formfactors/GEsgl.csv"
    )
    with open(datafile_fits_quark1) as csvfile:
        dataread = csv.reader(csvfile, delimiter=",", quotechar="|")
        rows = np.array([x for x in dataread])
        # qvals = np.array([float(i) for i in rows[0]])
        fitlist_g4_quark1 = np.array(
            [[float(value) for value in row] for row in rows[1:]]
        ).T
    with open(datafile_fits_quark2) as csvfile:
        dataread = csv.reader(csvfile, delimiter=",", quotechar="|")
        rows = np.array([x for x in dataread])
        # qvals = np.array([float(i) for i in rows[0]])
        fitlist_g4_quark2 = np.array(
            [[float(value) for value in row] for row in rows[1:]]
        ).T

    # For gamma2
    datafile_fits_quark1 = Path(
        "/home/mischa/Documents/PhD/analysis_code/feynhell_analysis/data/pickles/32x64/Feyn-Hell_b5p50kp120900kp120900/kp120900kp120900/formfactors/GMdbl.csv"
    )
    datafile_fits_quark2 = Path(
        "/home/mischa/Documents/PhD/analysis_code/feynhell_analysis/data/pickles/32x64/Feyn-Hell_b5p50kp120900kp120900/kp120900kp120900/formfactors/GMsgl.csv"
    )
    with open(datafile_fits_quark1) as csvfile:
        dataread = csv.reader(csvfile, delimiter=",", quotechar="|")
        rows = np.array([x for x in dataread])
        # qvals = np.array([float(i) for i in rows[0]])
        fitlist_g2_quark1 = np.array(
            [[float(value) for value in row] for row in rows[1:]]
        ).T
    with open(datafile_fits_quark2) as csvfile:
        dataread = csv.reader(csvfile, delimiter=",", quotechar="|")
        rows = np.array([x for x in dataread])
        # qvals = np.array([float(i) for i in rows[0]])
        fitlist_g2_quark2 = np.array(
            [[float(value) for value in row] for row in rows[1:]]
        ).T

    # ======================================================================
    # plot the results of the three-point fn ratio fits
    # GE
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom0_quark1,
        operators_chroma,
        operators_tex_chroma,
        ratios_g4_quark1[0],
        fitlist_g4_quark1[0],
        ylim=(2.25, 2.4),
    )
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom1_quark1,
        operators_chroma,
        operators_tex_chroma,
        ratios_g4_quark1[1],
        fitlist_g4_quark1[1],
        ylim=(0, 1.5),
    )
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom2_quark1,
        operators_chroma,
        operators_tex_chroma,
        ratios_g4_quark1[3],
        fitlist_g4_quark1[3],
        ylim=(0, 0.8),
    )
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom3_quark1,
        operators_chroma,
        operators_tex_chroma,
        ratios_g4_quark1[5],
        fitlist_g4_quark1[5],
        ylim=(-0.15, 0.5),
    )

    # Quark 2
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom0_quark2,
        operators_chroma,
        operators_tex_chroma,
        ratios_g4_quark2[0],
        fitlist_g4_quark2[0],
        ylim=(1.125, 1.2),
    )
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom1_quark2,
        operators_chroma,
        operators_tex_chroma,
        ratios_g4_quark2[1],
        fitlist_g4_quark2[1],
        ylim=(0, 0.8),
    )
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom2_quark2,
        operators_chroma,
        operators_tex_chroma,
        ratios_g4_quark2[3],
        fitlist_g4_quark2[3],
        ylim=(0, 0.5),
    )
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom3_quark2,
        operators_chroma,
        operators_tex_chroma,
        ratios_g4_quark2[5],
        fitlist_g4_quark2[5],
        ylim=(-0.15, 0.5),
    )

    # ======================================================================
    # plot the results of the three-point fn ratio fits
    # GM
    # Quark 1
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom1_quark1_GM,
        operators_chroma,
        operators_tex_chroma,
        ratios_g2_quark1[0],
        fitlist_g2_quark1[0],
        ylim=(0, 2.5),
    )
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom2_quark1_GM,
        operators_chroma,
        operators_tex_chroma,
        ratios_g2_quark1[2],
        fitlist_g2_quark1[2],
        ylim=(-0.1, 1.5),
    )
    # Quark 2
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom1_quark2_GM,
        operators_chroma,
        operators_tex_chroma,
        ratios_g2_quark2[0],
        fitlist_g2_quark2[0],
        ylim=(-0.7, 0.1),
    )
    plots.plot_3point_nonzeromom_comp(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_mom2_quark2_GM,
        operators_chroma,
        operators_tex_chroma,
        ratios_g2_quark2[2],
        fitlist_g2_quark2[2],
        ylim=(-0.5, 0.2),
    )

    return


if __name__ == "__main__":
    main()
