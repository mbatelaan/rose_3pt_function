import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
from ratiofitting_emff import select_2pt_fit
from ratiofitting_emff import FormatMom
from analysis import stats
from analysis import fitfunc


_metadata = {"Author": "Mischa Batelaan", "Creator": __file__}

_colors = [
    (0, 0, 0),
    (0.9, 0.6, 0),
    (0.35, 0.7, 0.9),
    (0, 0.6, 0.5),
    (0.95, 0.9, 0.25),
    (0, 0.45, 0.7),
    (0.8, 0.4, 0),
    (0.8, 0.6, 0.7),
]

_fmts = ["s", "^", "o", "p", ".", "v", "P", ",", "*"]


def plot_ratios_plateau_fit(
    ratios, plateaus, redchisqs, src_snk_times, delta_t, plotdir, plotparam, title=""
):
    time = np.arange(64)
    labels = [
        r"$t_{\mathrm{sep}}=10$",
        r"$t_{\mathrm{sep}}=13$",
        r"$t_{\mathrm{sep}}=16$",
    ]

    f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 5))
    # f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 4))
    for icorr, corr in enumerate(ratios):
        plot_time = time[: src_snk_times[icorr]] - src_snk_times[icorr] / 2
        plot_time2 = time - src_snk_times[icorr] / 2
        plot_time_fit = (
            np.arange(src_snk_times[icorr] + 1)[delta_t[icorr] : -delta_t[icorr]]
            - src_snk_times[icorr] / 2
        )

        ydata = np.average(corr, axis=0)
        yerror = np.std(corr, axis=0)

        axarr[icorr].errorbar(
            plot_time2[1 : src_snk_times[icorr]],
            ydata[1 : src_snk_times[icorr]],
            yerror[1 : src_snk_times[icorr]],
            capsize=4,
            elinewidth=1,
            color=_colors[icorr],
            fmt=_fmts[icorr],
            # markerfacecolor="none",
            # label=labels[icorr],
        )

        axarr[icorr].fill_between(
            plot_time_fit,
            [np.average(plateaus[icorr]) - np.std(plateaus[icorr])]
            * len(plot_time_fit),
            [np.average(plateaus[icorr]) + np.std(plateaus[icorr])]
            * len(plot_time_fit),
            alpha=0.3,
            linewidth=0,
            color=_colors[3],
            label=f"$\chi^2_{{\mathrm{{dof}}}}={redchisqs[icorr]:.2f}$",
        )

        # axarr[icorr].grid(True)
        axarr[icorr].legend(fontsize=15, loc="upper left")
        axarr[icorr].set_title(labels[icorr])
        axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
        # axarr[icorr].set_ylabel(r"$R(t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18)
        axarr[icorr].set_ylabel(
            rf"$R(t_{{\mathrm{{sep}}}}, \tau; {plotparam[1]}; \Gamma_{{\textrm{{{plotparam[2].lower()}}} }})$",
            labelpad=5,
            fontsize=18,
        )
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(-0.5, src_snk_times[icorr] + 0.5)
        axarr[icorr].set_xlim(
            plot_time2[0] - 0.5, plot_time2[src_snk_times[icorr]] + 0.5
        )
        # axarr[icorr].set_ylim(1.104, 1.181)

    savefile = plotdir / Path(f"{title}.pdf")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # savefile2 = plotdir / Path(f"{title}.png")
    # plt.savefig(savefile2, dpi=50)
    plt.close()
    return


def plot_ratio_fit_paper(
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

    # f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 5))
    f, axarr = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(7, 4))
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

        # axarr[icorr].grid(True, alpha=0.4)
        # axarr[icorr].legend(fontsize=15, loc="upper left")
        axarr[icorr].set_title(labels[icorr])

        axarr[icorr].set_xlabel(r"$\tau-t_{\mathrm{sep}}/2$", labelpad=14, fontsize=18)
        axarr[icorr].set_ylabel(r"$R(t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18)
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(-src_snk_times[-1] - 1, src_snk_times[-1] + 1)
        axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)
        # axarr[icorr].set_ylim(1.104, 1.181)

    savefile = plotdir / Path(f"{title}.pdf")
    # savefile_ylim = plotdir / Path(f"{title}_ylim.pdf")
    # savefile2 = plotdir / Path(f"{title}.png")
    # savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.savefig(savefile2, dpi=500)
    # plt.savefig(savefile3, dpi=100)
    # plt.ylim(1.104, 1.181)
    # plt.savefig(savefile_ylim)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit_together(
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

    f, axarr = plt.subplots(figsize=(7, 5))
    f.subplots_adjust(wspace=0, bottom=0.15)

    # Plot the 3pt fn ratios
    ydata0 = np.average(ratios[0], axis=0)
    yerror0 = np.std(ratios[0], axis=0)
    plot_time0 = time - (src_snk_times[0]) / 2
    axarr.errorbar(
        plot_time0[1 : src_snk_times[0]],
        ydata0[1 : src_snk_times[0]],
        yerror0[1 : src_snk_times[0]],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
        label=labels[0],
        # markerfacecolor="none",
    )
    ydata1 = np.average(ratios[1], axis=0)
    yerror1 = np.std(ratios[1], axis=0)
    plot_time1 = time - (src_snk_times[1]) / 2
    axarr.errorbar(
        plot_time1[1 : src_snk_times[1]],
        ydata1[1 : src_snk_times[1]],
        yerror1[1 : src_snk_times[1]],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        label=labels[1],
        # markerfacecolor="none",
    )
    ydata2 = np.average(ratios[2], axis=0)
    yerror2 = np.std(ratios[2], axis=0)
    plot_time2 = time - (src_snk_times[2]) / 2
    axarr.errorbar(
        plot_time2[1 : src_snk_times[2]],
        ydata2[1 : src_snk_times[2]],
        yerror2[1 : src_snk_times[2]],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
        label=labels[2],
        # markerfacecolor="none",
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

    # plot the two-exp fit results
    plot_x_values0 = (
        np.arange(src_snk_times[0] + 1)[delta_t:-delta_t] - (src_snk_times[0]) / 2
    )
    axarr.plot(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        color=_colors[0],
    )

    plot_x_values1 = (
        np.arange(src_snk_times[1] + 1)[delta_t:-delta_t] - (src_snk_times[1]) / 2
    )
    axarr.plot(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        color=_colors[1],
    )
    plot_x_values2 = (
        np.arange(src_snk_times[2] + 1)[delta_t:-delta_t] - (src_snk_times[2]) / 2
    )
    axarr.plot(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        color=_colors[2],
    )

    # Plot the two-exp fits to the ratio
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

    # Plot the parameter value from the fit
    axarr.axhline(np.average(fit_param_boot[:, 0]), color="k", alpha=0.4)
    plot_time3 = np.array([-20, 20])
    axarr.fill_between(
        plot_time3,
        [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
        * len(plot_time3),
        [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
        * len(plot_time3),
        alpha=0.3,
        linewidth=0,
        color=_colors[6],
        label=r"$\textrm{2-exp fit}$",
    )

    axarr.set_xlim(-8, 8)
    axarr.set_xlabel(r"$\tau - t_{\textrm{sep}}/2$", labelpad=14, fontsize=18)
    # axarr.set_ylabel(r"$R(t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18)
    axarr.set_ylabel(
        rf"$R(t_{{\mathrm{{sep}}}}, \tau; {plotparam[1]}; \Gamma_{{\textrm{{{plotparam[2].lower()}}} }})$",
        labelpad=5,
        fontsize=18,
    )
    axarr.label_outer()
    axarr.legend(fontsize="x-small")

    savefile = plotdir / Path(f"{title}.pdf")
    # savefile2 = plotdir / Path(f"{title}.png")
    # savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit_together_comp(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_ratio,
    FH_fit,
    title="",
    ylim=(0, 2),
):
    time = np.arange(64)
    labels = [
        r"$t=10$",
        r"$t=13$",
        r"$t=16$",
    ]

    f, axarr = plt.subplots(figsize=(7, 5))
    f.subplots_adjust(wspace=0, bottom=0.15)

    # Plot the 3pt fn ratios
    ydata0 = np.average(ratios[0], axis=0)
    yerror0 = np.std(ratios[0], axis=0)
    plot_time0 = time  # - (src_snk_times[0]) / 2
    axarr.errorbar(
        plot_time0[1 : src_snk_times[0]],
        ydata0[1 : src_snk_times[0]],
        yerror0[1 : src_snk_times[0]],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
        label=labels[0],
        # markerfacecolor="none",
    )
    ydata1 = np.average(ratios[1], axis=0)
    yerror1 = np.std(ratios[1], axis=0)
    plot_time1 = time  # - (src_snk_times[1]) / 2
    axarr.errorbar(
        plot_time1[1 : src_snk_times[1]],
        ydata1[1 : src_snk_times[1]],
        yerror1[1 : src_snk_times[1]],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        label=labels[1],
        # markerfacecolor="none",
    )
    ydata2 = np.average(ratios[2], axis=0)
    yerror2 = np.std(ratios[2], axis=0)
    plot_time2 = time  # - (src_snk_times[2]) / 2
    axarr.errorbar(
        plot_time2[1 : src_snk_times[2]],
        ydata2[1 : src_snk_times[2]],
        yerror2[1 : src_snk_times[2]],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
        label=labels[2],
        # markerfacecolor="none",
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

    # plot the two-exp fit results
    plot_x_values0 = np.arange(src_snk_times[0] + 1)[
        delta_t:-delta_t
    ]  # - (src_snk_times[0]) / 2
    axarr.plot(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        color=_colors[0],
    )

    plot_x_values1 = np.arange(src_snk_times[1] + 1)[
        delta_t:-delta_t
    ]  # - (src_snk_times[1]) / 2
    axarr.plot(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        color=_colors[1],
    )
    plot_x_values2 = np.arange(src_snk_times[2] + 1)[
        delta_t:-delta_t
    ]  # - (src_snk_times[2]) / 2
    axarr.plot(
        plot_x_values2,
        np.average(ratio_fit[:, step_indices[2] : step_indices[3]], axis=0),
        color=_colors[2],
    )

    # Plot the two-exp fits to the ratio
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

    # Plot the parameter value from the fit
    axarr.axhline(np.average(fit_param_boot[:, 0]), color="k", alpha=0.4)
    plot_time3 = np.array([-20, 60])
    axarr.fill_between(
        plot_time3,
        [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
        * len(plot_time3),
        [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
        * len(plot_time3),
        alpha=0.3,
        linewidth=0,
        color=_colors[6],
        label=r"$\textrm{2-exp fit}$",
    )

    # ======================================================================
    # Plot the FH results
    yavgeff = np.average(FH_ratio, axis=1)
    yerreff = np.std(FH_ratio, axis=1)
    dE_avg = np.average(FH_fit)
    dE_std = np.std(FH_fit)
    time = np.arange(0, len(yavgeff))
    efftime = time
    plt.errorbar(
        efftime,
        yavgeff,
        yerreff,
        capsize=2,
        elinewidth=1,
        color=_colors[5],
        marker=_fmts[3],
        linewidth=0,
        markerfacecolor="none",
        label=r"$\textrm{FH ratio}$",
    )
    tdatadE = np.arange(8, 15)
    plt.plot(
        tdatadE,
        [dE_avg] * len(tdatadE),
        color=_colors[5],
    )
    plt.fill_between(
        tdatadE,
        [dE_avg - dE_std] * len(tdatadE),
        [dE_avg + dE_std] * len(tdatadE),
        color=_colors[5],
        # label=r"$D_1$",
        alpha=0.3,
        linewidth=0,
        label=r"$\textrm{FH ratio fit}$",
    )

    # ======================================================================

    # axarr.set_xlim(-8, 8)
    axarr.set_xlim(0, 21)
    axarr.set_ylim(ylim)
    axarr.set_xlabel(r"$\tau - t/2$", labelpad=14, fontsize=18)
    # axarr.set_ylabel(r"$R(t, \tau)$", labelpad=5, fontsize=18)
    axarr.set_ylabel(
        # rf"$R(t, \tau; {plotparam[1]}; \Gamma_{{\textrm{{{plotparam[2].lower()}}} }})$",
        r"$G_{E,p,\textrm{Eff.}}^u(Q^2)$",
        labelpad=5,
        fontsize=18,
    )
    axarr.label_outer()
    axarr.legend(fontsize="x-small")

    savefile = plotdir / Path(f"{title}_comparison.pdf")
    print(savefile)
    # savefile2 = plotdir / Path(f"{title}.png")
    # savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()
    return


def plot_ratio_fit_together_comp_two(
    ratios,
    ratio_fit,
    delta_t,
    src_snk_times,
    redchisq,
    fit_param_boot,
    plotdir,
    plotparam,
    FH_ratio,
    FH_fit,
    quark_flavour,
    FF_name="E",
    title="",
    ylim=(0, 2),
):
    time = np.arange(64)
    labels = [
        r"$t=10$",
        r"$t=13$",
        r"$t=16$",
    ]

    # f, axarr = plt.subplots(figsize=(7, 5))
    f, axarr = plt.subplots(
        1, 2, figsize=(7, 5), sharey=False, gridspec_kw={"width_ratios": [3, 1]}
    )
    f.subplots_adjust(wspace=0, bottom=0.15, left=0.15)

    # Plot the 3pt fn ratios
    ydata0 = np.average(ratios[0], axis=0)
    yerror0 = np.std(ratios[0], axis=0)
    plot_time0 = time  # - (src_snk_times[0]) / 2
    axarr[0].errorbar(
        plot_time0[1 : src_snk_times[0]],
        ydata0[1 : src_snk_times[0]],
        yerror0[1 : src_snk_times[0]],
        capsize=4,
        elinewidth=1,
        color=_colors[0],
        fmt=_fmts[0],
        label=labels[0],
        # markerfacecolor="none",
    )
    ydata1 = np.average(ratios[1], axis=0)
    yerror1 = np.std(ratios[1], axis=0)
    plot_time1 = time  # - (src_snk_times[1]) / 2
    axarr[0].errorbar(
        plot_time1[1 : src_snk_times[1]],
        ydata1[1 : src_snk_times[1]],
        yerror1[1 : src_snk_times[1]],
        capsize=4,
        elinewidth=1,
        color=_colors[1],
        fmt=_fmts[1],
        label=labels[1],
        # markerfacecolor="none",
    )
    ydata2 = np.average(ratios[2], axis=0)
    yerror2 = np.std(ratios[2], axis=0)
    plot_time2 = time  # - (src_snk_times[2]) / 2
    axarr[0].errorbar(
        plot_time2[1 : src_snk_times[2]],
        ydata2[1 : src_snk_times[2]],
        yerror2[1 : src_snk_times[2]],
        capsize=4,
        elinewidth=1,
        color=_colors[2],
        fmt=_fmts[2],
        label=labels[2],
        # markerfacecolor="none",
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

    # plot the two-exp fit results
    plot_x_values0 = np.arange(src_snk_times[0] + 1)[
        delta_t:-delta_t
    ]  # - (src_snk_times[0]) / 2
    axarr[0].plot(
        plot_x_values0,
        np.average(ratio_fit[:, step_indices[0] : step_indices[1]], axis=0),
        color=_colors[0],
    )

    plot_x_values1 = np.arange(src_snk_times[1] + 1)[
        delta_t:-delta_t
    ]  # - (src_snk_times[1]) / 2
    axarr[0].plot(
        plot_x_values1,
        np.average(ratio_fit[:, step_indices[1] : step_indices[2]], axis=0),
        color=_colors[1],
    )
    plot_x_values2 = np.arange(src_snk_times[2] + 1)[
        delta_t:-delta_t
    ]  # - (src_snk_times[2]) / 2
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

    # # Plot the parameter value from the fit
    # axarr[0].axhline(np.average(fit_param_boot[:, 0]), color="k", alpha=0.4)
    # plot_time3 = np.array([-20, 60])
    # axarr[0].fill_between(
    #     plot_time3,
    #     [np.average(fit_param_boot[:, 0]) - np.std(fit_param_boot[:, 0])]
    #     * len(plot_time3),
    #     [np.average(fit_param_boot[:, 0]) + np.std(fit_param_boot[:, 0])]
    #     * len(plot_time3),
    #     alpha=0.3,
    #     linewidth=0,
    #     color=_colors[6],
    #     label=r"$\textrm{2-exp fit}$",
    # )

    # ======================================================================
    # Plot the FH results
    yavgeff = np.average(FH_ratio, axis=1)
    yerreff = np.std(FH_ratio, axis=1)
    dE_avg = np.average(FH_fit)
    dE_std = np.std(FH_fit)
    time = np.arange(0, len(yavgeff))
    efftime = time
    axarr[0].errorbar(
        efftime + 0.15,
        yavgeff,
        yerreff,
        capsize=2,
        elinewidth=1,
        color=_colors[5],
        marker=_fmts[3],
        linewidth=0,
        markerfacecolor="none",
        label=r"$\textrm{FH ratio}$",
    )
    tdatadE = np.arange(8, 18)
    axarr[0].plot(
        tdatadE,
        [dE_avg] * len(tdatadE),
        color=_colors[5],
    )
    axarr[0].fill_between(
        tdatadE,
        [dE_avg - dE_std] * len(tdatadE),
        [dE_avg + dE_std] * len(tdatadE),
        color=_colors[5],
        # label=r"$D_1$",
        alpha=0.3,
        linewidth=0,
        # label=r"$\textrm{FH ratio fit}$",
    )

    # ======================================================================
    # Plot the fit results on the second subplot
    # Fit a constant to the ratios
    t10_ratio_data = ratios[0][:, delta_t:-delta_t]
    fitparam_t10 = stats.fit_bootstrap(
        fitfunc.constant,
        [1],
        np.arange(len(t10_ratio_data[0])),
        t10_ratio_data,
        bounds=None,
        time=False,
        fullcov=False,
    )
    t10_const_fit = fitparam_t10["param"]
    # t10_const_fit = np.average(t10_ratio_data, axis=1)

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
    plt.errorbar(
        4,
        dE_avg,
        dE_std,
        capsize=4,
        elinewidth=1,
        color=_colors[5],
        fmt=_fmts[3],
    )
    # axarr[1].errorbar(
    #     4,
    #     np.average(FH_data["deltaE_fit"]),
    #     np.std(FH_data["deltaE_fit"]),
    #     capsize=4,
    #     elinewidth=1,
    #     color=_colors[3],
    #     fmt=_fmts[3],
    # )
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
    axarr[1].set_ylim(ylim)

    # axarr[0].set_xlim(-8, 8)
    axarr[0].set_xlim(0, 19)
    axarr[0].set_ylim(ylim)
    # axarr[0].set_xlabel(r"$\tau - t/2$", labelpad=14, fontsize=18)
    axarr[0].set_xlabel(r"$\tau,t$", labelpad=14, fontsize=18)
    # axarr[0].set_ylabel(r"$R(t, \tau)$", labelpad=5, fontsize=18)
    axarr[0].set_ylabel(
        # rf"$R(t, \tau; {plotparam[1]}; \Gamma_{{\textrm{{{plotparam[2].lower()}}} }})$",
        # r"$G_{E,p,\textrm{Eff.}}^u(Q^2)$",
        rf"$G_{{{FF_name},p}}^{quark_flavour.lower()}(Q^2)$",
        labelpad=5,
        fontsize=18,
    )
    axarr[0].label_outer()
    axarr[0].legend(fontsize="x-small")

    savefile = plotdir / Path(f"{title}_comparison_two.pdf")
    print(savefile)
    # savefile2 = plotdir / Path(f"{title}.png")
    # savefile3 = plotdir / Path(f"{title}_small.png")
    savefile.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(savefile)
    # plt.show()
    plt.close()
    return


def plot_3point_zeromom(
    latticedir,
    resultsdir,
    plotdir,
    datadir,
    corr_choices,
    operators_chroma,
    operators_tex_chroma,
):
    """Read the ratio for the parameters listed in the dict corr_choices,
    Read the fit for this ratio with a plateau and 2-exp function.
    Plot the fit results.
    """

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    config_num = 999
    sink_mom = "p0_0_0"

    # Loop over the parameter choices
    for ichoice, corrdata in enumerate(corr_choices):
        mom = corrdata["momentum"]
        operator = operators_chroma[corrdata["chroma_index"]]
        flav = corrdata["quark_flavour"]
        pol = corrdata["pol"]
        reim = corrdata["reim"]
        ir = np.where(np.array(["real", "imag"]) == reim)[0][0]
        delta_t = corrdata["delta_t"]
        delta_t_plateau = corrdata["delta_t_plateau"]

        # Read in the ratio data
        datafile_ratio = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_3pt_ratios_t10_t13_t16.pkl"
        )
        with open(datafile_ratio, "rb") as file_in:
            ratio_list_reim = pickle.load(file_in)

        # Read the fit results from pickle files
        # Ratio fit
        datafile_ratio_plateau = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_{reim}_3pt_ratio_plateau_fit.pkl"
        )
        with open(datafile_ratio_plateau, "rb") as file_in:
            plateau_list = pickle.load(file_in)
        plateau_param_list = [fit["param"] for fit in plateau_list]
        plateau_redchisq_list = [fit["redchisq"] for fit in plateau_list]

        # Two-exp fit
        datafile_ratio_2exp = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_{reim}_3pt_ratio_2exp_fit.pkl"
        )
        with open(datafile_ratio_2exp, "rb") as file_in:
            fit_params_ratio = pickle.load(file_in)
        fit_param_ratio_boot = fit_params_ratio["fit_param_boot"]
        ratio_fit_boot = fit_params_ratio["fitted_ratio_boot"]
        redchisq_ratio = fit_params_ratio["red_chisq_fit"]
        delta_t = fit_params_ratio["delta_t"]

        # ======================================================================
        # Plot the results with plateau fits
        plot_ratios_plateau_fit(
            ratio_list_reim[ir],
            plateau_param_list,
            plateau_redchisq_list,
            src_snk_times,
            delta_t_plateau,
            plotdir,
            [mom, operators_tex_chroma[corrdata["chroma_index"]], pol, reim],
            title=f"{mom}/{pol}/ratio_plateau_fit_{reim}_{operator}_{flav}",
        )
        # ======================================================================
        # Plot the results of two-exp fit to the ratio
        plot_ratio_fit_paper(
            ratio_list_reim[ir],
            ratio_fit_boot,
            delta_t,
            src_snk_times,
            redchisq_ratio,
            fit_param_ratio_boot,
            plotdir,
            [mom, operators_tex_chroma[corrdata["chroma_index"]], pol, reim],
            title=f"{mom}/{pol}/ratio_2expfit_{reim}_{operator}_{flav}_{mom}_paper",
        )
        plot_ratio_fit_together(
            ratio_list_reim[ir],
            ratio_fit_boot,
            delta_t,
            src_snk_times,
            redchisq_ratio,
            fit_param_ratio_boot,
            plotdir,
            [mom, operators_tex_chroma[corrdata["chroma_index"]], pol, reim],
            title=f"{mom}/{pol}/ratio_2expfit_{reim}_{operator}_{flav}_{mom}_together",
        )
    return


def plot_3point_nonzeromom(
    latticedir,
    resultsdir,
    plotdir,
    datadir,
    corr_choices,
    operators_chroma,
    operators_tex_chroma,
):
    """Read the ratio for the parameters listed in the dict corr_choices,
    Read the fit for this ratio with a plateau and 2-exp function.
    Plot the fit results.
    """

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    config_num = 999
    # sink_mom = "p0_0_0"

    # Loop over the parameter choices
    for ichoice, corrdata in enumerate(corr_choices):
        print("\n", ichoice)
        # mom = corrdata["momentum"]
        # sink_mom = corrdata["sink_mom"]
        # source_mom = corrdata["source_mom"]

        snk_mom = corrdata["snk_mom"]
        mom_ = corrdata["mom"]
        src_mom = snk_mom - mom_
        source_mom = FormatMom(src_mom)
        sink_mom = FormatMom(snk_mom)
        mom = FormatMom(mom_)

        # Because the picklelime code mislabels the sign on the source momenta
        neg_src_mom = -src_mom
        neg_snk_mom = -snk_mom
        neg_source_mom = FormatMom(neg_src_mom)
        neg_sink_mom = FormatMom(neg_snk_mom)

        operator = operators_chroma[corrdata["chroma_index"]]
        flav = corrdata["quark_flavour"]
        pol = corrdata["pol"]
        reim = corrdata["reim"]
        ir = np.where(np.array(["real", "imag"]) == reim)[0][0]
        delta_t = corrdata["delta_t"]
        delta_t_plateau = corrdata["delta_t_plateau"]
        print(f"{mom=}, {operator=}, {flav=}, {pol=}")

        # Read in the ratio data
        datafile_ratio = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_3pt_ratios_t10_t13_t16.pkl"
        )
        with open(datafile_ratio, "rb") as file_in:
            ratio_list_reim = pickle.load(file_in)

        # Read the fit results from pickle files
        # Ratio fit
        datafile_ratio_plateau = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_{reim}_3pt_ratio_plateau_fit.pkl"
        )
        with open(datafile_ratio_plateau, "rb") as file_in:
            plateau_list = pickle.load(file_in)
        plateau_param_list = [fit["param"] for fit in plateau_list]
        plateau_redchisq_list = [fit["redchisq"] for fit in plateau_list]

        # Two-exp fit
        datafile_ratio_2exp = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_{reim}_3pt_ratio_2exp_fit.pkl"
        )
        with open(datafile_ratio_2exp, "rb") as file_in:
            fit_params_ratio = pickle.load(file_in)
        fit_param_ratio_boot = fit_params_ratio["fit_param_boot"]
        ratio_fit_boot = fit_params_ratio["fitted_ratio_boot"]
        redchisq_ratio = fit_params_ratio["red_chisq_fit"]
        delta_t = fit_params_ratio["delta_t"]
        print("red. chisq = ", fit_params_ratio["red_chisq_fit"])

        # ======================================================================
        # Plot the results with plateau fits
        plot_ratios_plateau_fit(
            ratio_list_reim[ir],
            plateau_param_list,
            plateau_redchisq_list,
            src_snk_times,
            delta_t_plateau,
            plotdir,
            [mom, operators_tex_chroma[corrdata["chroma_index"]], pol, reim],
            title=f"{mom}/{pol}/ratio_plateau_fit_{reim}_{operator}_{flav}",
        )
        # ======================================================================
        # Plot the results of two-exp fit to the ratio
        plot_ratio_fit_paper(
            ratio_list_reim[ir],
            ratio_fit_boot,
            delta_t,
            src_snk_times,
            redchisq_ratio,
            fit_param_ratio_boot,
            plotdir,
            [mom, operators_tex_chroma[corrdata["chroma_index"]], pol, reim],
            title=f"{mom}/{pol}/ratio_2expfit_{reim}_{operator}_{flav}_{mom}_paper",
        )
        plot_ratio_fit_together(
            ratio_list_reim[ir],
            ratio_fit_boot,
            delta_t,
            src_snk_times,
            redchisq_ratio,
            fit_param_ratio_boot,
            plotdir,
            [mom, operators_tex_chroma[corrdata["chroma_index"]], pol, reim],
            title=f"{mom}/{pol}/ratio_2expfit_{reim}_{operator}_{flav}_{mom}_together",
        )
    return


def plot_3point_nonzeromom_comp(
    latticedir,
    resultsdir,
    plotdir,
    datadir,
    corr_choices,
    operators_chroma,
    operators_tex_chroma,
    FH_ratio,
    FH_fit,
    ylim=(0, 2),
):
    """Read the ratio for the parameters listed in the dict corr_choices,
    Read the fit for this ratio with a plateau and 2-exp function.
    Plot the fit results.
    """

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    config_num = 999
    # Loop over the parameter choices
    for ichoice, corrdata in enumerate(corr_choices):
        # mom = corrdata["momentum"]
        # sink_mom = corrdata["sink_mom"]
        # source_mom = corrdata["source_mom"]

        snk_mom = corrdata["snk_mom"]
        mom_ = corrdata["mom"]
        src_mom = snk_mom - mom_
        source_mom = FormatMom(src_mom)
        sink_mom = FormatMom(snk_mom)
        mom = FormatMom(mom_)
        mom_index1 = float(mom[2])

        # Because the picklelime code mislabels the sign on the source momenta
        neg_src_mom = -src_mom
        neg_snk_mom = -snk_mom
        neg_source_mom = FormatMom(neg_src_mom)
        neg_sink_mom = FormatMom(neg_snk_mom)

        operator = operators_chroma[corrdata["chroma_index"]]
        op_index = corrdata["chroma_index"]
        flav = corrdata["quark_flavour"]
        pol = corrdata["pol"]
        reim = corrdata["reim"]
        ir = np.where(np.array(["real", "imag"]) == reim)[0][0]
        delta_t = corrdata["delta_t"]
        delta_t_plateau = corrdata["delta_t_plateau"]
        tmin_choice = corrdata["tmin_choice"]
        # quark_flavour = corrdata["quark_flavour"]

        # ===============================================================
        # Read in the energies from 2pt. function fits
        kappa_combs = ["kp120900kp120900"]
        # source
        datafile_source = datadir / Path(
            f"{kappa_combs[0]}_{neg_source_mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_source, "rb") as file_in:
            fit_data_source = pickle.load(file_in)
        # Pick out the chosen 2pt fn fits
        best_fit_source, fit_params_source = select_2pt_fit(
            fit_data_source,
            tmin_choice,
            datadir,
        )
        # sink
        datafile_sink = datadir / Path(
            f"{kappa_combs[0]}_{neg_sink_mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_sink, "rb") as file_in:
            fit_data_sink = pickle.load(file_in)
        # Pick out the chosen 2pt fn fits
        best_fit_sink, fit_params_sink = select_2pt_fit(
            fit_data_sink,
            tmin_choice,
            datadir,
        )
        energy_3pt = 0.5 * (fit_params_source[:, 1] + fit_params_sink[:, 1])

        # ===============================================================
        # Read in the mass
        # source
        datafile_source = datadir / Path(
            f"{kappa_combs[0]}_p+0+0+0_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_source, "rb") as file_in:
            fit_data_source = pickle.load(file_in)
        # Pick out the chosen 2pt fn fits
        best_fit_source, fit_params_source = select_2pt_fit(
            fit_data_source,
            tmin_choice,
            datadir,
        )
        mass_3pt = fit_params_source[:, 1]

        # ===============================================================
        # Read in the ratio data
        datafile_ratio = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_3pt_ratios_t10_t13_t16.pkl"
        )
        with open(datafile_ratio, "rb") as file_in:
            ratio_list_reim = pickle.load(file_in)

        # ===============================================================
        # Read the fit results from pickle files
        # Ratio fit
        datafile_ratio_plateau = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_{reim}_3pt_ratio_plateau_fit.pkl"
        )
        with open(datafile_ratio_plateau, "rb") as file_in:
            plateau_list = pickle.load(file_in)
        plateau_param_list = [fit["param"] for fit in plateau_list]
        plateau_redchisq_list = [fit["redchisq"] for fit in plateau_list]

        # ===============================================================
        # Two-exp fit
        datafile_ratio_2exp = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_{reim}_3pt_ratio_2exp_fit.pkl"
        )
        with open(datafile_ratio_2exp, "rb") as file_in:
            fit_params_ratio = pickle.load(file_in)
        fit_param_ratio_boot = fit_params_ratio["fit_param_boot"]
        ratio_fit_boot = fit_params_ratio["fitted_ratio_boot"]
        redchisq_ratio = fit_params_ratio["red_chisq_fit"]
        delta_t = fit_params_ratio["delta_t"]

        # ======================================================================
        # Multiply the kinematic variables to get the effective form factor GE:
        print(f"{np.shape(mass_3pt)=}")
        print(f"{np.shape(energy_3pt)=}")
        # print(f"{np.shape(ratio_list_reim[ir])=}")
        print(f"{np.shape(ratio_fit_boot)=}")
        print(f"{np.shape(fit_param_ratio_boot)=}")
        # print(f"{np.average(mass_3pt*energy_3pt)=}")
        # kinematic_term = 1 / ( mass_3pt * energy_3pt)

        if op_index == 8:
            # GE
            kinematic_term = energy_3pt / mass_3pt
            FF_name = "E"
        elif op_index == 2:
            # GM
            # kinematic_term = energy_3pt / mass_3pt
            lattice_length = 32
            kinematic_term = 2 * energy_3pt / (mom_index1 * 2 * np.pi / lattice_length)
            FF_name = "M"
        print(f"{np.average(kinematic_term)=}")

        ratio_list_choice = [
            np.einsum("ij,i->ij", ratio_list, kinematic_term)
            for ratio_list in ratio_list_reim[ir]
        ]

        ratio_fit_boot = np.einsum("ij,i->ij", ratio_fit_boot, kinematic_term)
        fit_param_ratio_boot = np.einsum(
            "ij,i->ij", fit_param_ratio_boot, kinematic_term
        )
        # ======================================================================
        # Plot the results of two-exp fit to the ratio and the FH results
        # ======================================================================
        plot_ratio_fit_together_comp(
            ratio_list_choice,
            ratio_fit_boot,
            delta_t,
            src_snk_times,
            redchisq_ratio,
            fit_param_ratio_boot,
            plotdir,
            [mom, operators_tex_chroma[corrdata["chroma_index"]], pol, reim],
            FH_ratio,
            FH_fit,
            title=f"{mom}/{pol}/ratio_2expfit_{reim}_{operator}_{flav}_{mom}_together",
            ylim=ylim,
        )
        plot_ratio_fit_together_comp_two(
            ratio_list_choice,
            ratio_fit_boot,
            delta_t,
            src_snk_times,
            redchisq_ratio,
            fit_param_ratio_boot,
            plotdir,
            [mom, operators_tex_chroma[corrdata["chroma_index"]], pol, reim],
            FH_ratio,
            FH_fit,
            flav,
            FF_name=FF_name,
            title=f"{mom}/{pol}/ratio_2expfit_{reim}_{operator}_{flav}_{mom}_together",
            ylim=ylim,
        )
    return
