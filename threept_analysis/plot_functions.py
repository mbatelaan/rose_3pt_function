import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

# from formatting import err_brackets

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

_fmts = ["s", "^", "o", ".", "p", "v", "P", ",", "*"]


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
    savefile2 = plotdir / Path(f"{title}.png")
    plt.savefig(savefile2, dpi=50)
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

    for ichoice, corrdata in enumerate(corr_choices):
        mom = corrdata["momentum"]
        operator = operators_chroma[corrdata["chroma_index"]]
        pol = corrdata["pol"]
        reim = corrdata["reim"]
        ir = np.where(np.array(["real", "imag"]) == reim)[0][0]
        delta_t = corrdata["delta_t"]
        delta_t_plateau = corrdata["delta_t_plateau"]
        print(f"{ir=}")

        # Read in the ratio data
        datafile_ratio = datadir / Path(
            f"{mom}_{operator}_{pol}_{rel}_3pt_ratios_t10_t13_16.pkl"
        )
        with open(datafile_ratio, "rb") as file_in:
            ratio_list_reim = pickle.load(file_in)

        # for ir, reim in enumerate(["real", "imag"]):

        # Read the fit results from pickle files
        datafile_ratio_plateau = datadir / Path(
            f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_plateau_fit.pkl"
        )
        with open(datafile_ratio_plateau, "rb") as file_in:
            plateau_list = pickle.load(file_in)
        plateau_param_list = [fit["param"] for fit in plateau_list]
        plateau_redchisq_list = [fit["redchisq"] for fit in plateau_list]

        datafile_ratio_2exp = datadir / Path(
            f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_2exp_fit.pkl"
        )
        with open(datafile_ratio_2exp, "rb") as file_in:
            fit_params_ratio = pickle.load(file_in)
        [
            fit_param_ratio_boot,
            ratio_fit_boot,
            fit_param_ratio_avg,
            redchisq_ratio,
            best_fit,
        ] = fit_params_ratio

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
            title=f"{mom}/{pol}/ratio_plateau_fit_{reim}_{operator}",
        )
        # ======================================================================
        # Plot the results of the fit to the ratio
        plot_ratio_fit_paper(
            ratio_list_reim[ir],
            ratio_fit_boot,
            delta_t,
            src_snk_times,
            redchisq_ratio,
            fit_param_ratio_boot,
            plotdir,
            [mom, operators_tex_chroma[corrdata["chroma_index"]], pol, reim],
            title=f"{mom}/{pol}/ratio_2expfit_{reim}_{operator}_{mom}_paper",
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
            title=f"{mom}/{pol}/ratio_2expfit_{reim}_{operator}_{mom}_together",
        )
    return
