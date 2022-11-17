import numpy as np
from pathlib import Path

import pickle
import csv
import scipy.optimize as syopt
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

from plot_utils import save_plot
from formatting import err_brackets
from analysis.evxptreaders import evxptdata
from analysis.bootstrap import bootstrap
from analysis import stats
from analysis import fitfunc
import fit_functions as ff
import plot_functions as plots

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


def select_2pt_fit(fit_data_list, tmin_choice, datadir, mom, weight_tol=0.01):
    """Sort through the fits to the two-point function and pick out one fit result that has the chosen tmin and tmax"""
    # weight_tol = 0.01
    fitweights = np.array([fit["weight"] for fit in fit_data_list])
    best_fit_index = np.argmin(fitweights)
    fitweights = np.where(fitweights > weight_tol, fitweights, 0)
    fitweights = fitweights / sum(fitweights)
    fitparams = np.array([fit["param"] for fit in fit_data_list])
    fit_times = [fit["x"] for fit in fit_data_list]
    # chosen_time = np.where([times[0] == tmin_choice for times in fit_times])[0][0]
    # best_fit = fit_data_list[chosen_time]
    best_fit_index = np.argmax(fitweights)
    best_fit = fit_data_list[best_fit_index]
    # weighted_fit = best_fit["param"]
    fit_params = best_fit["param"]

    return best_fit, fit_params


def fit_ratio_plateau(ratio, src_snk_time, delta_t):
    """Fit to the three-point function with a two-exponential function, which includes parameters from the two-point functions"""

    fit_data = ratio[:, delta_t : src_snk_time + 1 - delta_t]
    tau_values = np.arange(src_snk_time + 1)[delta_t:-delta_t]
    # print(f"{np.shape(fit_data)=}")
    # print(f"{fit_data=}")

    if len(fit_data[0]) == 1:
        fitparam = {
            "x": tau_values,
            "y": fit_data,
            "fitfunction": fitfunc.constant.__doc__,
            "paramavg": np.average(fit_data),
            "param": fit_data,
            "redchisq": 1,
            "chisq": 1,
            "dof": 1,
        }
    else:
        fitparam = stats.fit_bootstrap(
            fitfunc.constant,
            [1],
            tau_values,
            fit_data,
            bounds=None,
            time=False,
            fullcov=False,
        )

    return fitparam


def fit_ratio_2exp(
    ratio_list,
    twopt_fit_params,
    src_snk_times,
    delta_t,
    fitfnc_2exp,
):
    """Fit to the three-point function with a two-exponential function, which includes parameters from the two-point functions"""

    # Set the parameters from the twoptfn
    A_0 = twopt_fit_params[:, 0]
    A_1 = twopt_fit_params[:, 0] * twopt_fit_params[:, 2]
    M = twopt_fit_params[:, 1]
    DeltaM = np.exp(twopt_fit_params[:, 3])

    # Create the fit data
    fitdata = np.concatenate(
        (
            ratio_list[0][:, delta_t : src_snk_times[0] + 1 - delta_t],
            ratio_list[1][:, delta_t : src_snk_times[1] + 1 - delta_t],
            ratio_list[2][:, delta_t : src_snk_times[2] + 1 - delta_t],
        ),
        axis=1,
    )
    t_values = np.concatenate(
        (
            [src_snk_times[0]] * (src_snk_times[0] + 1 - 2 * delta_t),
            [src_snk_times[1]] * (src_snk_times[1] + 1 - 2 * delta_t),
            [src_snk_times[2]] * (src_snk_times[2] + 1 - 2 * delta_t),
        )
    )
    tau_values = np.concatenate(
        (
            np.arange(src_snk_times[0] + 1)[delta_t:-delta_t],
            np.arange(src_snk_times[1] + 1)[delta_t:-delta_t],
            np.arange(src_snk_times[2] + 1)[delta_t:-delta_t],
        )
    )

    # Fit to the average of the data
    x_avg = [
        tau_values,
        t_values,
        np.average(A_0),
        np.average(A_1),
        np.average(M),
        np.average(DeltaM),
    ]
    p0 = [1, 1, 1]
    fitdata_avg = np.average(fitdata, axis=0)
    fitdata_std = np.std(fitdata, axis=0)
    cvinv = np.linalg.inv(np.cov(fitdata.T))
    var_inv = np.diag(1 / (fitdata_std**2))
    resavg = syopt.minimize(
        fitfunc.chisqfn,
        p0,
        args=(fitfnc_2exp, x_avg, fitdata_avg, var_inv),
        method="Nelder-Mead",
        options={"disp": False},
    )

    fit_param_avg = resavg.x
    # ratio_fit_avg = fitfnc_2exp(x_avg, fit_param_avg)
    chisq = fitfunc.chisqfn(resavg.x, fitfnc_2exp, x_avg, fitdata_avg, cvinv)
    redchisq = chisq / (len(fitdata_avg) - len(p0))
    print(f"{redchisq=}")
    print(f"{resavg.fun/(len(fitdata_avg) - len(p0))=}")

    # Fit to each bootstrap
    p0 = fit_param_avg
    nboot = np.shape(ratio_list[0])[0]
    fit_param_boot = []
    ratio_fit_boot = []
    for iboot in np.arange(nboot):
        x = [
            tau_values,
            t_values,
            A_0[iboot],
            A_1[iboot],
            M[iboot],
            DeltaM[iboot],
        ]
        res = syopt.minimize(
            fitfunc.chisqfn,
            p0,
            args=(fitfnc_2exp, x, fitdata[iboot], var_inv),
            method="Nelder-Mead",
            options={"disp": False},
        )
        fit_param_boot.append(res.x)
        ratio_fit_boot.append(fitfnc_2exp(x, res.x))
    ratio_fit_boot = np.array(ratio_fit_boot)
    fit_param_boot = np.array(fit_param_boot)

    chisq_ = fitfunc.chisqfn(
        np.average(fit_param_boot, axis=0), fitfnc_2exp, x_avg, fitdata_avg, cvinv
    )
    redchisq_ = chisq_ / (len(fitdata_avg) - len(p0))
    print(f"{redchisq_=}")

    return (
        fit_param_boot,
        ratio_fit_boot,
        fit_param_avg,
        redchisq,
    )


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
        axarr[icorr].set_ylabel(
            r"$R(\vec{p}\, ; t_{\mathrm{sep}}, \tau)$", labelpad=5, fontsize=18
        )
        axarr[icorr].label_outer()
        # axarr[icorr].set_xlim(-src_snk_times[-1] - 1, src_snk_times[-1] + 1)
        axarr[icorr].set_xlim(plot_time2[0] - 0.5, plot_time2[src_snk_times[-1]] + 0.5)
        # axarr[icorr].set_ylim(1.104, 1.181)

    # f.suptitle(
    #     rf"{plotparam[3]} 3-point function ratio with $\hat{{\mathcal{{O}}}}=${plotparam[1]}, $\Gamma = ${plotparam[2]}, $\vec{{q}}\, ={plotparam[0][1:]}$ with two-state fit $\chi^2_{{\mathrm{{dof}}}}={redchisq:.2f}$"
    # )
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


def make_double_ratio(
    threeptfn_n2sig,
    threeptfn_sig2n,
    twoptfn_sigma_real,
    twoptfn_neutron_real,
    src_snk_time,
):
    """Make the ratio of two-point and three-point functions which produces the plateau
    This ratio uses both the nucleon to sigma transition and the sigma to nucleon transition. Referenced in Flynn 2007 paper."""

    three_point_product = np.einsum(
        "ijk,ijk->ijk",
        threeptfn_n2sig[:, : src_snk_time + 1],
        threeptfn_sig2n[:, : src_snk_time + 1],
    )
    denominator = (
        twoptfn_sigma_real[:, src_snk_time] * twoptfn_neutron_real[:, src_snk_time]
    ) ** (-1)
    ratio = np.sqrt(np.einsum("ijk,i->ijk", three_point_product, denominator))
    return ratio


def fit_3point_zeromom(
    latticedir,
    resultsdir,
    plotdir,
    datadir,
    corr_choices,
    operators_chroma,
    operators_tex_chroma,
):
    """Read the ratio for the parameters listed in the dict corr_choices,
    fit to this ratio with a plateau and 2-exp function.
    Save the fit results.
    """

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    config_num = 999
    sink_mom = "p0_0_0"
    fitfnc_2exp = ff.threept_ratio_zeromom

    for ichoice, corrdata in enumerate(corr_choices):
        mom = corrdata["momentum"]
        operator = operators_chroma[corrdata["chroma_index"]]
        pol = corrdata["pol"]
        reim = corrdata["reim"]
        ir = np.where(np.array(["real", "imag"]) == reim)[0][0]
        delta_t = corrdata["delta_t"]
        delta_t_plateau = corrdata["delta_t_plateau"]
        tmin_choice = corrdata["tmin_choice"]

        # ======================================================================
        # Read the results of the fit to the two-point functions
        kappa_combs = ["kp120900kp120900"]
        datafile = datadir / Path(f"{kappa_combs[0]}_{mom}_{rel}_fitlist_2pt_2exp.pkl")
        with open(datafile, "rb") as file_in:
            fit_data = pickle.load(file_in)
        # Pick out the chosen 2pt fn fits
        best_fit, fit_params = select_2pt_fit(
            fit_data,
            tmin_choice,
            datadir,
            mom,
        )

        # ======================================================================
        # Read in the ratio data
        datafile_ratio = datadir / Path(
            f"{mom}_{operator}_{pol}_{rel}_3pt_ratios_t10_t13_16.pkl"
        )
        with open(datafile_ratio, "rb") as file_in:
            ratio_list_reim = pickle.load(file_in)

        # ======================================================================
        # fit to the ratio of 3pt and 2pt functions with a two-exponential function
        t10_plateau_fit = fit_ratio_plateau(
            ratio_list_reim[ir][0], src_snk_times[0], delta_t_plateau[0]
        )
        t13_plateau_fit = fit_ratio_plateau(
            ratio_list_reim[ir][1], src_snk_times[1], delta_t_plateau[1]
        )
        t16_plateau_fit = fit_ratio_plateau(
            ratio_list_reim[ir][2], src_snk_times[2], delta_t_plateau[2]
        )
        plateau_list = [
            t10_plateau_fit,
            t13_plateau_fit,
            t16_plateau_fit,
        ]
        plateau_param_list = [
            t10_plateau_fit["param"],
            t13_plateau_fit["param"],
            t16_plateau_fit["param"],
        ]
        redchisq_list = [
            t10_plateau_fit["redchisq"],
            t13_plateau_fit["redchisq"],
            t16_plateau_fit["redchisq"],
        ]

        fitfnc_2exp = ff.threept_ratio_zeromom
        (
            fit_param_ratio_boot,
            ratio_fit_boot,
            fit_param_ratio_avg,
            redchisq_ratio,
        ) = fit_ratio_2exp(
            ratio_list_reim[ir],
            fit_params,
            src_snk_times,
            delta_t,
            fitfnc_2exp,
        )
        fit_params_ratio = [
            fit_param_ratio_boot,
            ratio_fit_boot,
            fit_param_ratio_avg,
            redchisq_ratio,
            # best_fit,
            delta_t,
        ]

        # Save the fit results to pickle files
        datafile_ratio_plateau = datadir / Path(
            f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_plateau_fit.pkl"
        )
        with open(datafile_ratio_plateau, "wb") as file_out:
            pickle.dump(plateau_list, file_out)

        datafile_ratio_2exp = datadir / Path(
            f"{mom}_{operator}_{pol}_{rel}_{reim}_3pt_ratio_2exp_fit.pkl"
        )
        with open(datafile_ratio_2exp, "wb") as file_out:
            pickle.dump(fit_params_ratio, file_out)

    return


def save_ratios_zeromom(
    latticedir,
    resultsdir,
    datadir,
    corr_choices,
    operators_chroma,
):
    """Loop over the operators and momenta"""

    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    config_num = 999
    nboot = 500
    lattice_time_exent = 64
    sink_mom = "p0_0_0"

    for ichoice, corrdata in enumerate(corr_choices):
        mom = corrdata["momentum"]
        operator = operators_chroma[corrdata["chroma_index"]]
        pol = corrdata["pol"]
        reim = corrdata["reim"]
        ir = np.where(np.array(["real", "imag"]) == reim)[0][0]

        # ======================================================================
        # Read the two-point function data
        twoptfn_filename = latticedir / Path(
            f"twoptfn/barspec/32x64/unpreconditioned_slrc/kp120900kp120900/sh_gij_p21_75-sh_gij_p21_75/{mom}/barspec_nucleon_{rel}_{config_num}cfgs.pickle"
        )
        twoptfn = read_pickle(twoptfn_filename, nboot=500, nbin=1)
        twoptfn_real = twoptfn[:, :, 0]

        # Read in the 3pt function data
        threeptfn_pickle_t10 = latticedir / Path(
            f"bar3ptfn_t10_U/bar3ptfn/32x64/unpreconditioned_slrc/kp120900tkp120900_kp120900kp120900/NUCL_U_{pol}_NONREL_gI_t10_{sink_mom}/sh_gij_p21_75-sh_gij_p21_75/{mom}/bar3ptfn_{operator}_{config_num}cfgs.pickle"
        )
        threeptfn_pickle_t13 = latticedir / Path(
            f"bar3ptfn_t13_U/bar3ptfn/32x64/unpreconditioned_slrc/kp120900tkp120900_kp120900kp120900/NUCL_U_{pol}_NONREL_gI_t13_{sink_mom}/sh_gij_p21_75-sh_gij_p21_75/{mom}/bar3ptfn_{operator}_{config_num}cfgs.pickle"
        )
        threeptfn_pickle_t16 = latticedir / Path(
            f"bar3ptfn_t16_U/bar3ptfn/32x64/unpreconditioned_slrc/kp120900tkp120900_kp120900kp120900/NUCL_U_{pol}_NONREL_gI_t16_{sink_mom}/sh_gij_p21_75-sh_gij_p21_75/{mom}/bar3ptfn_{operator}_{config_num}cfgs.pickle"
        )
        threeptfn_t10 = read_pickle(threeptfn_pickle_t10, nboot=500, nbin=1)
        threeptfn_t13 = read_pickle(threeptfn_pickle_t13, nboot=500, nbin=1)
        threeptfn_t16 = read_pickle(threeptfn_pickle_t16, nboot=500, nbin=1)

        # ======================================================================
        # Construct the simple ratio of 3pt and 2pt functions
        ratio_t10 = np.einsum("ijk,i->ijk", threeptfn_t10, twoptfn_real[:, 10] ** (-1))
        ratio_t13 = np.einsum("ijk,i->ijk", threeptfn_t13, twoptfn_real[:, 13] ** (-1))
        ratio_t16 = np.einsum("ijk,i->ijk", threeptfn_t16, twoptfn_real[:, 16] ** (-1))
        ratio_list_reim = [
            [
                ratio_t10[:, :, 0],
                ratio_t13[:, :, 0],
                ratio_t16[:, :, 0],
            ],
            [
                ratio_t10[:, :, 1],
                ratio_t13[:, :, 1],
                ratio_t16[:, :, 1],
            ],
        ]

        datafile_ratio = datadir / Path(
            f"{mom}_{operator}_{pol}_{rel}_3pt_ratios_t10_t13_16.pkl"
        )
        with open(datafile_ratio, "wb") as file_out:
            pickle.dump(ratio_list_reim, file_out)
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
    fitfnc_2exp = ff.threept_ratio_zeromom

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
        plots.plot_ratios_plateau_fit(
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
        plots.plot_ratio_fit_paper(
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
        plots.plot_ratio_fit_together(
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


def get_Qsquared_values(datadir, tmin_choices_nucl, tmin_choices_sigm):
    """Loop over the momenta and read the two-point functions, then get the energies from them and use those to calculate the Q^2 values."""

    a = 0.074
    L = 32
    src_snk_times = np.array([10, 13, 16])
    rel = "nr"
    kappa_combs = ["kp121040kp121040", "kp121040kp120620"]
    momenta = ["p+0+0+0", "p+1+0+0", "p+1+1+0"]
    momenta_values = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

    # Load the fit values for the Q^2=0 correlators
    datafile_n = datadir / Path(
        f"{kappa_combs[0]}_{momenta[0]}_{rel}_fitlist_2pt_2exp.pkl"
    )
    with open(datafile_n, "rb") as file_in:
        fit_data_n_mom0 = pickle.load(file_in)
    datafile_s = datadir / Path(
        f"{kappa_combs[1]}_{momenta[0]}_{rel}_fitlist_2pt_2exp.pkl"
    )
    with open(datafile_s, "rb") as file_in:
        fit_data_s_mom0 = pickle.load(file_in)

    # Extract the chosen fit's energy from the data
    # Neutron
    fit_times_n = [fit["x"] for fit in fit_data_n_mom0]
    chosen_time_n = np.where(
        [times[0] == tmin_choices_nucl[0] for times in fit_times_n]
    )[0][0]
    best_fit_n_mom0 = fit_data_n_mom0[chosen_time_n]
    energy_n_mom0 = np.average(best_fit_n_mom0["param"][:, 1])
    # Sigma
    fit_times_s = [fit["x"] for fit in fit_data_s_mom0]
    chosen_time_s = np.where(
        [times[0] == tmin_choices_sigm[0] for times in fit_times_s]
    )[0][0]
    best_fit_s_mom0 = fit_data_s_mom0[chosen_time_s]
    energy_s_mom0 = np.average(best_fit_s_mom0["param"][:, 1])

    qsquared_sig2n_list = []
    qsquared_n2sig_list = []
    for imom, mom in enumerate(momenta):
        # Load the fit values for correlators at the given momentum
        datafile_n = datadir / Path(
            f"{kappa_combs[0]}_{mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_n, "rb") as file_in:
            fit_data_n = pickle.load(file_in)
        datafile_s = datadir / Path(
            f"{kappa_combs[1]}_{mom}_{rel}_fitlist_2pt_2exp.pkl"
        )
        with open(datafile_s, "rb") as file_in:
            fit_data_s = pickle.load(file_in)

        # Extract the chosen fit's energy from the data
        # Neutron
        fit_times_n = [fit["x"] for fit in fit_data_n]
        chosen_time_n = np.where(
            [times[0] == tmin_choices_nucl[imom] for times in fit_times_n]
        )[0][0]
        best_fit_n = fit_data_n[chosen_time_n]
        energy_n = np.average(best_fit_n["param"][:, 1])
        # Sigma
        fit_times_s = [fit["x"] for fit in fit_data_s]
        chosen_time_s = np.where(
            [times[0] == tmin_choices_sigm[imom] for times in fit_times_s]
        )[0][0]
        best_fit_s = fit_data_s[chosen_time_s]
        energy_s = np.average(best_fit_s["param"][:, 1])

        sig2n_Qsq = Q_squared_energies(
            energy_s, energy_n_mom0, momenta_values[imom], momenta_values[0], L, a
        )
        n2sig_Qsq = Q_squared_energies(
            energy_n, energy_s_mom0, momenta_values[imom], momenta_values[0], L, a
        )
        qsquared_sig2n_list.append(sig2n_Qsq)
        qsquared_n2sig_list.append(n2sig_Qsq)
        print(f"{sig2n_Qsq=}")
        print(f"{n2sig_Qsq=}")

    qsquared_sig2n_list = np.array(qsquared_sig2n_list)
    qsquared_n2sig_list = np.array(qsquared_n2sig_list)

    # Save the data to a file
    datafile_sig2n = datadir / Path(f"Qsquared_sig2n.pkl")
    datafile_n2sig = datadir / Path(f"Qsquared_n2sig.pkl")
    with open(datafile_sig2n, "wb") as file_out:
        pickle.dump(qsquared_sig2n_list, file_out)
    with open(datafile_n2sig, "wb") as file_out:
        pickle.dump(qsquared_n2sig_list, file_out)

    return qsquared_sig2n_list, qsquared_n2sig_list


def Q_squared_energies(E1, E2, n1, n2, L, a):
    """Returns Q^2 between two particles with momentum and twisted BC's
    n1, n2 are arrays which contain the fourier momenta for the first and second particle.
    L is the spatial lattice extent
    a is the lattice spacing
    """
    energydiff = np.sqrt(E2**2) - np.sqrt(E1**2)
    qvector_diff = ((2 * n2) - (2 * n1)) * (np.pi / L)
    Qsquared = (
        -1
        * (energydiff**2 - np.dot(qvector_diff, qvector_diff))
        * (0.1973**2)
        / (a**2)
    )
    return Qsquared


def main():
    plt.style.use("./mystyle.txt")
    plt.rc("text.latex", preamble=r"\usepackage{physics}")

    # --- directories ---
    latticedir = Path.home() / Path("Documents/PhD/lattice_results/rose_3pt_function/")
    resultsdir = Path.home() / Path("Dropbox/PhD/analysis_code/rose_3pt_function/")
    plotdir = resultsdir / Path("plots/")
    plotdir2 = plotdir / Path("twopoint/")
    datadir = resultsdir / Path("data/")

    plotdir.mkdir(parents=True, exist_ok=True)
    plotdir2.mkdir(parents=True, exist_ok=True)
    datadir.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # Fitting to the ratios
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
        "g0",
        "g1",
        "\gamma_1\gamma_2",
        "g2",
        "g02",
        "g12",
        "g53",
        "g3",
        "g03",
        "g13",
        "\gamma_3\gamma_5",
        "g23",
        "g51",
        "g05",
        "g5",
    ]
    corr_choices = [
        {
            "chroma_index": 0,
            "op_name": "Scalar",
            "pol": "UNPOL",
            "momentum": "p+0+0+0",
            "reim": "real",
            "delta_t": 4,
            "delta_t_plateau": [4, 6, 6],
            "tmin_choice": 4,
        },
        {
            "chroma_index": 3,
            "op_name": "Tensor",
            "pol": "POL",
            "momentum": "p+0+0+0",
            "reim": "imag",
            "delta_t": 4,
            "delta_t_plateau": [4, 5, 6],
            "tmin_choice": 4,
        },
        {
            "chroma_index": 11,
            "op_name": "Axial",
            "pol": "POL",
            "momentum": "p+0+0+0",
            "reim": "imag",
            "delta_t": 4,
            "delta_t_plateau": [4, 4, 4],
            "tmin_choice": 4,
        },
    ]

    # ======================================================================
    # Make the ratios
    save_ratios_zeromom(
        latticedir,
        resultsdir,
        datadir,
        corr_choices,
        operators_chroma,
    )

    # fit = True
    fit = False
    if fit:
        # ======================================================================
        # Construct a ratio with two 3pt functions and fit it for the zero momentum transfer case
        fit_3point_zeromom(
            latticedir,
            resultsdir,
            plotdir,
            datadir,
            corr_choices,
            operators_chroma,
            operators_tex_chroma,
        )

    # ======================================================================
    # plot the results of the three-point fn ratio fits
    plot_3point_zeromom(
        latticedir,
        resultsdir,
        plotdir,
        datadir,
        corr_choices,
        operators_chroma,
        operators_tex_chroma,
    )

    return


if __name__ == "__main__":
    main()
