import numpy as np
from pathlib import Path

import pickle
import csv
import scipy.optimize as syopt
import matplotlib.pyplot as plt

from formatting import err_brackets
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
    nboot = 500
    sink_mom = "p0_0_0"
    fitfnc_2exp = ff.threept_ratio_zeromom

    for ichoice, corrdata in enumerate(corr_choices):
        mom = corrdata["momentum"]
        operator = operators_chroma[corrdata["chroma_index"]]
        operator_tex = operators_tex_chroma[corrdata["chroma_index"]]
        flav = corrdata["quark_flavour"]
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
            f"{mom}_{operator}_{flav}_{pol}_{rel}_3pt_ratios_t10_t13_t16.pkl"
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
        # fit_params_ratio = [
        #     fit_param_ratio_boot,
        #     ratio_fit_boot,
        #     fit_param_ratio_avg,
        #     redchisq_ratio,
        #     # best_fit,
        #     delta_t,
        # ]
        fit_params_ratio = {
            "fit_param_boot": fit_param_ratio_boot,
            "fitted_ratio_boot": ratio_fit_boot,
            # "" : fit_param_ratio_avg,
            "red_chisq_fit": redchisq_ratio,
            # best_fit,
            "delta_t": delta_t,
        }

        # three_exp_fitting = {
        #     "ratio_data_t10_t13_t16": ratio_list_reim[ir],
        #     "t10_plateau_fit": t10_plateau_fit,
        #     "t13_plateau_fit": t13_plateau_fit,
        #     "t16_plateau_fit": t16_plateau_fit,
        #     "2expfit_param_boot": fit_param_ratio_boot,
        #     "2expfitted_ratio_boot": ratio_fit_boot,
        #     "red_chisq_fit": redchisq_ratio,
        #     "delta_t": delta_t,
        # }

        # ======================================================================
        # Save the fit results to pickle files
        datafile_ratio_plateau = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_{reim}_3pt_ratio_plateau_fit.pkl"
        )
        with open(datafile_ratio_plateau, "wb") as file_out:
            pickle.dump(plateau_list, file_out)

        datafile_ratio_2exp = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_{reim}_3pt_ratio_2exp_fit.pkl"
        )
        with open(datafile_ratio_2exp, "wb") as file_out:
            pickle.dump(fit_params_ratio, file_out)

        # ======================================================================
        # Save the fit results to csv files
        datafile_ratio_plateau = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_{reim}_3pt_ratio_plateau_fit.csv"
        )
        with open(datafile_ratio_plateau, "w") as csvfile:
            datawrite = csv.writer(csvfile, delimiter=",", quotechar="|")
            datawrite.writerow(["B00(t10)", "B00(t13)", "B00(t16)"])
            for i in range(nboot):
                datawrite.writerow([tsink[i][0] for tsink in plateau_param_list])

        datafile_ratio_2exp = datadir / Path(
            f"{mom}_{operator}_{flav}_{pol}_{rel}_{reim}_3pt_ratio_2exp_fit.csv"
        )
        with open(datafile_ratio_2exp, "w") as csvfile:
            datawrite = csv.writer(csvfile, delimiter=",", quotechar="|")
            datawrite.writerow(["B00"])
            for i in range(nboot):
                datawrite.writerow([fit_param_ratio_boot[i, 0]])

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
        flav = corrdata["quark_flavour"]
        pol = corrdata["pol"]
        reim = corrdata["reim"]
        ir = np.where(np.array(["real", "imag"]) == reim)[0][0]

        if flav == "U":
            kappa_FH = "kp120900tkp120900_kp120900kp120900"
        elif flav == "D":
            kappa_FH = "kp120900tkp120900_kp120900"

        # ======================================================================
        # Read the two-point function data
        twoptfn_filename = latticedir / Path(
            f"twoptfn/barspec/32x64/unpreconditioned_slrc/kp120900kp120900/sh_gij_p21_75-sh_gij_p21_75/{mom}/barspec_nucleon_{rel}_{config_num}cfgs.pickle"
        )
        twoptfn = read_pickle(twoptfn_filename, nboot=500, nbin=1)
        twoptfn_real = twoptfn[:, :, 0]

        # Read in the 3pt function data
        threeptfn_pickle_t10 = latticedir / Path(
            f"bar3ptfn_t10_{flav}/bar3ptfn/32x64/unpreconditioned_slrc/{kappa_FH}/NUCL_{flav}_{pol}_NONREL_gI_t10_{sink_mom}/sh_gij_p21_75-sh_gij_p21_75/{mom}/bar3ptfn_{operator}_{config_num}cfgs.pickle"
        )
        threeptfn_pickle_t13 = latticedir / Path(
            f"bar3ptfn_t13_{flav}/bar3ptfn/32x64/unpreconditioned_slrc/{kappa_FH}/NUCL_{flav}_{pol}_NONREL_gI_t13_{sink_mom}/sh_gij_p21_75-sh_gij_p21_75/{mom}/bar3ptfn_{operator}_{config_num}cfgs.pickle"
        )
        threeptfn_pickle_t16 = latticedir / Path(
            f"bar3ptfn_t16_{flav}/bar3ptfn/32x64/unpreconditioned_slrc/{kappa_FH}/NUCL_{flav}_{pol}_NONREL_gI_t16_{sink_mom}/sh_gij_p21_75-sh_gij_p21_75/{mom}/bar3ptfn_{operator}_{config_num}cfgs.pickle"
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
            f"{mom}_{operator}_{flav}_{pol}_{rel}_3pt_ratios_t10_t13_t16.pkl"
        )
        with open(datafile_ratio, "wb") as file_out:
            pickle.dump(ratio_list_reim, file_out)
    return


def main():
    # Plot style
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

    # ======================================================================
    # List of dictionaries where each dictionary contains the parameters of the correlators we want to analyse. The code will loop over each of the choices and run the analysis.
    corr_choices = [
        {
            "chroma_index": 0,
            "op_name": "Scalar",
            "quark_flavour": "U",
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
            "quark_flavour": "U",
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
            "quark_flavour": "U",
            "pol": "POL",
            "momentum": "p+0+0+0",
            "reim": "imag",
            "delta_t": 4,
            "delta_t_plateau": [4, 4, 4],
            "tmin_choice": 4,
        },
        {
            "chroma_index": 0,
            "op_name": "Scalar",
            "quark_flavour": "D",
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
            "quark_flavour": "D",
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
            "quark_flavour": "D",
            "pol": "POL",
            "momentum": "p+0+0+0",
            "reim": "imag",
            "delta_t": 4,
            "delta_t_plateau": [4, 4, 4],
            "tmin_choice": 4,
        },
    ]

    # ======================================================================
    # Read in the 2pt. an d 3pt. correlators, construct the ratios and save them to files
    save_ratios_zeromom(
        latticedir,
        resultsdir,
        datadir,
        corr_choices,
        operators_chroma,
    )

    fit = True
    # fit = False
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
    plots.plot_3point_zeromom(
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
