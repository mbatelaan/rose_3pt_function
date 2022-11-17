import numpy as np
from pathlib import Path
import pickle
import matplotlib.pyplot as plt

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
