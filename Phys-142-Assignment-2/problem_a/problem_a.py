from typing import Any
import numpy as np
import matplotlib.pyplot as plt

import constants as const
from helpers import (
    expectation,
    normalize,
    x_operator,
    K_0,
    tunnel_curve_approximation,
    cos_fit,
)


def run() -> tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:
    psi_t = const.PSI_INITIAL
    expectation_x = np.zeros(const.NT + 1)

    for t in range(const.NT + 1):
        expectation_x[t] = expectation(x_operator, psi_t)
        psi_t = normalize(K_0 @ psi_t)

    popt = tunnel_curve_approximation(expectation_x, (1.2, 0.11, -0.02))

    return expectation_x, popt


def display(
    expectation_x: np.ndarray[float, Any],
    popt: np.ndarray[float, Any],
    show: bool = False,
    save: bool = False
) -> None:
    plt.plot(const.T, expectation_x)
    print(f"approximate tunneling time: {np.abs(np.pi / popt[1])}")
    plt.plot(const.T, cos_fit(const.T, *popt))

    plt.title(r"$\langle x \rangle$ vs Time")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\langle x \rangle $")

    if save:
        plt.savefig(fname="problem_a.png")

    if show:
        plt.show()



