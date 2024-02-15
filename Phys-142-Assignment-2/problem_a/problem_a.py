from typing import Any
import numpy as np
import matplotlib.pyplot as plt

import constants
from helpers import expectation, normalize, x_operator, K_0


def run(timesteps: int) -> np.ndarray[float, Any]:
    psi_t = constants.PSI_INITIAL
    expectation_x = np.zeros(timesteps)

    for t in range(timesteps):
        expectation_x[t] = expectation(x_operator, psi_t)
        psi_t = normalize(K_0 @ psi_t)

    return expectation_x


def display(expectation_x: np.ndarray[float, Any], show: bool = False) -> None:
    plt.plot(expectation_x)

    plt.title(r"$\langle x \rangle$ vs Time")
    plt.xlabel(r"$t$")
    plt.ylabel(r"$\langle x \rangle $")

    plt.savefig(fname="problem_a.png")
    if show:
        plt.show()
    plt.clf()
