from typing import Any
from matplotlib import pyplot as plt
import numpy as np


import constants
from helpers import expectation, H_operator, x_operator, normalize


def delta_E(
    psi_plus: np.ndarray[float, Any], psi_minus: np.ndarray[float, Any]
) -> float:
    psi_symmetric = 2**-0.5 * (psi_plus + psi_minus)
    psi_asymmetric = 2**-0.5 * (psi_plus - psi_minus)

    E_0 = expectation(H_operator, psi_symmetric)
    E_1 = expectation(H_operator, psi_asymmetric)

    return E_1 - E_0


def t_tunnel(psi_initial: np.ndarray[complex, Any]) -> float:
    timesteps = 3000

    psi_t = psi_initial
    expectation_x = np.zeros(timesteps)

    for t in range(timesteps):
        psi_t = normalize(constants.K @ psi_t)
        expectation_x[t] = expectation(x_operator, psi_t)

    return float(np.argmin(expectation_x))


def run(
    alphas: np.ndarray[float, Any]
) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:

    delta_Es = np.zeros(alphas.shape)
    t_tunnels = np.zeros(alphas.shape)

    x_mins = alphas**-0.5

    for i, x_min in enumerate(x_mins):

        psi_plus = (constants.OMEGA / np.pi) ** (1 / 4) * np.exp(
            -constants.OMEGA / 2 * (constants.X - x_min) ** 2
        )
        psi_minus = (constants.OMEGA / np.pi) ** (1 / 4) * np.exp(
            -constants.OMEGA / 2 * (constants.X + x_min) ** 2
        )

        delta_Es[i] = delta_E(psi_plus, psi_minus)
        t_tunnels[i] = t_tunnel(psi_plus)

    return delta_Es, t_tunnels


def display(
    data: tuple[np.ndarray[float, Any], np.ndarray[float, Any]], show: bool = False
) -> None:
    delta_Es, t_tunnels = data

    plt.plot(delta_Es, t_tunnels)

    plt.title(r"$\Delta E vs t_{tunnel}$")
    plt.xlabel(r"$\Delta E$")
    plt.ylabel(r"$t_{tunnel}$")

    plt.savefig(fname="problem_a.png")
    if show:
        plt.show()
    plt.clf()
