from typing import Any
from matplotlib import pyplot as plt
import numpy as np


import constants
from helpers import expectation, H_operator, x_operator, normalize, expectation_vectorized, normalize_vectorized


def delta_E(
    psi_plus: np.ndarray[float, Any], psi_minus: np.ndarray[float, Any]
) -> float:
    psi_symmetric = 2**-0.5 * (psi_plus + psi_minus)
    psi_asymmetric = 2**-0.5 * (psi_plus - psi_minus)

    E_0 = expectation(H_operator, psi_symmetric)
    E_1 = expectation(H_operator, psi_asymmetric)

    return E_1 - E_0


def t_tunnel(psi_initial: np.ndarray[complex, Any], timesteps: int) -> float:
    psi_t = psi_initial
    expectation_x = np.zeros(timesteps)

    for t in range(timesteps):
        psi_t = normalize(constants.K @ psi_t)
        expectation_x[t] = expectation(x_operator, psi_t)

    return float(np.argmin(expectation_x))


def run(
    alphas: np.ndarray[float, Any], timesteps: int = 2000
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
        t_tunnels[i] = t_tunnel(psi_plus, timesteps)

    return delta_Es, t_tunnels


def delta_E_vectorized(
    psi_pluses: np.ndarray[complex, Any], psi_minuses: np.ndarray[complex, Any], axis: int
) -> np.ndarray[float, Any]:
    psis_symmetric = 2**-0.5 * (psi_pluses + psi_minuses)
    psis_asymmetric = 2**-0.5 * (psi_pluses - psi_minuses)

    E_0 = expectation_vectorized(H_operator, psis_symmetric, axis=axis)
    E_1 = expectation_vectorized(H_operator, psis_asymmetric, axis=axis)

    return E_1 - E_0


def t_tunnel_vectorized(
    psi_initial: np.ndarray[complex, Any], timesteps: int, axis
) -> np.ndarray[float, Any]:
    psi_t = psi_initial
    expectation_x = np.zeros((timesteps, psi_initial.shape[-1]))

    for t in range(timesteps):
        psi_t = normalize_vectorized(constants.K @ psi_t, axis=axis)
        expectation_x[t] = expectation_vectorized(x_operator, psi_t, axis=axis)

    return np.argmin(expectation_x, axis=0).astype(float)


def run_vectorized(
    alphas: np.ndarray[float, Any], timesteps: int = 1500
) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:

    t_tunnels = np.zeros(alphas.shape)

    x_mins = alphas**-0.5

    psi_pluses = (constants.OMEGA / np.pi) ** (1 / 4) * np.exp(
        -constants.OMEGA / 2 * (constants.X - x_mins) ** 2
    )

    psi_minuses = (constants.OMEGA / np.pi) ** (1 / 4) * np.exp(
        -constants.OMEGA / 2 * (constants.X + x_mins) ** 2
    )

    delta_Es = delta_E_vectorized(psi_pluses, psi_minuses, axis=0)

    t_tunnels = t_tunnel_vectorized(psi_pluses, timesteps, axis=0)


    return delta_Es, t_tunnels


def display(
    data: tuple[np.ndarray[float, Any], np.ndarray[float, Any]], show: bool = False
) -> None:
    delta_Es, t_tunnels = data

    plt.plot(delta_Es, t_tunnels)

    plt.title(r"$\Delta E$ vs $t_{tunnel}$")
    plt.xlabel(r"$\Delta E$")
    plt.ylabel(r"$t_{tunnel}$")

    plt.savefig(fname="problem_d.png")
    if show:
        plt.show()
    plt.clf()
