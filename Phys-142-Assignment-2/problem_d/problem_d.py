from typing import Any
from matplotlib import pyplot as plt
import numpy as np


import constants
from helpers import expectation, H_operator, x_operator, normalize, expectation_vectorized, normalize_vectorized, K, V


def delta_E(
    psi_plus: np.ndarray[float, Any], psi_minus: np.ndarray[float, Any], alpha: np.ndarray[float, Any]
) -> float:
    psi_symmetric = 2**-0.5 * (psi_plus + psi_minus)
    psi_asymmetric = 2**-0.5 * (psi_plus - psi_minus)

    H_alpha = lambda psi: H_operator(psi, alpha)

    E_0 = expectation(H_alpha, psi_symmetric)
    E_1 = expectation(H_alpha, psi_asymmetric)

    return E_1 - E_0


def t_tunnel(psi_initial: np.ndarray[complex, Any], alpha: np.ndarray[float, Any], timesteps: int) -> float:
    psi_t = psi_initial
    expectation_x = np.zeros(timesteps)

    for t in range(timesteps):
        psi_t = normalize(K(alpha) @ psi_t)
        expectation_x[t] = expectation(x_operator, psi_t)

    return float(np.argmin(expectation_x))


def run(
    alphas: np.ndarray[float, Any], timesteps: int = 1500
) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:

    delta_Es = np.zeros(alphas.shape)
    t_tunnels = np.zeros(alphas.shape)

    x_mins = alphas**-0.5

    for i, (x_min, alpha) in enumerate(zip(x_mins, alphas)):

        psi_plus = (constants.OMEGA / np.pi) ** (1 / 4) * np.exp(
            -constants.OMEGA / 2 * (constants.X - x_min) ** 2
        )
        psi_minus = (constants.OMEGA / np.pi) ** (1 / 4) * np.exp(
            -constants.OMEGA / 2 * (constants.X + x_min) ** 2
        )

        delta_Es[i] = delta_E(psi_plus, psi_minus, alpha)
        t_tunnels[i] = t_tunnel(psi_plus, alpha, timesteps)

    return delta_Es, t_tunnels


def delta_E_vectorized(
    psi_pluses: np.ndarray[complex, Any], psi_minuses: np.ndarray[complex, Any], alphas: np.ndarray[float, Any], axis: int
) -> np.ndarray[float, Any]:
    psis_symmetric = 2**-0.5 * (psi_pluses + psi_minuses)
    psis_asymmetric = 2**-0.5 * (psi_pluses - psi_minuses)

    H_alphas = lambda psi: (H_operator(psi, alphas))

    E_0 = expectation_vectorized(H_alphas, psis_symmetric.transpose(1,0,2), axis=axis)
    E_1 = expectation_vectorized(H_alphas, psis_asymmetric.transpose(1,0,2), axis=axis)

    return E_1 - E_0


def t_tunnel_vectorized(
    psi_initial: np.ndarray[complex, Any], alphas: np.ndarray[float, Any], timesteps: int, axis: int
) -> np.ndarray[float, Any]:
    psi_t = psi_initial
    expectation_x = np.zeros((timesteps, alphas.shape[0]))

    for t in range(timesteps):
        psi_t = normalize_vectorized(K(alphas) @ psi_t, axis=axis)
        expectation_x[t] = expectation_vectorized(x_operator, psi_t, axis=axis).flat

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

    delta_Es = delta_E_vectorized(psi_pluses, psi_minuses, alphas, axis=1)

    t_tunnels = t_tunnel_vectorized(psi_pluses, alphas, timesteps, axis=1)


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
