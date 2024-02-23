from typing import Any
from matplotlib import pyplot as plt
import numpy as np


import constants as const
from helpers import (
    expectation,
    H_operator,
    x_operator,
    normalize,
    K,
    V,
    tunnel_curve_approximation,
    inverse_curve_approximation,
    inverse_fit,
)


def delta_E(
    psi_plus: np.ndarray[float, Any],
    psi_minus: np.ndarray[float, Any],
    alpha: np.ndarray[float, Any],
) -> np.ndarray[float, Any]:
    psi_symmetric = 2**-0.5 * (psi_plus + psi_minus)
    psi_asymmetric = 2**-0.5 * (psi_plus - psi_minus)

    H_alpha = lambda psi: H_operator(psi, alpha)

    E_0 = expectation(H_alpha, psi_symmetric)
    E_1 = expectation(H_alpha, psi_asymmetric)

    return np.abs((E_1 - E_0)[:, 0, 0])


def tunnel_time(
    psi: np.ndarray[complex, Any], alpha: np.ndarray[float, Any]
) -> np.ndarray[float, Any]:

    psi_t = psi
    expectation_x = np.zeros((alpha.size, const.NT + 1))

    K_alpha = K(alpha)

    for t in range(const.NT + 1):
        psi_t = normalize(K_alpha @ psi_t)
        expectation_x[:, t] = expectation(x_operator, psi_t).flat

    popts = np.zeros((alpha.shape[0], 3))

    for i, (x_bar, a) in enumerate(zip(expectation_x, alpha)):
        popts[i] = tunnel_curve_approximation(
            x_bar, (1.2, 0.11 * (i / 2 + 1), 0)
        ).flat

    return np.abs(np.pi / popts[:, 1])


def run(
    alpha: np.ndarray[float, Any]
) -> tuple[np.ndarray[float, Any], np.ndarray[float, Any]]:
    x_min = alpha**-0.5

    psi_plus = (const.OMEGA / np.pi) ** (1 / 4) * np.exp(
        -const.OMEGA / 2 * (const.X - x_min) ** 2
    )

    psi_minus = (const.OMEGA / np.pi) ** (1 / 4) * np.exp(
        -const.OMEGA / 2 * (const.X + x_min) ** 2
    )

    delta_Es = delta_E(psi_plus, psi_minus, alpha)
    tunnel_times = tunnel_time(psi_plus, alpha)

    return delta_Es, tunnel_times


def display(
    data: tuple[np.ndarray[float, Any], np.ndarray[float, Any]],
    show: bool = False,
    save: bool = False,
) -> None:
    delta_Es, t_tunnels = data

    proportionality_const = inverse_curve_approximation(delta_Es, t_tunnels)

    plt.scatter(delta_Es, t_tunnels, label=r"$\Delta E$ vs $t_{tunnel}$")
    plt.plot(delta_Es, inverse_fit(delta_Es, proportionality_const), label=f"{proportionality_const:.3f} / "+r"$\Delta E$")
    plt.legend()

    plt.title(r"$\Delta E$ vs $t_{tunnel}$")
    plt.xlabel(r"$\Delta E$")
    plt.ylabel(r"$t_{tunnel}$")

    if save:
        plt.savefig(fname="problem_d.png")
    if show:
        plt.show()
        
