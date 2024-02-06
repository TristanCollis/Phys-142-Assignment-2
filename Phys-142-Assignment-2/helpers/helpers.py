from typing import Any, Callable
import numpy as np
import constants


def integrate(integrand: np.ndarray[complex, Any], dx: float = constants.DX) -> complex:
    return np.sum(integrand) * dx


def normalize(psi: np.ndarray[complex, Any]) -> np.ndarray[complex, Any]:
    return psi / np.sqrt(integrate(np.abs(psi) ** 2))


def expectation(
    operator: Callable[[np.ndarray[complex, Any]], np.ndarray[complex, Any]],
    psi: np.ndarray[complex, Any],
) -> float:
    return np.real(integrate(operator(psi)))


def partial_dx(
    psi: np.ndarray[complex, Any], dx: float = constants.DX
) -> np.ndarray[complex, Any]:
    return (psi[2:] - psi[:-2]) / (2 * dx)


def x_operator(psi: [np.ndarray[complex, Any]]) -> np.ndarray[complex, Any]:
    return constants.X * np.abs(psi) ** 2


def H_operator(psi: np.ndarray[complex, Any]) -> np.ndarray[complex, Any]:
    return np.conj(psi)[2:-2] * (
        -(0.5) * partial_dx(partial_dx(psi))
        + constants.V(constants.X[2:-2]) * psi[2:-2]
    )