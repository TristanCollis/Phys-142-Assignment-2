from typing import Any, Callable
import numpy as np
import constants


def integrate(integrand: np.ndarray[complex, Any], dx: float = constants.DX) -> complex:
    return np.sum(integrand) * dx


def integrate_vectorized(integrand: np.ndarray[complex, Any],  axis: int, dx: float = constants.DX) -> np.ndarray[complex, Any]:
    return np.sum(integrand, axis=axis) * dx


def normalize(psi: np.ndarray[complex, Any], axis: int | None = None) -> np.ndarray[complex, Any]:
    return psi / np.sqrt(integrate(np.abs(psi) ** 2))


def normalize_vectorized(psi: np.ndarray[complex, Any], axis: int) -> np.ndarray[complex, Any]:
    return psi / np.sqrt(integrate_vectorized(np.abs(psi) ** 2, axis=axis))


def expectation(
    operator: Callable[[np.ndarray[complex, Any]], np.ndarray[complex, Any]],
    psi: np.ndarray[complex, Any],
    axis: int | None = None
) -> float:
    return np.real(integrate(operator(psi)))


def expectation_vectorized(
        operator: Callable[[np.ndarray[complex, Any]], np.ndarray[complex, Any]],
    psi: np.ndarray[complex, Any],
    axis: int
) -> np.ndarray[float, Any]:
    return np.real(integrate_vectorized(operator(psi), axis=axis))


def partial_dx(
    psi: np.ndarray[complex, Any], dx: float = constants.DX
) -> np.ndarray[complex, Any]:
    return (psi[2:] - psi[:-2]) / (2 * dx)


def x_operator(psi: np.ndarray[complex, Any], x: np.ndarray[float, Any] = constants.X) -> np.ndarray[complex, Any]:
    return x * np.abs(psi) ** 2


def H_operator(psi: np.ndarray[complex, Any]) -> np.ndarray[complex, Any]:
    return np.conj(psi)[2:-2] * (
        -(0.5) * partial_dx(partial_dx(psi))
        + constants.V(constants.X[2:-2]) * psi[2:-2]
    )