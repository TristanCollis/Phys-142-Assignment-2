from typing import Any, Callable
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import constants as const
from scipy.optimize import curve_fit


def V(
    x: np.ndarray[float, Any], alphas: np.ndarray[float, Any] = const.ALPHA_AS_ARRAY
) -> np.ndarray[float, Any]:
    return alphas * x**4 - 2 * x**2 + 1 / alphas


def K(
    alphas: np.ndarray[float, Any] = const.ALPHA_AS_ARRAY
) -> np.ndarray[complex, Any]:
    return np.exp(
        1j
        * (
            (1 / (2 * const.EPSILON)) * (const.X.transpose(0, 2, 1) - const.X) ** 2
            - V((const.X.transpose(0, 2, 1) + const.X) / 2, alphas) * const.EPSILON
        )
    )


K_0 = K()


def integrate_x(
    integrand: np.ndarray[complex, Any], dx: float = const.DX, axis: int = 1
) -> np.ndarray[complex, Any]:
    return np.expand_dims(np.sum(integrand, axis=axis), axis=axis) * dx


def normalize(psi: np.ndarray[complex, Any]) -> np.ndarray[complex, Any]:
    return psi / np.sqrt(integrate_x(np.abs(psi) ** 2))


def expectation(
    operator: Callable[[np.ndarray[complex, Any]], np.ndarray[complex, Any]],
    psi: np.ndarray[complex, Any],
    axis: int = 1,
) -> np.ndarray[float, Any]:
    return np.real(integrate_x(operator(psi), axis=axis))


def propagate(
    K: np.ndarray[complex, Any], psi: np.ndarray[complex, Any]
) -> np.ndarray[complex, Any]:
    return normalize((K @ psi.transpose(1, 2, 0)).transpose(2, 0, 1))


def partial_dx(
    psi: np.ndarray[complex, Any], dx: float = const.DX
) -> np.ndarray[complex, Any]:
    return (psi[:, 2:, :] - psi[:, :-2, :]) / (2 * dx)


def x_operator(
    psi: np.ndarray[complex, Any], x: np.ndarray[float, Any] = const.X
) -> np.ndarray[complex, Any]:
    return x * np.abs(psi) ** 2


def H_operator(
    psi: np.ndarray[complex, Any], alpha: np.ndarray[float, Any] = const.ALPHA_AS_ARRAY
) -> np.ndarray[complex, Any]:
    return np.conj(psi)[:, 2:-2, :] * (
        -(0.5) * partial_dx(partial_dx(psi))
        + V(const.X, alpha)[:, 2:-2, :] * psi[:, 2:-2, :]
    )


def cos_fit(
    t: np.ndarray[float, Any], A: float, B: float, C: float
) -> np.ndarray[float, Any]:
    return A * np.cos(B * t + C)


def inverse_fit(x: np.ndarray[float, Any], A: float) -> np.ndarray[float, Any]:
    return A / x


def tunnel_curve_approximation(
    x_bar: np.ndarray[float, Any], p0: tuple[float, float, float]
) -> np.ndarray[float, Any]:

    popt, _ = curve_fit(cos_fit, const.T, x_bar, p0=p0)
    return popt


def inverse_curve_approximation(
    delta_Es: np.ndarray[float, Any], t_tunnels: np.ndarray[float, Any]
) -> float:
    popt, _ = curve_fit(inverse_fit, delta_Es, t_tunnels)

    return popt[0]


def animate(
    frame_data: np.ndarray[float, Any], x_axis: np.ndarray[float, Any], filename: str
) -> None:
    fig, ax = plt.subplots()
    (line1,) = ax.plot([], [], "b.")

    def init():
        ax.set_xlim(x_axis[0], x_axis[-1])
        ax.set_ylim(0, 1)

        return (line1,)

    def update(frame):
        line1.set_data(x_axis, frame_data[frame])
        return line1

    ani = FuncAnimation(
        fig, update, frames=np.arange(frame_data.shape[0]), init_func=init, interval=50
    )

    ani.save(filename)
