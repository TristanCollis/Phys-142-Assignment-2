from typing import Any
import numpy as np

EPSILON = np.pi / 128

X_0 = -4
X_ND = 4
ND = 600
DX = (X_ND - X_0) / ND

X = np.linspace(X_0, X_ND, ND + 1).reshape(-1, 1)

ALPHA = 0.4

X_MIN = ALPHA ** -(1 / 2)

OMEGA = 2 ** (3 / 2)

PSI_PLUS = (OMEGA / np.pi) ** (1 / 4) * np.exp(-OMEGA / 2 * (X - X_MIN) ** 2)
PSI_MINUS = (OMEGA / np.pi) ** (1 / 4) * np.exp(-OMEGA / 2 * (X + X_MIN) ** 2)

PSI_INITIAL = PSI_PLUS

PSI_SYMMETRIC = 2 ** -(1 / 2) * (PSI_PLUS + PSI_MINUS)
PSI_ASYMMETRIC = 2 ** -(1 / 2) * (PSI_PLUS - PSI_MINUS)


def V(x: np.ndarray[float, Any], alpha: float = ALPHA) -> np.ndarray[float, Any]:
    return alpha * x**4 - 2 * x**2 + 1 / alpha


K = np.exp(1j * ((1 / (2 * EPSILON)) * (X - X.T) ** 2 - V((X.T + X) / 2) * EPSILON))
