from typing import Any, Callable
import numpy as np

X_0 = -4
X_ND = 4
ND = 600
DX = (X_ND - X_0) / ND

X = np.linspace(X_0, X_ND, ND + 1).reshape(1, -1, 1)

PERIOD = 20 * np.pi
EPSILON = np.pi / 128
NT = int(PERIOD / EPSILON)

T = np.linspace(0, PERIOD, NT + 1)

ALPHA = 0.4
ALPHA_AS_ARRAY = np.array([ALPHA]).reshape(1,1,1)

X_MIN = ALPHA_AS_ARRAY ** -(1 / 2)

OMEGA = 2 ** (3 / 2)

PSI_PLUS = (OMEGA / np.pi) ** (1 / 4) * np.exp(-OMEGA / 2 * (X - X_MIN) ** 2)
PSI_MINUS = (OMEGA / np.pi) ** (1 / 4) * np.exp(-OMEGA / 2 * (X + X_MIN) ** 2)

PSI_INITIAL = PSI_PLUS

PSI_SYMMETRIC = 2 ** -(1 / 2) * (PSI_PLUS + PSI_MINUS)
PSI_ASYMMETRIC = 2 ** -(1 / 2) * (PSI_PLUS - PSI_MINUS)
