import numpy as np

import constants as const
from helpers import animate, K_0, normalize


def run(filename: str = "problem_c.mp4") -> None:
    psi_t = const.PSI_INITIAL
    psi_pdf = np.zeros((const.NT + 1, const.ND + 1))

    for t in range(const.NT + 1):
        psi_pdf[t] = (np.abs(psi_t) ** 2).flat
        psi_t = normalize(K_0 @ psi_t)

    animate(np.squeeze(psi_pdf), np.squeeze(const.X), filename)


def display() -> None: ...
