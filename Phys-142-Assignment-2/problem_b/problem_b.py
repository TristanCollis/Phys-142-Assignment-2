from typing import Any
import numpy as np

from helpers import expectation, H_operator
import constants


def run() -> tuple[float, float]:
    E_0 = expectation(H_operator, constants.PSI_SYMMETRIC)
    E_1 = expectation(H_operator, constants.PSI_ASYMMETRIC)

    return E_0[0,0,0], E_1[0,0,0]


def display(energy_values: tuple[float, float]) -> None:
    E_0, E_1 = energy_values
    print(f"E_0 = {E_0}")
    print(f"E_1 = {E_1}")
    print(f"delta E = {abs(E_0 - E_1)}")
