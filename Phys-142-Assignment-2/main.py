import numpy as np
import constants
import helpers
import problem_a
import problem_b
import problem_c
import problem_d

# problem_a.display(problem_a.run(3000), show=True)
# problem_b.display(problem_b.run())

alphas = constants.ALPHA * np.linspace(0.5, 2, 25)

# problem_d.display(problem_d.run_vectorized(constants.ALPHA * np.linspace(0.5, 2, 20)), show=True)