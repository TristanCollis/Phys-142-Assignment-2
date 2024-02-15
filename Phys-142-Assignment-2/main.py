import numpy as np
import matplotlib.pyplot as plt
import constants
import helpers
import problem_a
import problem_b
import problem_c
import problem_d

# problem_a.display(problem_a.run(1500), show=True)
# problem_b.display(problem_b.run())
alphas = constants.ALPHA * np.linspace(0.5, 5, 10).reshape(-1,1,1)
Es_v, Ts_v = problem_d.run_vectorized(alphas=alphas, timesteps=5)
plt.plot(alphas.flat, Es_v.flat)
plt.show()
plt.clf()

plt.plot(alphas.flat, Ts_v.flat)
plt.show()
plt.clf()