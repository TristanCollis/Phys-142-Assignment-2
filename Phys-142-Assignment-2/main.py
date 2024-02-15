import numpy as np
import constants as const
import problem_a
import problem_b
import problem_c
import problem_d

problem_a.display(*problem_a.run(), show=True, save=True)

problem_b.display(problem_b.run())

problem_c.run()

alpha = const.ALPHA * np.linspace(1, 2, 9).reshape(-1,1,1)
problem_d.display(problem_d.run(alpha=alpha), show=True, save=True)
