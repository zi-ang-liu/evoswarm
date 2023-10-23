'''
Implementation of the particle swarm optimization algorithm.
Non-vectorized version for learning purpose.
To speed up, please use vectorized version.
'''

import numpy as np
import matplotlib.pyplot as plt
from problems import *


class ParticleSwarm():

    def __init__(self, problem, n_gen=100, n_pop=10, w=0.8, c1=0.5, c2=0.5):
        self.problem = problem
        self.n_dim = problem.n_dim
        self.n_gen = n_gen
        self.n_pop = n_pop
        self.ub = problem.upper_bound
        self.lb = problem.lower_bound
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def optimize(self):

        # initialization
        x = self.lb + (self.ub - self.lb) * \
            np.random.rand(self.n_pop, self.n_dim)
        v = np.zeros((self.n_pop, self.n_dim))
        f_x = np.zeros((self.n_pop, 1))
        pbest_x = np.zeros((self.n_pop, self.n_dim))
        pbest_f_x = np.zeros((self.n_pop, 1))
        gbest_x = np.zeros((1, self.n_dim))
        gbest_f_x = np.zeros((1, 1))

        for i in range(self.n_pop):
            f_x[i] = self.problem.evaluate(x[i])
            pbest_x[i] = x[i]
            pbest_f_x[i] = f_x[i]

        gbest_x = pbest_x[np.argmin(pbest_f_x)]
        gbest_f_x = np.min(pbest_f_x)

        gen = 0

        # data for convergence graph
        record_obj_val = np.zeros(self.n_gen+1)
        record_obj_val[gen] = gbest_f_x

        # iteration
        while gen < self.n_gen:

            for i in range(self.n_pop):

                # update velocity
                v[i] = self.w * v[i] + self.c1 * np.random.rand(1, self.n_dim) * (pbest_x[i] - x[i]) + \
                    self.c2 * np.random.rand(1, self.n_dim) * (gbest_x - x[i])

                # update position
                x[i] = x[i] + v[i]

                # repair
                mask_bound = np.random.uniform(
                    low=self.lb, high=self.ub, size=(self.n_dim))
                x[i] = np.where(x[i] < self.lb, mask_bound, x[i])
                x[i] = np.where(x[i] > self.ub, mask_bound, x[i])

                # evaluation
                f_x[i] = self.problem.evaluate(x[i])

                # update pbest
                if f_x[i] < pbest_f_x[i]:
                    pbest_x[i] = x[i]
                    pbest_f_x[i] = f_x[i]
                    if pbest_f_x[i] < gbest_f_x:
                        gbest_x = pbest_x[i]
                        gbest_f_x = pbest_f_x[i]

            gen = gen + 1

            # record
            record_obj_val[gen] = gbest_f_x

        return gbest_x, gbest_f_x, record_obj_val
    
if __name__ == '__main__':

    # create problem
    problem = Sphere(n_dim=10)

    # create optimizer and solve problem
    optimizer = ParticleSwarm(problem)
    solution, solution_fitness, record_obj_val = optimizer.optimize()
    print('Solution: ', solution)
    print('Solution fitness: ', solution_fitness)

    # plot convergence
    plt.plot(np.arange(optimizer.n_gen+1), record_obj_val)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()    
