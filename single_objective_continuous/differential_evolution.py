'''
Implementation of the differential evolution algorithm.
Non-vectorized version for learning purpose.
To speed up, please use vectorized version.
'''

import numpy as np
import matplotlib.pyplot as plt
from benchmark import *

class DifferentialEvolution():

    def __init__(self, problem, n_gen=100, n_pop=10, F=0.8, CR=0.5):
        self.problem = problem
        self.n_dim = problem.n_dim
        self.n_gen = n_gen
        self.n_pop = n_pop
        self.ub = problem.upper_bound
        self.lb = problem.lower_bound
        self.F = F
        self.CR = CR

    def optimize(self):
       
        # initialization
        x = self.lb + (self.ub - self.lb) * np.random.rand(self.n_pop, self.n_dim)
        u = np.zeros((self.n_pop, self.n_dim))
        v = np.zeros((self.n_pop, self.n_dim))
        f_x = np.zeros((self.n_pop,1))
        f_u = np.zeros((self.n_pop,1))
        
        for i in range(self.n_pop):
            f_x[i] = self.problem.evaluate(x[i])

        gen = 0

        # data for convergence graph
        record_obj_val = np.zeros(self.n_gen+1)
        record_obj_val[gen] = np.min(f_x)
        
        # iteration
        while gen < self.n_gen:

            for i in range(self.n_pop):

                # mutation
                random_idx = np.random.randint(0, self.n_pop, size=(3))
                r1, r2, r3 = random_idx[0], random_idx[1], random_idx[2]
                # v[i]=x[r1]+F(x[r2]-x[r3])
                v = x[r1] + self.F * (x[r2] - x[r3])

                # repair
                mask_bound = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_dim))
                v = np.where(v < self.lb, mask_bound, v)
                v = np.where(v > self.ub, mask_bound, v)

                # crossover
                mask_co = (np.random.rand(self.n_dim) < self.CR)
                jrand =np.random.randint(0, self.n_dim, size=(1))
                mask_co[jrand] = True
                u = np.where(mask_co, v, x[i])

                # evaluation
                f_u = self.problem.evaluate(u)

                # selection
                x[i] = np.where(f_u<f_x[i], u, x[i])
                f_x[i] = np.where(f_u<f_x[i], f_u, f_x[i])

            gen = gen + 1

            # record
            record_obj_val[gen] = np.min(f_x)
            # print('Generation: ', gen, 'Best fitness: ', np.min(f_x))


        return x[np.argmin(f_x)], np.min(f_x), record_obj_val

if __name__ == '__main__':

    # create problem
    problem = Sphere(n_dim=10)

    # create optimizer and solve problem
    optimizer = DifferentialEvolution(problem)
    solution, solution_fitness, record_obj_val = optimizer.optimize()
    print('Solution: ', solution)
    print('Solution fitness: ', solution_fitness)

    # plot convergence
    plt.plot(np.arange(optimizer.n_gen+1), record_obj_val)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.show()