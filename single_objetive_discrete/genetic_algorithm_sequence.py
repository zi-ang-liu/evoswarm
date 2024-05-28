"""
Implementation of the genetic algorithm for sequence optimization problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from problems import TSP


class GeneticAlgorithm:
    def __init__(
        self,
        problem,
        n_gen=5000,
        n_pop=20,
        mutation_rate=0.01,
        crossover_rate=0.8,
        selection_method="roulette",
        crossover_method="order_crossover",
        mutate_method="swap",
        elitism=True,
    ):
        """
        initialize genetic algorithm
        """
        self.problem = problem
        self.n_gen = n_gen
        self.n_pop = n_pop
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.selection_method = selection_method
        self.crossover_method = crossover_method
        self.mutate_method = mutate_method
        self.elitism = elitism

    def select_parents_roulette(self, population, fitness):
        """
        select parents using roulette wheel selection for minimization problems
        """
        # compute probabilities for minimization problem, the smaller the fitness, the higher the probability
        fitness = np.max(fitness) - fitness
        probabilities = fitness / np.sum(fitness)

        # select parents index using roulette wheel selection without replacement using probabilities
        parents_index = np.random.choice(self.n_pop, 2, replace=False, p=probabilities)
        parents = population[parents_index]

        return parents

    def select_parents(self, population, fitness, selection_method):
        if selection_method == "roulette":
            return self.select_parents_roulette(population, fitness)

    def order_crossover(self, parents):
        """
        order crossover
        """
        # generate crossover point
        crossover_point = np.sort(
            np.random.choice(self.problem.n_elements, 2, replace=False)
        )

        # initialize offspring
        offspring = -1 * np.ones((2, self.problem.n_elements), dtype=int)

        for i in range(2):
            offspring[i, crossover_point[0] : crossover_point[1]] = parents[
                i, crossover_point[0] : crossover_point[1]
            ]
            # Copy the remaining genetic material from the second parent to the first offspring
            remaining = np.setdiff1d(
                parents[1 - i], offspring[i, crossover_point[0] : crossover_point[1]]
            )
            for j in range(self.problem.n_elements):
                if offspring[i, j] == -1:
                    offspring[i, j] = remaining[0]
                    remaining = np.delete(remaining, 0)

        return offspring

    def crossover(self, parents):
        if self.crossover_method == "order_crossover":
            return self.order_crossover(parents)

    def mutate_swap(self, children):
        """
        swap two elements in the child
        """
        # generate mutation point
        mutation_point = np.sort(
            np.random.choice(self.problem.n_elements, 2, replace=False)
        )

        # swap elements
        children[:, mutation_point] = children[:, mutation_point[::-1]]

        return children

    def mutate(self, children):
        if self.mutate_method == "swap":
            return self.mutate_swap(children)

    def optimize(self):

        # initialize population as random permutation of the sequence
        population = np.zeros((self.n_pop, self.problem.n_elements), dtype=int)
        for i in range(self.n_pop):
            population[i] = np.random.permutation(self.problem.n_elements)

        # initialize fitness array
        fitness = np.ones(self.n_pop) * np.inf
        for i in range(self.n_pop):
            fitness[i] = self.problem.evaluate(population[i])

        # initialize best individual
        best_fitness = np.min(fitness)
        best_individual = population[np.argmin(fitness)]

        # initialize array to store best fitness values
        best_fitness_array = np.zeros(self.n_gen)

        # iterate over generations
        for gen in range(self.n_gen):

            # initialize new population
            new_population = np.zeros((self.n_pop, self.problem.n_elements), dtype=int)
            new_fitness = np.zeros((self.n_pop))

            # iterate over population
            for i in range(0, self.n_pop, 2):

                # select parents
                parents = self.select_parents(
                    population, fitness, self.selection_method
                )

                # crossover
                children = self.crossover(parents)

                # mutate
                children = self.mutate(children)

                # evaluate children
                children_fitness = np.zeros(2)
                for j in range(2):
                    children_fitness[j] = self.problem.evaluate(children[j])

                # update new population
                new_population[i] = children[0]
                new_population[i + 1] = children[1]

                # update new fitness
                new_fitness[i] = children_fitness[0]
                new_fitness[i + 1] = children_fitness[1]

            # elitism
            if self.elitism:
                new_population[-1] = best_individual
                new_fitness[-1] = best_fitness

            # update population
            population = new_population
            fitness = new_fitness

            # update best individual
            best_fitness = np.min(fitness)
            best_individual = population[np.argmin(fitness)]

            # store best fitness
            best_fitness_array[gen] = best_fitness

            # save figure for the best individual every 10 generations
            # if gen % 50 == 0:
            #     x = best_individual
            #     plt.figure()
            #     plt.scatter(coordinates[:, 0], coordinates[:, 1])
            #     for i in range(n_cities):
            #         plt.text(coordinates[i, 0], coordinates[i, 1], str(i))
            #     for i in range(n_cities):
            #         plt.plot([coordinates[x[i-1], 0], coordinates[x[i], 0]], [coordinates[x[i-1], 1], coordinates[x[i], 1]], 'r')
            #     plt.savefig(f'figure_{gen}.png')
            #     plt.close()

        return best_individual, best_fitness, best_fitness_array


if __name__ == "__main__":

    # test TSP
    n_cities = 15
    coordinates = np.random.rand(n_cities, 2)

    # compute distance matrix from coordinates
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    print(distance_matrix)

    tsp = TSP(distance_matrix)

    optimizer = GeneticAlgorithm(tsp)
    best_individual, best_fitness, best_fitness_array = optimizer.optimize()

    plt.plot(best_fitness_array)
    plt.show()

    # plot solution
    # x = best_individual
    # plt.figure()
    # plt.scatter(coordinates[:, 0], coordinates[:, 1])
    # for i in range(n_cities):
    #     plt.text(coordinates[i, 0], coordinates[i, 1], str(i))
    # for i in range(n_cities):
    #     plt.plot([coordinates[x[i-1], 0], coordinates[x[i], 0]], [coordinates[x[i-1], 1], coordinates[x[i], 1]], 'r')
    # plt.show()

    # create gif, loop inifinitely
    # import imageio
    # images = []
    # for i in range(0, 5000, 50):
    #     images.append(imageio.imread(f'figure_{i}.png'))
    # imageio.mimsave('tsp.gif', images, duration=0.1)
