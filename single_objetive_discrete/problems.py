'''benchmark problems for single-objective integer optimization'''

import numpy as np
import matplotlib.pyplot as plt

class TSP():
    '''travelling salesman problem'''
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix
        self.n_cities = len(distance_matrix)
        self.n_elements = len(distance_matrix)
        self.name = 'tsp'

    def evaluate(self, x):
        '''evaluate travelling salesman problem'''
        assert len(x) == self.n_cities
        distance = 0
        for i in range(self.n_cities):
            distance += self.distance_matrix[x[i-1], x[i]]
        return distance
    

if __name__ == '__main__':
    
    # test TSP
    coordinates = np.random.rand(5, 2)
    
    # compute distance matrix from coordinates
    n_cities = len(coordinates)
    distance_matrix = np.zeros((n_cities, n_cities))
    for i in range(n_cities):
        for j in range(n_cities):
            distance_matrix[i, j] = np.linalg.norm(coordinates[i] - coordinates[j])
    print(distance_matrix)

    tsp = TSP(distance_matrix)
    x = np.random.permutation(n_cities)
    print(x)
    print(tsp.evaluate(x))

    # plot solution
    plt.figure()
    plt.scatter(coordinates[:, 0], coordinates[:, 1])
    for i in range(n_cities):
        plt.text(coordinates[i, 0], coordinates[i, 1], str(i))
    for i in range(n_cities):
        plt.plot([coordinates[x[i-1], 0], coordinates[x[i], 0]], [coordinates[x[i-1], 1], coordinates[x[i], 1]], 'r')
    plt.show()
