'''basic bhenchmark functions for single objective optimization'''

import numpy as np

class Ackley():
    '''ackley function'''
    def __init__(self, n_n_dim=2, bounds=None):
        self.n_n_dim = n_n_dim
        if bounds is None:
            bounds = [-32.768, 32.768]
            self.lower_bound = np.array([bounds[0]]*n_n_dim)
            self.upper_bound = np.array([bounds[1]]*n_n_dim)
        else:
            self.lower_bound = np.array(bounds[0])
            self.upper_bound = np.array(bounds[1])
            assert len(self.lower_bound) == n_n_dim
            assert len(self.upper_bound) == n_n_dim
        self.optimal = 0
        self.optimal_x = np.zeros(n_n_dim)
        self.name = 'ackley'

    def evaluate(self, x):
        '''evaluate ackley function'''
        assert len(x) == self.n_n_dim
        a = 20
        b = 0.2
        c = 2*np.pi
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c*x))
        term1 = -a*np.exp(-b*np.sqrt(sum1/self.n_n_dim))
        term2 = -np.exp(sum2/self.n_n_dim)
        return term1 + term2 + a + np.exp(1)

class Rastrigin():
    '''rastrigin function'''
    def __init__(self, n_dim=2, bounds=None):
        self.n_dim = n_dim
        if bounds is None:
            bounds = [-5.12, 5.12]
            self.lower_bound = np.array([bounds[0]]*n_dim)
            self.upper_bound = np.array([bounds[1]]*n_dim)
        else:
            self.lower_bound = np.array(bounds[0])
            self.upper_bound = np.array(bounds[1])
            assert len(self.lower_bound) == n_dim
            assert len(self.upper_bound) == n_dim
        self.optimal = 0
        self.optimal_x = np.zeros(n_dim)
        self.name = 'rastrigin'

    def evaluate(self, x):
        '''evaluate rastrigin function'''
        assert len(x) == self.n_dim
        return 10*self.n_dim + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    
class Sphere():
    '''sphere function'''
    def __init__(self, n_dim=2, bounds=None):
        self.n_dim = n_dim
        if bounds is None:
            bounds = [-5.12, 5.12]
            self.lower_bound = np.array([bounds[0]]*n_dim)
            self.upper_bound = np.array([bounds[1]]*n_dim)
        else:
            self.lower_bound = np.array(bounds[0])
            self.upper_bound = np.array(bounds[1])
            assert len(self.lower_bound) == n_dim
            assert len(self.upper_bound) == n_dim
        self.optimal = 0
        self.optimal_x = np.zeros(n_dim)
        self.name = 'sphere'

    def evaluate(self, x):
        '''evaluate sphere function'''
        assert len(x) == self.n_dim
        return np.sum(x**2)
    

if __name__ == '__main__':
    function = Ackley(3)
    x = np.array([1, 2])
    print(function.evaluate(x))

    function = Ackley(n_n_dim=3)
    x = np.array([1, 2, 3])
    print(function.evaluate(x))

