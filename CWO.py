import time

import numpy as np


def CWO(population, obj_func, lb, ub, max_iter):
    """
    Carpet Weaver Optimization (CWO) algorithm.

    Parameters:
    obj_func : function - Objective function to minimize
    lb : list - Lower bounds for decision variables
    ub : list - Upper bounds for decision variables
    dim : int - Number of decision variables
    pop_size : int - Population size (default=30)
    max_iter : int - Maximum number of iterations (default=100)

    Returns:
    best_solution : ndarray - Best solution found
    best_score : float - Best objective function value
    """
    # Initialize population
    pop_size, dim = population.shape[0], population.shape[1]
    fitness = np.apply_along_axis(obj_func, 1, population)

    # Best solution tracking
    best_idx = np.argmin(fitness)
    best_solution = population[best_idx].copy()
    best_score = fitness[best_idx]
    convergence = np.zeros(max_iter)
    ct = time.time()
    for t in range(max_iter):
        # Phase 1: Exploration - Weaving based on pattern
        pattern = np.random.uniform(lb, ub, dim)
        new_population = population + (1 - 2 * np.random.rand(pop_size, dim)) * (
                    pattern - np.random.randint(1, 3) * population)
        new_population = np.clip(new_population, lb, ub)

        new_fitness = np.apply_along_axis(obj_func, 1, new_population)
        improved = new_fitness < fitness
        population[improved] = new_population[improved]
        fitness[improved] = new_fitness[improved]

        # Phase 2: Exploitation - Creative modifications
        new_population = (1 + ((1 - 2 * np.random.rand(pop_size, dim)) / t)) * population
        new_population = np.clip(new_population, lb, ub)

        new_fitness = np.apply_along_axis(obj_func, 1, new_population)
        improved = new_fitness < fitness
        population[improved] = new_population[improved]
        fitness[improved] = new_fitness[improved]

        # Update best solution
        best_idx = np.argmin(fitness)
        if fitness[best_idx] < best_score:
            best_solution = population[best_idx].copy()
            best_score = fitness[best_idx]
        convergence[t] = best_score

    ct = time.time() - ct
    return best_score, convergence, best_solution, ct


