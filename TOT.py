import time

import numpy as np


def TOT(population, objective_function, lb, ub, max_iter):
    """Three-on-Three Optimization Algorithm"""

    # Initialize population
    pop_size, dim = population.shape[0], population.shape[1]
    fitness = np.array([objective_function(ind) for ind in population])

    # Track best solution
    best_idx = np.argmin(fitness)
    bestsol = population[best_idx]
    bestfit = fitness[best_idx]

    convergence = [bestfit]  # Store fitness over iterations
    ct = time.time()
    for iteration in range(max_iter):
        new_population = population.copy()

        # Apply 3-on-3 modification
        for i in range(0, pop_size - 2, 3):
            idx = np.random.choice(pop_size, 3, replace=False)
            x1, x2, x3 = population[idx]

            # Apply simple crossover or mutation
            new_x = (x1 + x2 + x3) / 3  # Combine three solutions

            # Ensure within bounds
            new_x = np.clip(new_x, lb, ub)

            # Evaluate new solution
            new_fitness = objective_function(new_x)

            # Replace worst solution among the three
            worst_idx = np.argmax([fitness[idx[0]], fitness[idx[1]], fitness[idx[2]]])
            new_population[idx[worst_idx]] = new_x[idx[worst_idx]]
            fitness[idx[worst_idx]] = new_fitness[idx[worst_idx]]

            # Update best solution
            if new_fitness[i] < bestfit:
                bestfit = new_fitness[i]
                bestsol = new_x[i]

        population = new_population
        convergence.append(bestfit)
    ct = time.time() - ct
    return bestfit, convergence, bestsol, ct
