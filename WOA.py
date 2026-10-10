import time
import numpy as np


def WOA(pop, objective_function, lower_bound, upper_bound, max_iter):
    """
    Wombat Optimization Algorithm (WOA)
    Inspired by wombat behavior: foraging (exploration) and escaping (exploitation)

    Parameters:
    - objective_function: Function to be minimized
    - pop_size: Number of wombats (population size)
    - dim: Number of dimensions (variables in the optimization problem)
    - lower_bound: Minimum value for each variable
    - upper_bound: Maximum value for each variable
    - max_iter: Number of iterations

    Returns:
    - best_solution: Best solution found
    - best_fitness: Best objective function value
    """
    pop_size, dim = pop.shape[0], pop.shape[1]

    # Evaluate the initial fitness of each wombat
    fitness = np.apply_along_axis(objective_function, 1, pop)

    # Identify the best wombat (solution) in the initial population
    best_idx = np.argmin(fitness)
    best_solution, best_fitness = pop[best_idx], fitness[best_idx]
    convergence = np.zeros(max_iter)
    ct = time.time()
    # Main optimization loop
    for t in range(max_iter):
        # ----------------------------- Foraging Phase (Exploration) ----------------------------- #
        for i in range(pop_size):
            # Select candidate wombats with better fitness
            candidates = np.where(fitness < fitness[i])[0]

            if len(candidates) > 0:
                # Choose a random better solution as a target
                target = pop[np.random.choice(candidates)]

                # Move towards the target with some randomness
                new_pos = pop[i] + np.random.rand(dim) * (target - np.random.randint(1, 3) * pop[i])

                # Keep the new position within bounds
                pop[i] = np.clip(new_pos, lower_bound[i], upper_bound[i])

        # ----------------------------- Escape Phase (Exploitation) ----------------------------- #
        for i in range(pop_size):
            # Adjust the position using a decay factor based on iteration count
            new_pos = pop[i] + (1 - 2 * np.random.rand(dim)) * ((upper_bound - lower_bound) / (t + 1))

            # Keep the new position within bounds
            pop[i] = np.clip(new_pos[i], lower_bound[i], upper_bound[i])

        # Update fitness values
        fitness = np.apply_along_axis(objective_function, 1, pop)

        # Identify the best solution in the updated population
        best_idx = np.argmin(fitness)
        best_solution, best_fitness = pop[best_idx], fitness[best_idx]
        convergence[t] = best_fitness

    ct = time.time() - ct
    return best_fitness, convergence, best_solution, ct

