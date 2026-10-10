import time
import numpy as np


def PROPOSED(X, fitness, lowerbound, upperbound, Max_iterations):
    SearchAgents, dimension = X.shape[0], X.shape[1]
    lowerbound = np.ones(dimension) * lowerbound  # Lower limit for variables
    upperbound = np.ones(dimension) * upperbound  # Upper limit for variables

    fit = np.zeros(SearchAgents)
    for i in range(SearchAgents):
        L = X[i, :]
        fit[i] = fitness(L)

    fbest = float('inf')
    Xbest = np.zeros(dimension)

    SOA_curve = np.zeros(Max_iterations)
    ct = time.time()
    for t in range(Max_iterations):
        # Update the best member and worst member
        best = np.min(fit)
        blocation = np.argmin(fit)

        if t == 0:
            Xbest = X[blocation, :]  # Optimal location
            fbest = best  # The optimization objective function
        elif best < fbest:
            fbest = best
            Xbest = X[blocation, :]

        # Update SOA population
        for i in range(SearchAgents):
            # Phase 1: Exploration
            K = np.where(fit < fit[i])[0]
            if K.size != 0:
                KK = np.random.choice(K)
            else:
                KK = i

            expert = X[KK, :]
            worstfit = np.max(fbest)
            Bestfit = np.min(fbest)
            if np.random.rand() < 0.5:
                I = round(1 + np.random.rand())
                r = ((fbest ** 2) / (worstfit ** 2) + (Bestfit))   # PROPOSED Updation
            else:
                I = round(1 + np.random.rand(dimension))
                r = np.random.rand(dimension)

            X_P1 = X[i, :] + r * (expert - I * X[i, :])  # Eq. (3)
            X_P1 = np.clip(X_P1, lowerbound, upperbound)

            # Update position based on Eq (4)
            L = X_P1
            F_P1 = fitness(L)
            if F_P1 < fit[i]:
                X[i, :] = X_P1
                fit[i] = F_P1

        # Phase 2: exploitation (local search)
        for i in range(SearchAgents):
            if np.random.rand() < 0.5:
                X_P2 = X[i, :] + ((1 - 2 * np.random.rand(dimension)) / (t + 1)) * X[i, :]  # Eq.(5)
                X_P2 = np.clip(X_P2, lowerbound, upperbound)
            else:
                X_P2 = X[i, :] + lowerbound / (t + 1) + np.random.rand() * (
                        upperbound / (t + 1) - lowerbound / (t + 1))  # Eq(5)
                X_P2 = np.clip(X_P2, lowerbound / (t + 1), upperbound / (t + 1))
                X_P2 = np.clip(X_P2, lowerbound, upperbound)

            # Update position based on Eq (6)
            L = X_P2
            F_P2 = fitness(L)
            if F_P2 < fit[i]:
                X[i, :] = X_P2
                fit[i] = F_P2

        SOA_curve[t] = fbest

    Best_score = fbest
    Best_pos = Xbest
    ct = time.time() - ct
    return Best_score, SOA_curve, Best_pos, ct
