import numpy as np
from Glob_Vars import Glob_Vars
from Model_RNN import Model_STA_ABi_RNN


def Objfun_Cls(Soln):
    Data = Glob_Vars.Data
    Target = Glob_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        learnper = round(Data.shape[0] * 0.75)
        train_data = Data[learnper:, :]
        train_target = Target[learnper:, :]
        test_data = Data[:learnper, :]
        test_target = Target[:learnper, :]
        Eval = Model_STA_ABi_RNN(train_data, train_target, test_data, test_target, sol[i].astype('int'))
        Fitn[i] = 1 / (Eval[4] + Eval[12])  # (Accuracy + F1Score)
    return Fitn
