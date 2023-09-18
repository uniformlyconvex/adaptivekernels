import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import torch
import pandas as pd

from src.designs.demand import Demand
from src.models.mekiv import MEKIV
from src.models.kiv_adaptive_ridge import KIVAdaptiveRidge
from src.structures.stage_data import StageData

LMBDA_SEARCH_SPACE = torch.exp(torch.linspace(-8,3,100))
XI_SEARCH_SPACE = torch.exp(torch.linspace(-8,3,100))
RHOS = {0.25, 0.5, 0.9}
MERRORS = {0.5,1,2}
NO_RUNS = 20


def get_pandas_csv(rho: float, merror: float) -> str:
    return f'./results/mekiv_{rho}_{merror}.csv'

def do_run(rho: float, merror: float, run: int) -> None:
    design = Demand(rho=rho)
    
    X, M, N, Y, Z = design.generate_MEKIV_data(no_points=1000, merror_type = 'gaussian', merror_scale=merror)
    test_data = design.generate_test_data(1000)

    mekiv = MEKIV(M, N, Y, Z, lmbda_search_space=LMBDA_SEARCH_SPACE, xi_search_space=XI_SEARCH_SPACE, real_X=X)
    mekiv_preds = mekiv.predict(test_data.X)
    mekiv_mse = test_data.evaluate_preds(mekiv_preds)

    MN_stagedata = StageData.from_all_data((M+N)/2, Y, Z)
    M_stagedata = StageData.from_all_data(M, Y, Z)
    Oracle = StageData.from_all_data(X,Y,Z)

    kiv_MN = KIVAdaptiveRidge(MN_stagedata, lmbda=LMBDA_SEARCH_SPACE, xi=XI_SEARCH_SPACE)
    kiv_MN_preds = kiv_MN.predict(test_data.X)
    kiv_MN_mse = test_data.evaluate_preds(kiv_MN_preds)

    kiv_M = KIVAdaptiveRidge(M_stagedata, lmbda=LMBDA_SEARCH_SPACE, xi=XI_SEARCH_SPACE)
    kiv_M_preds = kiv_M.predict(test_data.X)
    kiv_M_mse = test_data.evaluate_preds(kiv_M_preds)

    kiv_Oracle = KIVAdaptiveRidge(Oracle, lmbda=LMBDA_SEARCH_SPACE, xi=XI_SEARCH_SPACE)
    kiv_Oracle_preds = kiv_Oracle.predict(test_data.X)
    kiv_Oracle_mse = test_data.evaluate_preds(kiv_Oracle_preds)


    print(f'KIV-MN: {kiv_MN_mse:.4f}')
    print(f'KIV-M: {kiv_M_mse:.4f}')
    print(f'KIV-Oracle: {kiv_Oracle_mse:.4f}')
    print(f'MEKIV: {mekiv_mse:.4f}')

    dataframe = pd.read_csv(get_pandas_csv(rho, merror))
    dataframe = dataframe.append({'Run': run, 'Method': 'KIV-M', 'MSE': kiv_M_mse}, ignore_index=True)
    dataframe = dataframe.append({'Run': run, 'Method': 'KIV-MN', 'MSE': kiv_MN_mse}, ignore_index=True)
    dataframe = dataframe.append({'Run': run, 'Method': 'MEKIV', 'MSE': mekiv_mse}, ignore_index=True)
    dataframe = dataframe.append({'Run': run, 'Method': 'KIV-Oracle', 'MSE': kiv_Oracle_mse}, ignore_index=True)

    dataframe.to_csv(get_pandas_csv(rho, merror), index=False)

for rho in RHOS:
    for merror in MERRORS:
        csv_name = get_pandas_csv(rho, merror)
        try:
            dataframe = pd.read_csv(csv_name)
        except:
            dataframe = pd.DataFrame(columns=['Run', 'Method', 'MSE'])
            dataframe.to_csv(csv_name, index=False)
        
        runs = dataframe['Run'].unique()
        next_run = 0 if len(runs) == 0 else max(runs) + 1

        for run in range(next_run, NO_RUNS):
            do_run(rho, merror, run)







