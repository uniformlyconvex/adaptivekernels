import sys
import os

current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import torch
import pandas as pd

from src.designs.demand import Demand
from src.models.kiv_adaptive_ridge import KIVAdaptiveRidge
from src.models.bochner_kiv.bochner import BochnerKIV
from src.models.bochner_kiv.bochner_sampled import BochnerKIVSampled
from src.models.oos_bochner_kiv.oos_bochner import OOSBochnerKIV
from src.models.oos_bochner_kiv.oos_bochner_sampled import OOSBochnerKIVSampled
from src.models.dfiv import DFIV

RHOS = {0.25, 0.5, 0.9}
NO_RUNS = 20
SEARCH_SPACE = torch.exp(torch.linspace(-8,3,50))

X_NET = torch.nn.Sequential(
    torch.nn.Linear(3, 3, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 3),
    torch.nn.ReLU()
)

Z_NET = torch.nn.Sequential(
    torch.nn.Linear(3, 3, bias=True),
    torch.nn.ReLU(),
    torch.nn.Linear(3, 3),
    torch.nn.ReLU()
)

def get_pandas_csv(rho: float) -> str:
    return f'./results/demand_{rho}.csv'

def do_run(rho: float, run: int) -> None:
    design = Demand(rho=rho)
    data = design.generate_KIV_data(no_points=1000)
    test_data = design.generate_test_data()

    adaptive = KIVAdaptiveRidge(data, lmbda=SEARCH_SPACE, xi=SEARCH_SPACE)
    adaptive_preds = adaptive.predict(test_data.X)
    adaptive_mse = test_data.evaluate_preds(adaptive_preds)

    bochner = BochnerKIV(data, lmbda_search_space=SEARCH_SPACE, xi_search_space=SEARCH_SPACE, test_data=test_data, target=adaptive_mse)
    bochner_preds = bochner.predict(test_data.X)
    bochner_mse = test_data.evaluate_preds(bochner_preds)

    oos_bochner = OOSBochnerKIV(data, lmbda_search_space=SEARCH_SPACE, xi_search_space=SEARCH_SPACE, test_data=test_data, target=adaptive_mse)
    oos_bochner_preds = oos_bochner.predict(test_data.X)
    oos_bochner_mse = test_data.evaluate_preds(oos_bochner_preds)

    bochner_2lmp = BochnerKIVSampled(data, lmbda_search_space=SEARCH_SPACE, xi_search_space=SEARCH_SPACE, test_data=test_data, target=adaptive_mse, X_net=X_NET, Z_net=Z_NET)
    bochner_2lmp_preds = bochner_2lmp.predict(test_data.X)
    bochner_2lmp_mse = test_data.evaluate_preds(bochner_2lmp_preds)

    oos_bochner_2lmp = OOSBochnerKIVSampled(data, lmbda_search_space=SEARCH_SPACE, xi_search_space=SEARCH_SPACE, test_data=test_data, target=adaptive_mse, X_net=X_NET, Z_net=Z_NET)
    oos_bochner_2lmp_preds = oos_bochner_2lmp.predict(test_data.X)
    oos_bochner_2lmp_mse = test_data.evaluate_preds(oos_bochner_2lmp_preds)

    dfiv = DFIV(data, lmbda_search_space=SEARCH_SPACE, xi_search_space=SEARCH_SPACE, test_data=test_data, target=adaptive_mse)
    dfiv_preds = dfiv.predict(test_data.X)
    dfiv_mse = test_data.evaluate_preds(dfiv_preds)

    df = pd.read_csv(get_pandas_csv(rho))
    df = df.append({'Run': run, 'Method': 'Adaptive', 'MSE': adaptive_mse}, ignore_index=True)
    df = df.append({'Run': run, 'Method': 'Bochner', 'MSE': bochner_mse}, ignore_index=True)
    df = df.append({'Run': run, 'Method': 'OOS Bochner', 'MSE': oos_bochner_mse}, ignore_index=True)
    df = df.append({'Run': run, 'Method': 'DFIV', 'MSE': dfiv_mse}, ignore_index=True)
    df = df.append({'Run': run, 'Method': 'Bochner (2LMP)', 'MSE': bochner_2lmp_mse}, ignore_index=True)
    df = df.append({'Run': run, 'Method': 'OOS Bochner (2LMP)', 'MSE': oos_bochner_2lmp_mse}, ignore_index=True)

    dataframe.to_csv(get_pandas_csv(rho), index=False)


for rho in RHOS:
    csv_name = get_pandas_csv(rho)
    try:
        dataframe = pd.read_csv(csv_name)
    except:
        dataframe = pd.DataFrame(columns=['Run', 'Method', 'MSE'])
        dataframe.to_csv(csv_name, index=False)
    
    runs = dataframe['Run'].unique()
    next_run = 0 if len(runs) == 0 else max(runs) + 1

    for run in range(next_run, NO_RUNS):
        do_run(rho, run)