import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

RHOS = {0.25, 0.5, 0.9}

plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "Helvetica"
})

def get_pandas_csv(rho: float) -> str:
    return f'./results/demand_{rho}.csv'

for rho in RHOS:
    csv = get_pandas_csv(rho)
    df = pd.read_csv(csv)

    names = {
        'Adaptive': 'KIV',
        'DFIV': 'DFIV',
        'Bochner': 'BochnerKIV (Affine)',
        'OOS Bochner': 'BochnerKIV-OOS (Affine)',
        'Bochner (2LMP)': 'BochnerKIV (MLP)',
        'OOS Bochner (2LMP)': 'BochnerKIV-OOS (MLP)'
    }

    data = []
    for method in names.keys():
        data.append(
            df[df['Method'] == method]['MSE'].values
        )

    plt.figure(figsize=(12,6))
    # Box and whisker plot
    ax = sns.boxplot(
        data=data,
        orient='v',
    )
    ax.set_xlabel(r'Method')
    ax.set_ylabel(r'Out of sample $\log_{10}(\mathrm{MSE})$')
    plt.xticks(
        list(range(len(names))),
        names.values(),
    )
    plt.title('Demand design, $\\rho = ' + str(rho) + '$')
    plt.savefig(f'./results/demand_{rho}.png')
