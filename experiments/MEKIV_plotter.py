import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

RHOS = {0.25, 0.5, 0.9}
MERRORS = {0.5,1,2}

plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "Helvetica"
})

def get_pandas_csv(rho: float, merror: float) -> str:
    return f'./results/mekiv_{rho}_{merror}.csv'

for rho in RHOS:
    # Subplots of 3 different merrors
    fig, ax = plt.subplots(1,3, figsize=(12,5))
    for i, merror in enumerate(MERRORS):
        csv = get_pandas_csv(rho, merror)
        df = pd.read_csv(csv)

        names = ['KIV-M', 'KIV-MN', 'KIV-Oracle', 'MEKIV']

        data = []
        for method in names:
            data.append(
                df[df['Method'] == method]['MSE'].values
            )

        # Box and whisker plot
        sns.boxplot(
            data=data,
            orient='v',
            ax=ax[i]
        )
        ax[i].set_xticks(
            list(range(len(names))),
            names,
        )
        ax[i].set_title(f'$\sigma = {merror}$')
    # Set figure title and axes labels
    fig.suptitle('$\\rho = ' + str(rho) + '$')
    fig.text(0.5, 0.04, 'Method', ha='center')
    fig.text(0.08, 0.5, 'Out of sample $\log_{10}(\mathrm{MSE})$', va='center', rotation='vertical')
    
    plt.savefig(f'./results/mekiv_{rho}.png')
    