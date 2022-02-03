import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

popreturn = pd.read_csv('popreturn')

popreturn.reset_index()

dates = pd.date_range(start='2013-07', end='2021-11', freq='M')

popreturn['date'] = dates
popreturn.drop('Unnamed: 0', axis=1, inplace=True)
popreturn.set_index('date', inplace=True)


# T distribution fits
fits = np.empty(shape=(21, 3))

for i in range(len(popreturn.columns)):
    out = scipy.stats.distributions.t.fit(popreturn.iloc[:, i].dropna())
    fits[i, ] = out

# QQplots:
fig, axs = plt.subplots(3, 4)

axs = axs.ravel()

for i in np.arange(8, 20):
    res = scipy.stats.probplot(popreturn.iloc[:, i].dropna(), dist=scipy.stats.nct,
    sparams=(fits[i, 0], fits[i, 1], fits[i, 2]), plot=axs[(i-8)])
    axs[(i-8)].set_title(str(popreturn.columns[i] + " 's returns QQ-plot"))
plt.style.use('ggplot')
plt.show()


# Drawing random sample from best fitting T distribution
b = scipy.stats.nct.rvs(df=fits[0, 0], nc=fits[0, 1], scale=fits[0, 2], size=1000)


# Bootstrap parameter estimation
traders = ['jeppe', 'christian', 'harry', 'guillaime', 'libor', 'alderique', 'heloise']
num_replicates = 100000
stats = np.empty(shape=(num_replicates, len(traders)))


for count, trader in enumerate(traders):
    means = []
    l = len(popreturn[trader].loc[lambda x: x < 0.25].dropna())
    diff = popreturn[trader].loc[lambda x: x < 0.25].dropna() - popreturn['sp'].dropna()[-l::]

    for i in range(num_replicates):
        sample = np.random.choice(diff, size=l, replace=True)
        means.append(np.mean(sample))

    print(trader, 'Percent negative values:', round(np.sum(np.array(means) < 0) / len(means)*100, 2), '%')
    print(trader, '99% and 95% confidence interval estimates:', np.quantile(means, [0.005, 0.025, 0.5, 0.975, 0.995]))

    stats[:, count] = list(means)


stats_df = pd.DataFrame(stats, columns=[traders])
stats_df = pd.melt(stats_df, var_name='trader')

sns.violinplot(x='trader', y='value', data=stats_df)
plt.show()
