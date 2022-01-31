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

for i in range(12):
    res = scipy.stats.probplot(popreturn.iloc[:, i].dropna(), dist=scipy.stats.nct,
    sparams=(fits[i, 0], fits[i, 1], fits[i, 2]), plot=axs[i])
    axs[i].set_title(str(popreturn.columns[i] + " 's returns QQ-plot"))
plt.style.use('ggplot')
plt.show()


# Drawing random sample from best fitting T distribution
b = scipy.stats.nct.rvs(df=fits[0, 0], nc=fits[0, 1], scale=fits[0, 2], size=1000)


# Bootstrap parameter estimation
means = []
means1 = []
l = len(popreturn['jeppe'].dropna())
l1 = len(popreturn['sp'].dropna())

for i in range(100000):
    sample = np.random.choice(popreturn['jeppe'].dropna(), size=l, replace=True)
    sample1 = np.random.choice(popreturn['sp'].dropna(), size=l1, replace=True)
    means.append(np.mean(sample))
    means1.append(np.mean(sample1))

diff = np.array(means) - np.array(means1)
print(round(np.sum(diff < 0) / len(diff)*100, 2), '%')

sns.set(style="darkgrid")
fig = sns.kdeplot(diff, shade=True, color="r")
plt.vlines(0, 0, 60)
plt.show()
