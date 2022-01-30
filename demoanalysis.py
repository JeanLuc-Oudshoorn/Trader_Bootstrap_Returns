import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt


popreturn = pd.read_csv('popreturn')

popreturn.reset_index()

dates = pd.date_range(start='2013-07', end='2021-11', freq='M')

popreturn['date'] = dates
popreturn.drop('Unnamed: 0', axis=1, inplace=True)
popreturn.set_index('date', inplace=True)


## T distribution fits
fits = np.empty(shape=(21, 3))

for i in range(len(popreturn.columns)):
    out = scipy.stats.distributions.t.fit(popreturn.iloc[:, i].dropna())
    fits[i, ] = out


## QQplots:
fig, axs = plt.subplots(3, 4)

axs = axs.ravel()

for i in range(12):
    res = scipy.stats.probplot(popreturn.iloc[:,i].dropna(), dist=scipy.stats.nct,
                           sparams=(fits[i, 0], fits[i, 1], fits[i, 2]), plot=axs[i])
    axs[i].set_title(str(popreturn.columns[i] + " 's returns QQ-plot"))

plt.show()

## Drawing random sample
b = scipy.stats.nct.rvs(df=fits[0, 0], nc=fits[0, 1], scale=fits[0, 2], size=1000)
