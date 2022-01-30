import pandas as pd
import numpy as np
import scipy.stats

popreturn = pd.read_csv('popreturn')

popreturn.reset_index()

dates = pd.date_range(start='2013-07', end='2021-11', freq='M')

popreturn['date'] = dates
popreturn.drop('Unnamed: 0', axis=1, inplace=True)
popreturn.set_index('date', inplace=True)
print(popreturn.index.values)
print(popreturn.mean().sort_values(ascending=False))

fits = np.empty(shape=(21, 3))

for i in range(len(popreturn.columns)):
    out = scipy.stats.distributions.t.fit(popreturn.iloc[:, i].dropna())
    fits[i, ] = out

print(fits)

fits
