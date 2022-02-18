import pandas as pd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

popreturn = pd.read_csv('popreturn2.csv')

popreturn.reset_index()

dates = pd.date_range(start='2013-07', end='2022-02', freq='M')

popreturn['date'] = dates
popreturn.drop('Unnamed: 0', axis=1, inplace=True)
popreturn.set_index('date', inplace=True)

# 1.) Estimate mean monthly return per investor minus two times standard error
mu = {}

for name in popreturn.columns:
     mu[name] = popreturn[name].mean() - 2 * popreturn[name].std()/np.sqrt(len(popreturn[name].dropna()))

mu = pd.DataFrame.from_dict(mu, orient='index', columns=['exp_return'])
mu['var'] = popreturn.var()

# 2.) Estimate optimal allocation with convex optimization
mu = mu.loc[['jeppe', 'reinhardt', 'harry', 'christian', 'libor', 'alderique', 'guillaime', 'heloise'], 'exp_return']
popreturn = popreturn[['jeppe', 'reinhardt', 'harry', 'christian', 'libor', 'alderique', 'guillaime', 'heloise']]


def solve_problem(mu = mu, popreturn = popreturn, risk_pref = 0.1):
     mean_stock = mu.values
     cov_stock = popreturn.cov().values

     x = cp.Variable(len(mean_stock))

     stock_return = mean_stock @ x
     stock_risk = cp.quad_form(x, cov_stock)

     objective = cp.Maximize(stock_return - risk_pref * stock_risk)
     constraints = [x >= 0, cp.sum(x) == 1]
     prob = cp.Problem(objective=objective, constraints=constraints)
     return prob.solve(), x.value


# 3.) Plot optimal portfolio allocation for each risk preference
steps = np.linspace(0.01, 2, 100)
x_vals = np.zeros((steps.shape[0], 8))
profit = np.zeros(steps.shape[0])
for i, r in enumerate(steps):
     p, xs = solve_problem(mu, popreturn, risk_pref= r)
     x_vals[i, :] = xs
     profit[i] = p

plt.figure(figsize=(12, 4))
tickers = ["Jeppe", "Reinhardt", "Harry", "Christian", "Libor", "Alderique", "Guillaime", "Heloise"]
for idx, stock in enumerate(tickers):
    plt.plot(steps, x_vals[:, idx], label=stock)
plt.xlabel("risk avoidance")
plt.ylabel("proportion of investment")
plt.legend()
plt.show()

