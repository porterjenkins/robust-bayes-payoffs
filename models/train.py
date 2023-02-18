import pymc as pm
import pandas as pd
import arviz as az
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

fpath = '../sim/test-sim.csv'
df = pd.read_csv(fpath)
print(df.head())

le = LabelEncoder()
payoff = df['sales'].values

prod_names = df['product'].values
prod_idx = le.fit_transform(prod_names)
print(prod_idx)

with pm.Model() as model:
    #prod_ = pm.MutableData("prod_idx", floor_measure, dims="prod_id")

    """alpha = pm.Normal("alpha", 0, sigma=10)
    beta = pm.Normal("beta", mu=0, sigma=10)
    sigma = pm.Exponential("sigma", 5)

    theta = alpha + beta * floor_ind"""

    # priors
    mu = pm.TruncatedNormal("mu_upper", mu=1.5, sigma=1.0, lower=0.0)
    alpha = pm.Gamma("alpha", 2.0, 2.0)
    y = pm.NegativeBinomial("y", mu=mu, alpha=alpha, observed=payoff)


with model:
# draw 1000 posterior samples
    idata = pm.sample(1000, tune=1000, cores=1)
pm.model_to_graphviz(model)
az.plot_posterior(idata, show=True)