# robust-bayes-payoffs

This repo is designed to solve existing problems with Expected Payoff per Facing (EPF) ranking:
1. EPF is not robust to outliers. Extreme outliers from sensor noise or user error skew the EPF estimator and harm bias ranking
2. There is instrinsic uncertainty in the recommendation problem. We need to assign confidence values to product value estimates. 
3. We need a principled way to trade-off store- and cluster-leve information. 


Robust Bayesian Payoff (RBP) estimators can help solve these challenges in the following way:
1. We can put strong priors over the domain of expected sales values and discount outliers
2. The posterior predictive distribution measures naturally measures uncertainty. Additionally, we should sort by uncertainty penalize reward
3. A hierarchical model, cluster-level priors and store-level parameters will naturally trade-off information
