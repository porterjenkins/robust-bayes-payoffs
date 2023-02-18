import pymc as pm


class RBPModelBase(object):

    def __init__(self):
        self.model = None
        self.samples = None


    def build(self, data):
        # implemented by derived class
        pass

    def fit(self):
        # implemented by derived class
        pass

    def predict(self):
        # implemented by derived class
        pass

    def viz_graph(self):
        # implemented by derived class
        pass





class RBPHierachicalProductSegment(RBPModelBase):

    """
    Robust Bayesian Payoff estimates for split sim. Input sim is expected to be filtered to a single product,
        segment pair
    """

    def __init__(self):
        super(RBPHierachicalProductSegment, self).__init__()

    def build(self, ):

        with pm.Model(coords=coords) as model:
            stores = pm.MutableData("store_idx", store_idx, dims="obs_id")

            # global
            mu_global = pm.TruncatedNormal("mu_global", mu=5.0, sigma=5.0, lower=0.0)
            sig = pm.Exponential("sigma", 2.0)

            # priors
            mu = pm.TruncatedNormal("mu_store", mu=mu_global, sigma=sig, lower=0.0, dims="store")
            alpha = pm.Gamma("alpha", 2.0, 2.0)

            y = pm.NegativeBinomial("y", mu=mu[stores], alpha=alpha, observed=payoff, dims="obs_id")


class RBPHierachicalProduct(RBPModelBase):

    """
    Robust Bayesian Payoff estimates for joint sim. Input sim is expected to be filtered to a single product,
        across multiple segments
    """

    def __init__(self):
        super(RBPHierachicalProduct, self).__init__()
