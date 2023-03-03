import pymc as pm
import arviz as az

from dataset.rbp_dataset import RBPBaseDataset

class RBPModelBase(object):

    payoff_name = "payoff"
    # define variables in terms of level, loc and scale
    vars = {}

    def __init__(self, n_samples: int = 1000, n_tune: int = 1000):
        self.n = n_samples
        self.tune = n_tune
        self.model = None
        self.samples = None
        self.idata = None

    def build(self, dataset: RBPBaseDataset):
        # implemented by derived class
        pass

    def fit(self):
        # implemented by derived class
        with self.model:
            self.idata = pm.sample(draws=self.n, tune=self.tune)

        self.samples = {}
        for rv in self.vars:
            self.samples[rv] = self.idata["posterior"][rv].values


    def _get_vars(self, model: pm.Model):
        rvs = model.model.basic_RVs
        var_names = []
        for rv in rvs:
            var_names.append(rv.name)
        return var_names

    def predict(self):
        # implemented by derived class
        pass

    def viz_graph(self):
        # implemented by derived class
        pass

    def plot(self):
        az.plot_posterior(self.idata, show=True)






class RBPHierachicalProductSegment(RBPModelBase):

    """
    Robust Bayesian Payoff estimates for data split across segments. Input data is expected to be filtered to a single product,
        segment pair
    """

    global_loc = "mu_segment"
    global_scale = "sigma"
    lower_loc = "mu_store"
    lower_scale = "alpha_store"

    vars = [
        global_loc,
        global_scale,
        lower_loc,
        lower_scale
    ]

    def __init__(self):
        super(RBPHierachicalProductSegment, self).__init__()

    def build(self, dataset: RBPBaseDataset):
        """
        :param dataset: (RBPBaseDataset) Input dataset
        :return:
        """

        idx, coords = dataset.get_coords()
        payoff = dataset.get_payoff()

        with pm.Model(coords=coords) as model:
            stores = pm.MutableData("store_idx", idx["store"], dims="obs_id")

            # global priors
            mu_global = pm.TruncatedNormal(
                self.global_loc,
                mu=1.0,
                sigma=5.0,
                lower=0.0
            )
            sig = pm.Exponential(self.global_scale, 5.0)

            # lower priors
            mu = pm.TruncatedNormal(
                self.lower_loc,
                mu=mu_global,
                sigma=sig,
                lower=0.0,
                dims="store"
            )
            alpha = pm.Gamma(self.lower_scale, 4.0, 4.0)

            y = pm.NegativeBinomial(
                self.payoff_name,
                mu=mu[stores],
                alpha=alpha,
                observed=payoff,
                dims="obs_id"
            )

        self.model = model


class RBPHierachicalProduct(RBPModelBase):

    """
    Robust Bayesian Payoff estimates for joint data. Input data is expected to be filtered to a single product,
        across multiple segments
    """

    def __init__(self):
        super(RBPHierachicalProduct, self).__init__()

    def build(self, dataset: RBPBaseDataset):
        pass