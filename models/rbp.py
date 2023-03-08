import pymc as pm
import arviz as az
import yaml

from dataset.rbp_dataset import RBPBaseDataset



class RBPModelBase(object):

    payoff_name = "payoff"
    # define variables in terms of level, loc and scale
    vars = {}

    def __init__(self, cfg: str):
        cfg = self._read_yaml(cfg)
        self.prior = cfg["prior"]
        self.n = cfg["train"]["n_samples"]
        self.tune = cfg["train"]["n_tune"]
        # uncertainty penalty (mu - lambda*sigma)
        self.lam = cfg['ranking']["lambda"]
        self.model = None
        self.samples = None
        self.idata = None

    def _read_yaml(self, fpath):
        with open(fpath, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg
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
        return pm.model_to_graphviz(self.model)

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

    def __init__(self, cfg):
        super(RBPHierachicalProductSegment, self).__init__(cfg)

    def predict(self):
        if self.samples is not None:
            return self.samples[self.lower_loc].flatten()
        else:
            raise Exception("Calling predict() before fit() is not implemented.")

    def get_ranking_stat(self):
        """
        Calculate uncertainty-penalized Expected Payoff per Facing (EPF)
        :param lam: lambda parameter, which determines strength of uncertainty penalty
        :return: (float) EPF score
        """
        posterior = self.predict()
        mu = posterior.mean()
        sig = posterior.std()

        epf = mu - self.lam*sig

        return epf

    def build(self, dataset: RBPBaseDataset):
        """
        :param dataset: (RBPBaseDataset) Input dataset
        :return:
        """

        idx, coords = dataset.get_coords()
        payoff = dataset.get_payoff()
        feats, names = dataset.get_features()

        # TODO: enable m features. this assumes a single feature in 0th positio
        x_facings = feats[names[0]].values


        with pm.Model(coords=coords) as model:
            stores = pm.MutableData("store_idx", idx["store"], dims="obs_id")

            # global priors
            mu_global = pm.TruncatedNormal(
                self.global_loc,
                mu=self.prior[self.global_loc]['mu'],
                sigma=self.prior[self.global_loc]['sigma'],
                lower=0.0
            )

            sig = pm.TruncatedNormal(
                self.global_scale,
                mu=self.prior[self.global_scale]['mu'],
                sigma=self.prior[self.global_scale]['sigma'],
                lower=0.0
            )


            # lower priors
            mu = pm.TruncatedNormal(
                self.lower_loc,
                mu=mu_global,
                sigma=sig,
                lower=0.0,
                dims="store"
            )

            alpha = pm.Exponential(
                self.lower_scale,
                lam=self.prior[self.lower_scale]['lam']
            )

            e_y = x_facings*mu[stores]

            y = pm.NegativeBinomial(
                self.payoff_name,
                mu=e_y,
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