from models.rbp import RBPModelBase
from dataset.rbp_dataset import RBPBaseDataset

class NaiveEPFBaseline(RBPModelBase):

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

    def __init__(self, cfg=None):
        super(NaiveEPFBaseline, self).__init__(cfg)
        self.dta = None

    def _read_yaml(self, fpath):
        cfg = {
            "prior": {},
            "train": {
                "n_samples": -1,
                "n_tune": -1
            }

        }
        return cfg

    def build(self, dataset: RBPBaseDataset):
        self.dta = dataset

    def fit(self):
        df = self.dta.data
        df['y'] = df[self.dta.payoff_key] / df[self.dta.facings_key]
        self.samples = {
            self.global_loc: df['y'].mean(),
            self.lower_loc: df[['store', 'y']].groupby("store").mean().values.reshape((1, 1, -1)),
            self.global_scale: -1,
            self.lower_scale: -1
        }


    def predict(self):
        pass

    def viz_graph(self):
        pass

    def plot(self):
        pass