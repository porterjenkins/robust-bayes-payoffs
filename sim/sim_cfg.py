import yaml
import pandas as pd

class SimCfg(object):

    def __init__(self, cfg_dict_or_path):

        if isinstance(cfg_dict_or_path, str):
            cfg_dict = self._read_yaml(cfg_dict_or_path)
        else:
            cfg_dict = cfg_dict_or_path

        self.glob = cfg_dict["global"]
        self.stores = cfg_dict["stores"]
        self.df = self._stores_to_df(self.stores)

    def _read_yaml(self, fpath):
        with open(fpath, "r") as f:
            cfg = yaml.safe_load(f)
        return cfg

    @staticmethod
    def _stores_to_df(store_dta):
        dta = []
        params = []
        cols = ['store', 'segment', 'product']
        for store, store_dta in store_dta.items():
            seg = store_dta["segment"]
            for p, p_dta in store_dta["products"].items():
                dta.append(
                    [store, seg, p] + list(p_dta.values())
                )
                params = list(p_dta.keys())

        dta = pd.DataFrame(dta, columns=cols + params)

        return dta

    def get_ground_truth_by_seg(self):
        return self.df.groupby(["product","segment"]).mean(numeric_only=True)

    def get_ground_truth_by_store(self):
        return self.df.groupby(["product","store"]).mean(numeric_only=True)

    def get_ground_truth_by_seg_store(self):
        return self.df.groupby(["product","segment", "store"]).mean(numeric_only=True)
