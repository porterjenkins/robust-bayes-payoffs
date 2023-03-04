from typing import List, Union
import os

import pandas as pd
import yaml
import numpy as np


class NegBinomProduct(object):

    def __init__(self, name: str, theta: float, alpha: float):
        self.name = name
        self.theta = theta
        self.alpha = alpha

    @staticmethod
    def change_params(mu, alpha):
        n = alpha
        p = alpha / (mu + alpha)
        return n, p

    def gen(self, size: int, x: np.ndarray):
        mu = self.theta * x
        n, p = self.change_params(mu, self.alpha)
        y = np.random.negative_binomial(n, p, size)
        return y

class Store(object):
    def __init__(self, segment: int, name: str, n_samples: int, products: List[NegBinomProduct]):
        self.segment = segment
        self.name = name
        self.n_samples = n_samples
        self.products = products
        self.n_products = len(products)

class SalesDataGenerator(object):

    def __init__(self, name: str, n_seg: int, stores: List[Store], data_dir: str):
        self.name = name
        self.data_dir = data_dir
        self.n_seg = n_seg
        self.stores = stores
        self.fpath = os.path.join(data_dir, name + ".csv")

    @classmethod
    def build_from_cfg(cls, cfg_or_dict: Union[str, dict]):

        if isinstance(cfg_or_dict, str):
            with open(cfg_or_dict, 'r') as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = cfg_or_dict

        stores = []
        seg = set()
        for store, vals in cfg['stores'].items():
            seg.add(vals['segment'])
            products = []
            for p, p_data in vals['products'].items():
                products.append(
                    NegBinomProduct(
                        name=p,
                        theta=p_data['mu'],
                        alpha=p_data['alpha']
                    )
                )

            stores.append(
                Store(
                    segment=vals['segment'],
                    name=store,
                    n_samples=vals['n_samples'],
                    products=products
                )
            )

        sim = SalesDataGenerator(
            name = cfg['global']['name'],
            n_seg=len(seg),
            stores=stores,
            data_dir=cfg['global']['data_dir']
        )
        return sim

    def main(self):

        output = {
            "store": [],
            "segment": [],
            "product": [],
            "time": [],
            "sales": [],
            "facings": []
        }

        for s in self.stores:
            size = s.n_samples
            for p in s.products:
                x = np.random.randint(1, 5, size=size)
                y = p.gen(size=size, x=x)

                output['sales'] += y.tolist()
                output['product'] += [p.name]*size
                output['store'] += [s.name]*size
                output['segment'] += [s.segment]*size
                output['time'] += list(range(size))
                output['facings'] += x.tolist()

        output = pd.DataFrame(output)
        output.to_csv(self.fpath, index=False)
        return output



if __name__ == "__main__":
    sim = SalesDataGenerator.build_from_cfg("./sim_cfg.yaml")
    sim.main()

    """p = NegBinomProduct('coke', 2.0, 2.0)
    x = p.gen(size=100)


    import matplotlib.pyplot as plt
    plt.hist(x)
    plt.show()"""