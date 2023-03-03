import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

class RBPBaseDataset(object):

    product_key = "product"
    segment_key = "segment"
    payoff_key = "sales"
    store_key = "store"

    def __init__(self, group: str,  data: pd.DataFrame):
        self.group = group
        self.data = data

    def __str__(self):
        return self.group

    def get_coords(self):
        """
        Retrieve PyMC coords construct: Details dimensionality of groups
        :return: (dict)
        """
        # implemented by derived class
        pass

    def get_payoff(self):
        return self.data[self.payoff_key].values

    @classmethod
    def split(cls, df: pd.DataFrame):
        # implemented by derived class
        pass


    @classmethod
    def read_csv(cls, fpath):
        return pd.read_csv(fpath)

    @classmethod
    def get_data(cls, fpath: str, split_seg: bool = True):
        """

        :param fpath: filepath to csv
        :param split_seg: (bool) whether or not to construct datasets for product-segment pairs
                            - if True, one dataset for each product-segment pair
                            - if False, one dataset for each product
        :return: List[RPTDataset] a list of datasets
        """
        dta = RBPBaseDataset.read_csv(fpath)
        if split_seg:
            dta_list = RBPProductSegmentDataset.split(dta)
        else:
            dta_list = RBPProductDataset.split(dta)

        return dta_list



class RBPProductDataset(RBPBaseDataset):

    """
    Robust Bayesian Payoff dataset for data combined across segments. Data are filtered to a single product,
        across multiple segments
    """

    def __init__(self, group, data):
        super(RBPProductDataset, self).__init__(group, data)

    def get_coords(self):
        """
        Retrieve PyMC coords construct: Details dimensionality of groups
        :return: Tuple(dict, dict): index dictionary, coords dictionary
        """

        """
        coords = {
            #'product': prod_names,
            'store': store_names,
            'segment': seg_names
        }
        """

        # get store indices
        le = LabelEncoder()
        store_idx = le.fit_transform(self.data[self.store_key].values)
        store_names = le.classes_

        # get segment indices
        seg_idx = self.data[self.segment_key].values
        seg_names =self.data[self.segment_key].unique()

        # store to segment
        store_to_seg = self.data[[self.store_key, self.segment_key]].groupby(self.store_key).max()['segment'].values

        idx = {
            "store": store_idx,
            "segment": seg_idx,
            "store_to_seg": store_to_seg
        }

        coords = {
            "store": store_names,
            "segment": seg_names
        }

        return idx, coords

    @classmethod
    def split(cls, df: pd.DataFrame):
        """
        Return one RPBDataset for each product
        :param df: Input pd.DataFrame
        :return: List[RBPProductDataset] a list of datasets
        """
        datasets = []
        for k, dta in df.groupby(cls.product_key):
            rbf_dta = RBPProductDataset(k, dta)
            datasets.append(rbf_dta)

        return datasets


class RBPProductSegmentDataset(RBPBaseDataset):

    """
    Robust Bayesian Payoff dataset split across segments. Data are filtered to a single product,
        segment pair
    """

    def __init__(self, group, data):
        super(RBPProductSegmentDataset, self).__init__(group, data)

    def get_coords(self):
        """
        Retrieve PyMC coords construct: Details dimensionality of groups
        :return: (dict)
        """
        # get store indices
        le = LabelEncoder()
        store_idx = le.fit_transform(self.data['store'].values)
        store_names = le.classes_

        idx = {"store": store_idx}
        coords = {"store": store_names}

        return idx, coords


    @classmethod
    def split(cls, df: pd.DataFrame):
        """
        Return one RPBDataset for each product-segment pair
        :param df: Input pd.DataFrame
        :return: List[RBPProductSegmentDataset] a list of datasets
        """
        datasets = []
        for k, dta in df.groupby([cls.segment_key, cls.product_key]):
            g = [str(x) for x in k]
            g = ",".join(g)
            rbf_dta = RBPProductSegmentDataset(group=g, data=dta)
            datasets.append(rbf_dta)

        return datasets



if __name__ == "__main__":
    datasets = RBPBaseDataset.get_data("../sim/test-sim.csv", split_seg=False)
    print(datasets)
