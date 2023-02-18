import pandas as pd

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

    def __init__(self, group, data):
        super(RBPProductDataset, self).__init__(group, data)

    @classmethod
    def split(cls, df: pd.DataFrame):
        """
        Return one RPBDataset for each product
        :param df: Input pd.DataFrame
        :return: List[RBPProductDataset] a list of datasets
        """
        datasets = []
        for k, dta in df.groupby(cls.product_key):
            rbf_dta = RBPProductDataset(",".join(k), dta)
            datasets.append(rbf_dta)

        return datasets


class RBPProductSegmentDataset(RBPBaseDataset):

    def __init__(self, group, data):
        super(RBPProductSegmentDataset, self).__init__(group, data)

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
