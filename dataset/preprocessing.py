from logging import Logger
import pandas as pd
from datetime import datetime


class OutlierPreprocessor(object):

    def __init__(self, quantile_tol: float, filter_col: str = "quantity_sold"):
        """

        @param quantile_tol: (float) in [0,1]. quantile tolerance (ie, 0.95). All values smaller than this quantile
            are removed
        @param filter_col: (str) the column name, of a pandas dataframe, along which to filter
        """
        self.quantile_tol = quantile_tol
        self.filter_col = filter_col

    def __call__(self, obs: pd.DataFrame, *args, **kwargs):

        disp_quant = obs[["display_id", self.filter_col]].groupby("display_id").quantile(
            self.quantile_tol
        ).reset_index()
        disp_quant.rename(columns={self.filter_col: "upper_quant"}, inplace=True)

        output = pd.merge(obs, disp_quant, on="display_id", how='left')
        output = output[output[self.filter_col] <= output["upper_quant"]]

        return output

class SalesOnlyPreprocessor(object):

    def __init__(self):
        pass

    def __call__(self, obs: pd.DataFrame, *args, **kwargs):
        output = obs[obs["memo"] == "SA"]

        return output

class BadStorePreprocessor(object):
    def __init__(self, bad_store_ids=[]):
        self.bad_store_ids = bad_store_ids
        pass

    def __call__(self, obs: pd.DataFrame, *args, **kwargs):
        output = obs[~obs['store_id'].isin(self.bad_store_ids)]
        return output


class FacingsPreprocessor(object):
    def __init__(self):
        pass

    def __call__(self, obs: pd.DataFrame, *args, **kwargs):
        obs=obs.fillna({'num_product_facings': 0})
        
        obs.loc[obs['previous_post_scan_num_facings'].notnull() & obs['num_product_facings'] > obs['previous_post_scan_num_facings'], 'num_product_facings'] = obs['previous_post_scan_num_facings']
        
        obs.loc[obs['previous_post_scan_num_facings'].isnull(), "previous_post_scan_num_facings"] = obs['num_product_facings']

        return obs

class NullTimeDeltaValuePreprocessor(object):

    def __init__(self):
        pass

    def __call__(self, obs: pd.DataFrame, *args, **kwargs):
        filtered = obs[obs['timedelta'].notna()]
        filtered = filtered[filtered['timedelta'] != 0]

        return filtered

class MinProductSamples(object):
    def __init__(self, min_product_samples):
        self.min_product_samples = min_product_samples
        pass

    def __call__(self, obs: pd.DataFrame, *args, **kwargs):
        filtered = obs
        counts_list = filtered.groupby(["product_id"])["product_id"].count().reset_index(name='counts')
        filter_product_ids = counts_list[counts_list["counts"] < self.min_product_samples]["product_id"].tolist()
        filtered = filtered[~filtered["product_id"].isin(filter_product_ids)]

        return filtered

class MinimumSamplesPreprocessor(object):

    def __init__(self, required_samples=10):
        self.required_samples = required_samples

    def __call__(self, obs: pd.DataFrame, *args, **kwargs):
        disp_groups = obs[["display_id", "display_group"]].groupby(["display_id", "display_group"]).sum().reset_index()
        display_counts = disp_groups.groupby("display_id").count()
        display_counts.rename(columns={"display_group": "display_group_count"}, inplace=True)
        output = pd.merge(obs, display_counts, on="display_id", how='left')
        output = output[output["display_group_count"] >= self.required_samples]
        return output


class CompetitiveProductPreprocessor(object):

    def __init__(self, store_dist_dict: dict, logger: Logger = None):
        self.store_dist_dict = store_dist_dict
        self.logger = logger

    def __call__(self, obs: pd.DataFrame, *args, **kwargs):
        index_list = []
        for index, row in obs.iterrows():
            product_id = row["product_id"]
            store_id = row["store_id"]
            if product_id in self.store_dist_dict["company_products"]:
                index_list.append(True)
                continue
            
            if store_id not in self.store_dist_dict.keys():
                if self.logger:
                    self.logger.warning(f"Missing store distribution data for store: {store_id}")
                index_list.append(False)
                continue

            if product_id in self.store_dist_dict[store_id]:
                index_list.append(True)
                continue
            
            index_list.append(False)
        
        obs = obs[index_list]
        return obs


class TimeDeltaPreprocessor(object):

    """
    Filter observations with small timedeltas since last scan
        - input: minimum time delta in hours
        - data: time delta is in days
    """

    timedelta_key = "timedelta"

    def __init__(self, min_delta_hours: int):
        """

        :param min_delta_hours: minimum timedelta in hours
        """
        self.min_delta_hours = min_delta_hours
        self.min_delta_days = min_delta_hours / 24.0

    def __call__(self, obs: pd.DataFrame, *args, **kwargs):
        return obs[obs[self.timedelta_key] > self.min_delta_days]