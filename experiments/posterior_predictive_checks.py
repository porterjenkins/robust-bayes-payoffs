import yaml

from dataset.rbp_dataset import RBPBaseDataset
from sim.sim_cfg import SimCfg

SEGMENT_SPLIT = True
fpath = '../sim/test-sim.csv'
cfg_path = "../sim/sim_cfg.yaml"
cfg = SimCfg(cfg_path)


datasets = RBPBaseDataset.get_data(fpath, split_seg=SEGMENT_SPLIT)

for d in datasets:
    print(d)
