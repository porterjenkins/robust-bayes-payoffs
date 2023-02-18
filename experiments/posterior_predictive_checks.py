import yaml

from dataset.rbp_dataset import RBPBaseDataset

SEGMENT_SPLIT = True
fpath = '../sim/test-sim.csv'
cfg_path = "../sim/sim_cfg.yaml"
with open(cfg_path, "r") as f:
    cfg = yaml.safe_load(f)


datasets = RBPBaseDataset.get_data(fpath, split_seg=SEGMENT_SPLIT)

for d in datasets:
    print(d)
