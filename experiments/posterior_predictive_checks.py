import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))


from dataset.rbp_dataset import RBPBaseDataset
from models.rbp import RBPHierachicalProduct, RBPHierachicalProductSegment
from sim.sim_cfg import SimCfg

from models.rbp import RBPHierachicalProduct

SEGMENT_SPLIT = True
fpath = '../sim/test-sim.csv'
cfg_path = "../sim/sim_cfg.yaml"
cfg = SimCfg(cfg_path)


datasets = RBPBaseDataset.get_data(fpath, split_seg=SEGMENT_SPLIT)
model = RBPHierachicalProductSegment()

for d in datasets:
    print(d)
    model.build(d)
    model.fit()

