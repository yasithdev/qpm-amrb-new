import numpy as np
import pandas as pd
import seaborn as sns
import glob
import json
from matplotlib import pyplot as plt
import scipy.io as sio
from scipy.spatial.distance import pdist
from tqdm import tqdm
from itertools import product
from functools import partial

if __name__ == "__main__":
    plt.rcParams["legend.loc"] = "upper right"
    files = sorted(glob.glob("assets/results/*.json"))
    labels = sorted(glob.glob("assets/stats/*.json"))
    
    data = []
    for fp in tqdm(files):
        with open(fp, "r") as f:
            d = json.load(f)
            data.append(d)
    
    grps = []
    for fp in tqdm(labels):
        with open(fp, "r") as f:
            d = json.load(f)
            grps.append(d)
            
    def replace_ood_vals(df):
        return df.replace('0:1:2:3:4', 'A')\
    .replace('5:6:7:8:9', 'B')\
    .replace('0:1', 'A')\
    .replace('0:2:3', 'B')\
    .replace('1:4', 'A')\
    .replace('2:3', 'B')\
    .replace('0', 'A')\
    .replace('1', 'B')\
    .replace("", "Full")
    
    idx_dfm = ["model_name", "dataset_name", "ood"]
    idx_dfl = ["dataset_name", "ood"]
    model_names = ["resnet50_vicreg_ce", "resnet_ce_mse", "resnet_mse"]
    dfl = replace_ood_vals(pd.json_normalize(grps, max_level=1)).drop_duplicates(idx_dfl, keep='last').set_index(idx_dfl).sort_index().reset_index().fillna(np.nan)
    dfm = replace_ood_vals(pd.json_normalize(data, max_level=1)).drop_duplicates(idx_dfm, keep='last').set_index(idx_dfm).sort_index().reset_index().fillna(np.nan)
    dfm.columns = dfm.columns.str.replace('.', '_')
    dfm = dfm[dfm.model_name.isin(model_names)]

    # save result as mat files
    sio.savemat("assets/results_summary.mat", {"result": dfm.dropna(axis='columns', how='all').to_records()})
    sio.savemat("assets/labels_summary.mat", {"label": dfl.dropna(axis='columns', how='all').to_records()})
