import pandas as pd
import numpy as np
import h5py
import gzip
import pickle

def mat_to_df(m):
    mdtype = m.dtype
    ndata = {}
    for n in mdtype.names:
        data = []
        for x in m[n][0]:
            try:
                value = x[0].item()
            except:
                if len(x) == 1:
                    value = x[0]
                else:
                    value = None
            data.append(value)
        ndata[n] = data
    return pd.DataFrame({k: v for k, v in ndata.items() if k != "index"}, ndata["index"])

def parse_h5(path, key) -> np.ndarray:
    with h5py.File(path) as f:
        v = np.array(f.get(key))
    return v.reshape((v.shape[0], -1))

def parse_h5_v2(path, keys: list[str]) -> dict:
    data = {}
    with h5py.File(path) as f:
        for key in keys:
            v = np.array(f.get(key))
            data[key] = v.reshape((v.shape[0], -1))
    return data

def parse_stats(path):
    with gzip.open(path, "rb") as f:
        _stats = pickle.load(f)
    stats = {}
    for k, v in _stats.items():
        if not isinstance(v, dict): continue
        colmap = {"val": "val_data", "ind": "ind_data", "ood": "ood_data"}
        for n, c in colmap.items():
            if c not in v: continue
            stats[f"{k}_{n}"] = list(v[c])
    return pd.Series(stats)
