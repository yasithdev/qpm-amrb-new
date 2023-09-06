import logging
import os
from typing import Dict, List, Literal

import numpy as np


def load_gensim_dict(
    ds_path: str,
) -> Dict[str, Dict[str, float]]:
    gensim_path = os.path.join(ds_path, "gensim_strain.csv")
    gensim = np.genfromtxt(gensim_path, delimiter=",", dtype=str)
    gensim_labels: List[str] = list(gensim[0].astype(str))
    gensim_data = gensim[1:].astype(np.float32)
    gensim_dict = {
        k1: {k2: float(gensim_data[i, j]) for j, k2 in enumerate(gensim_labels)} for i, k1 in enumerate(gensim_labels)
    }
    return gensim_dict


def gensim(
    full_label_map: Dict[str, str],
    uniq_target_labels: List[str],
    cv_k: int,
    gensim_y: np.ndarray,
    N: int,
    mode: Literal[1, 2] = 1,
) -> np.ndarray:
    # split the two gensim measures into two symmetric matrices
    s1 = np.triu(gensim_y) + np.triu(gensim_y, 1).T  # type: ignore
    s2 = np.tril(gensim_y) + np.tril(gensim_y, -1).T  # type: ignore
    sim = (s1 + s2) / 2
    logging.info(f"Gensim Matrix: {sim.shape}")

    # compute the group indices
    group_idxs: Dict[str, List[int]] = {}
    for i, (_, tlabel) in enumerate(full_label_map.items()):
        if tlabel in group_idxs:
            group_idxs[tlabel].append(i)
        else:
            group_idxs[tlabel] = [i]

    assert N == len(group_idxs) - 1, "mismatch in label counts!"

    # target probability vector
    pvec = np.zeros((1, N), dtype=np.float32)

    # generate similarity matrix
    tlabel_k = uniq_target_labels[cv_k]
    for tlabel in group_idxs:
        if tlabel != tlabel_k:
            k = uniq_target_labels.index(tlabel)
            if k > uniq_target_labels.index(tlabel_k):
                k -= 1
            n = 0
            for i in group_idxs[tlabel_k]:
                for j in group_idxs[tlabel]:
                    pvec[0, k] += sim[i, j]
                    n += 1
            pvec[0, k] /= n

    return pvec / pvec.sum()
