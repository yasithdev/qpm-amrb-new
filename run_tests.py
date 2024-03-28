import json
import os
from argparse import ArgumentParser
from itertools import product
from typing import Any, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.io import loadmat
from scipy.spatial.distance import pdist
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from util import mat_to_df, parse_h5, parse_stats


def permutation_test_2samp(sa: np.ndarray, sb: np.ndarray, stat: Callable, tail: str, n: int):
    """
    Given two samples (sa, sb), compute the test statistic (stat) and approximate its p value through n permutations

    Args:
    sa: sample A
    sb: sample B
    stat: test statistic. invoked as f(sa, sb)
    n: number of permutations

    """
    svec = np.concatenate([sa, sb], axis=0)  # vectorize samples
    partition = [len(sa)]  # define partitions
    rng = np.random.RandomState(42)

    obs = stat(*np.split(svec, partition))  # compute statistic for true observation
    stats = []
    for i in range(n):  # compute statistic across n permutations
        sp = rng.permutation(svec)
        stats.append(stat(*np.split(sp, partition)))

    # compute p value with
    stats = np.array(stats)
    if tail == "left":
        c = (stats <= obs).sum()
    elif tail == "right":
        c = (stats >= obs).sum()
    elif tail == "both":
        aobs = abs(obs)
        cl, cr = (stats <= -aobs).sum(), (stats >= aobs).sum()
        c = 2 * min(cl, cr)
    p = (c + 1) / (n + 1)  # +1 to count the true observation

    # return statistic, p-value, and the n stats under the null
    return obs, p, stats


def mmd(x: np.ndarray, y: np.ndarray, **kwargs):
    """
    Compute the mean of the difference of means

    """
    return (x.mean(axis=0) - y.mean(axis=0)).mean()


def auc(*s: np.ndarray):
    sa, sb = s
    na, nb = len(sa), len(sb)
    S = np.concatenate([sa, sb], axis=0)
    # find the optimal hyperplane which separates sa and sb
    true = np.repeat([0, 1], [na, nb])
    model = LinearRegression().fit(S, true)
    pred = model.predict(S)

    # compute auc of the predictions
    auc = roc_auc_score(true, pred)

    return max(auc, 1 - auc)


def mrpp(*s: np.ndarray, metric: Any = "euclidean", **kwargs):
    """
    Compute the MRPP statistic for the given samples
    Multi-Response Permutation Procedure

    Idea - weighted sum of mean pairwise distance of each group
    """
    sa, sb = s

    # compute weights (wa, wb) and within-group pair counts (pa,pb)
    na, nb = len(sa), len(sb)
    n = na + nb
    wa, wb = na / n, nb / n
    assert wa + wb == 1.0

    # compute mean L2 distance
    da = pdist(sa, metric=metric).mean()
    db = pdist(sb, metric=metric).mean()

    # compute statistic
    δ = da * wa + db * wb

    return δ

def get_metrics(dfm, dfl, model_name, dataset_name, ood, label_names=None, use="metrics"):
    # get ind and ood labels
    rec = dfl[(dfl.dataset_name == dataset_name) & (dfl.ood == ood)].iloc[0]
    val_labels = ind_labels = ood_labels = []
    if rec.val_labels is not None:
        val_labels = rec.val_labels
    if rec.ind_labels is not None:
        ind_labels = rec.ind_labels
    if rec.ood_labels is not None:
        ood_labels = rec.ood_labels

    if label_names is not None:
        if len(label_names) == 1:
            l = label_names[0]
            val_labels = list(map(l.__getitem__, val_labels))
            ind_labels = list(map(l.__getitem__, ind_labels))
            ood_labels = list(map(l.__getitem__, ood_labels))

    if use == "metrics":
        # get old ind and ood metric data
        rec: pd.Series = dfm[(dfm.model_name == model_name) & (dfm.dataset_name == dataset_name) & (dfm.ood == ood)].iloc[0]
        assert type(rec) == pd.Series
        _val_data = rec[rec.index.str.endswith("_val_data")].dropna()
        _val_cols = sorted(_val_data.index.str[:-9])
        _val_data.index = pd.Index(_val_cols)
        _ind_data = rec[rec.index.str.endswith("_ind_data")].dropna()
        _ind_cols = sorted(_ind_data.index.str[:-9])
        _ind_data.index = pd.Index(_ind_cols)
        _ood_data = rec[rec.index.str.endswith("_ood_data")].dropna()
        _ood_cols = sorted(_ood_data.index.str[:-9])
        _ood_data.index = pd.Index(_ood_cols)
        assert _val_cols == _ind_cols == _ood_cols

        df_val = pd.DataFrame({**_val_data.to_dict(), "label": val_labels})
        df_ind = pd.DataFrame({**_ind_data.to_dict(), "label": ind_labels})
        df_ood = pd.DataFrame({**_ood_data.to_dict(), "label": ood_labels})
        metrics = _ind_cols

    elif use == "new_metrics":
        # get new ind and ood metric data
        rec = parse_stats(f"assets/results/{dataset_name}/{model_name}/stats_{ood}.gz")
        assert type(rec) == pd.Series
        _val_data = rec[rec.index.str.endswith("_val")].dropna()
        _val_cols = sorted(_val_data.index.str[:-4])
        _val_data.index = pd.Index(_val_cols)
        _ind_data = rec[rec.index.str.endswith("_ind")].dropna()
        _ind_cols = sorted(_ind_data.index.str[:-4])
        _ind_data.index = pd.Index(_ind_cols)
        _ood_data = rec[rec.index.str.endswith("_ood")].dropna()
        _ood_cols = sorted(_ood_data.index.str[:-4])
        _ood_data.index = pd.Index(_ood_cols)
        assert _val_cols == _ind_cols == _ood_cols

        df_val = pd.DataFrame({**_val_data.to_dict(), "label": val_labels})
        df_ind = pd.DataFrame({**_ind_data.to_dict(), "label": ind_labels})
        df_ood = pd.DataFrame({**_ood_data.to_dict(), "label": ood_labels})
        metrics = _ind_cols
    
    elif use == "embeddings":    
        # load ind and ood embeddings
        h5_val = parse_h5(f"assets/results/{dataset_name}/{model_name}/val_{ood}.h5", "emb")
        h5_ind = parse_h5(f"assets/results/{dataset_name}/{model_name}/ind_{ood}.h5", "emb")
        h5_ood = parse_h5(f"assets/results/{dataset_name}/{model_name}/ood_{ood}.h5", "emb")
    
        df_val = pd.DataFrame({**{idx: col for idx, col in enumerate(h5_val.T)}, "label": val_labels})
        df_ind = pd.DataFrame({**{idx: col for idx, col in enumerate(h5_ind.T)}, "label": ind_labels})
        df_ood = pd.DataFrame({**{idx: col for idx, col in enumerate(h5_ood.T)}, "label": ood_labels})
        assert len(df_val.columns) == len(df_ind.columns) == len(df_ood.columns)
        metrics = list(range(h5_val.shape[1]))

    elif use == "logits":    
        # load ind and ood logits
        h5_val = parse_h5(f"assets/results/{dataset_name}/{model_name}/val_{ood}.h5", "lgt")
        h5_ind = parse_h5(f"assets/results/{dataset_name}/{model_name}/ind_{ood}.h5", "lgt")
        h5_ood = parse_h5(f"assets/results/{dataset_name}/{model_name}/ood_{ood}.h5", "lgt")
    
        df_val = pd.DataFrame({**{idx: col for idx, col in enumerate(h5_val.T)}, "label": val_labels})
        df_ind = pd.DataFrame({**{idx: col for idx, col in enumerate(h5_ind.T)}, "label": ind_labels})
        df_ood = pd.DataFrame({**{idx: col for idx, col in enumerate(h5_ood.T)}, "label": ood_labels})
        assert len(df_val.columns) == len(df_ind.columns) == len(df_ood.columns)
        metrics = list(range(h5_val.shape[1]))

    elif use == "inputs":    
        # load ind and ood embeddings
        h5_val = parse_h5(f"assets/results/{dataset_name}/{model_name}/val_{ood}.h5", "inp")
        h5_ind = parse_h5(f"assets/results/{dataset_name}/{model_name}/ind_{ood}.h5", "inp")
        h5_ood = parse_h5(f"assets/results/{dataset_name}/{model_name}/ood_{ood}.h5", "inp")
    
        df_val = pd.DataFrame({**{idx: col for idx, col in enumerate(h5_val.T)}, "label": val_labels})
        df_ind = pd.DataFrame({**{idx: col for idx, col in enumerate(h5_ind.T)}, "label": ind_labels})
        df_ood = pd.DataFrame({**{idx: col for idx, col in enumerate(h5_ood.T)}, "label": ood_labels})
        metrics = list(range(h5_val.shape[1]))

    else:
        raise ValueError(use)

    return metrics, df_val, df_ind, df_ood


def get_stat_by_name(stat_name):
    if stat_name == "mrpp":
        stat, tail = mrpp, "left"
    elif stat_name == "mmd":
        stat, tail = mmd, "both"
    elif stat_name == "auc":
        stat, tail = auc, "right"
    else:
        raise ValueError(stat_name)
    return stat, tail


def scale(x, mu, std):
    return (x - mu) / std


# possible rescaling = "valset", "groups", "anchors", None
def htest_pairwise(dfm, dfl, model, dataset, ood, stat, sample_size, perms, rescale, label_names=None, use="metrics"):
    desc = f"{model},{dataset}-{ood},{stat}-{rescale},s{sample_size},p{perms}"
    stat_fn, tail = get_stat_by_name(stat)

    # get metric names and values
    metrics, val, ind, ood = get_metrics(dfm, dfl, model, dataset, ood, label_names, use)
    l_val, l_ind, l_ood = sorted(set(val.label)), sorted(set(ind.label)), sorted(set(ood.label))
    l_tst = l_ind + l_ood
    n_val, n_tst, n_ind, n_ood = len(l_val), len(l_tst), len(l_ind), len(l_ood)
    s_val, s_tst = val, pd.concat([ind, ood], axis=0, ignore_index=True)

    if rescale == "valset":
        # compute normalizing hyperparameters from validation data
        val_arr = val[metrics].astype(float).to_numpy()
        mu_T, std_T = mu(val_arr), std(val_arr)

    # variables to store all-to-ind comparisons of the testset
    δ = np.zeros((n_tst, n_ind))
    P = np.zeros((n_tst, n_ind))
    N = np.zeros((n_tst, n_ind, perms))

    # compute (obs, p) for all label-label pairs
    rng = np.random.RandomState(42)
    tests = list(product(range(n_tst), range(n_ind)))
    for i, j in tqdm(tests, desc=desc):
        GA = s_tst.loc[s_tst.label == l_tst[i], metrics].astype(float).to_numpy()
        GB = s_tst.loc[s_tst.label == l_tst[j], metrics].astype(float).to_numpy()
        # if len(GA) < sample_size:
        #     print(f"WARN - class {i} has too few samples")
        # if len(GB) < sample_size:
        #     print(f"WARN - class {j} has too few samples")
        # take a fixed sample from each observation
        GA, GB = rng.permutation(GA), rng.permutation(GB)
        sa, sb = GA[:sample_size], GB[-sample_size:]
        if rescale == "valset":
            sa, sb = scale(sa, mu_T, std_T), scale(sb, mu_T, std_T)
        elif rescale == "groups":
            G = np.concatenate([GA, GB], axis=0)
            mu_G, std_G = mu(G), std(G)
            sa, sb = scale(sa, mu_G, std_G), scale(sb, mu_G, std_G)
        elif rescale == "anchors":
            mu_B, std_B = mu(GB), std(GB)
            sa, sb = scale(sa, mu_B, std_B), scale(sb, mu_B, std_B)
        elif rescale is not None:
            raise ValueError(rescale)

        δ[i, j], P[i, j], N[i, j] = permutation_test_2samp(sa, sb, stat_fn, tail, perms)

    # return δ and P as dataframes
    δ = pd.DataFrame(data=δ, index=l_tst, columns=l_ind)
    P = pd.DataFrame(data=P, index=l_tst, columns=l_ind)

    return metrics, δ, P, N


# possible rescaling = "valset", "groups", "anchors", None
def htest_new_sample(dfm, dfl, model, dataset, ood, stat, sample_size, perms, rescale, label_names=None, use="metrics"):
    desc = f"{model},{dataset}-{ood},{stat}-{rescale},s{sample_size},p{perms}"
    stat_fn, tail = get_stat_by_name(stat)

    # get metric names and values
    metrics, val, ind, ood = get_metrics(dfm, dfl, model, dataset, ood, label_names, use)
    l_val, l_ind, l_ood = sorted(set(val.label)), sorted(set(ind.label)), sorted(set(ood.label))
    l_tst = l_ind + l_ood
    n_val, n_tst = len(l_val), len(l_tst)
    s_val, s_tst = val, pd.concat([ind, ood], axis=0, ignore_index=True)

    if rescale == "valset":
        # metrics = list(filter(lambda x: x[0] != "f", metrics)) # ignore fixed point metrics
        # compute normalizing hyperparameters from training data
        val_arr = s_val[metrics].astype(float).to_numpy()
        mu_T, std_T = mu(val_arr), std(val_arr)

    # variables to store all test-train comparisons
    δ = np.zeros((n_tst, n_val))
    P = np.zeros((n_tst, n_val))
    N = np.zeros((n_tst, n_val, perms))

    # compute (obs, p) for all label-label pairs
    rng = np.random.RandomState(42)
    tests = list(product(range(n_tst), range(n_val)))
    for i, j in tqdm(tests, desc=desc):
        GA = s_tst.loc[s_tst.label == l_tst[i], metrics].astype(float).to_numpy()
        GB = s_val.loc[s_val.label == l_val[j], metrics].astype(float).to_numpy()
        # if len(GA) < sample_size:
        #     print(f"WARN - class {i} has too few samples")
        # if len(GB) < sample_size:
        #     print(f"WARN - class {j} has too few samples")
        # take a fixed sample from each observation
        GA, GB = rng.permutation(GA), rng.permutation(GB)
        sa, sb = GA[:sample_size], GB[-sample_size:]
        if rescale == "valset":
            sa, sb = scale(sa, mu_T, std_T), scale(sb, mu_T, std_T)
        elif rescale == "groups":
            G = np.concatenate([GA, GB], axis=0)
            mu_G, std_G = mu(G), std(G)
            sa, sb = scale(sa, mu_G, std_G), scale(sb, mu_G, std_G)
        elif rescale == "anchors":
            mu_B, std_B = mu(GB), std(GB)
            sa, sb = scale(sa, mu_B, std_B), scale(sb, mu_B, std_B)
        elif rescale is not None:
            raise ValueError(rescale)

        δ[i, j], P[i, j], N[i, j] = permutation_test_2samp(sa, sb, stat_fn, tail, perms)

    # return δ and P as dataframes
    δ = pd.DataFrame(data=δ, index=l_tst, columns=l_val)
    P = pd.DataFrame(data=P, index=l_tst, columns=l_val)

    return metrics, δ, P, N


def inference(model, stat, sample_size, perms, basepath, evals, test, rescale, suffix, test_name, use):
    E = len(evals)
    fig, axs = plt.subplots(nrows=2, ncols=E, figsize=(E * 10, 15))
    stats = {}
    for i, (dataset, ood, typ, *label_names) in enumerate(evals):
        m, δ, p, n = test(dfm, dfl, model, dataset, ood, stat, sample_size, perms, rescale, label_names, use)
        axa, axb = axs[0][i], axs[1][i]
        sns.heatmap(δ, annot=True, fmt=".3f", ax=axa)
        sns.heatmap(p, annot=True, fmt=".3f", ax=axb)
        # sns.heatmap(p, annot=True, fmt=".3f", ax=axb, norm=LogNorm(vmin=1 / (perms + 1), vmax=1.0))
        axa.set_title(f"δ_obs: {dataset}-{ood}-{typ}")
        axb.set_title(f"P(δ<=δ_obs|H0): {dataset}-{ood}-{typ}")
        if test_name == "pairwise":
            ylabel, xlabel = "Test Sample", "InD Test Sample"
        elif test_name == "new_sample":
            ylabel, xlabel = "Test Sample", "Val Sample"
        axa.set_ylabel(ylabel)
        axb.set_ylabel(ylabel)
        axa.set_xlabel(xlabel)
        axb.set_xlabel(xlabel)
        os.makedirs(f"{basepath}/stats", exist_ok=True)
        if typ is None:
            dname = f"{dataset}_{ood}"
        else:
            dname = f"{dataset}_{ood}_{typ}"
        np.savez_compressed(f"{basepath}/stats/{model}_{dname}_{use}.npz", δ=δ.to_numpy(), p=p.to_numpy(), n=n)
        stats[dname] = {"δ": δ.to_dict(), "p": p.to_dict()}
    plt.suptitle(f"Test Statistic ({stat}) and P Value\nTest={test_name}, Model={model}, Type={typ}, Use={use}, #={len(m)}")
    plt.tight_layout()
    plt.savefig(f"{basepath}/{model}_{suffix}_{use}.pdf")
    plt.close()
    return stats


if __name__ == "__main__":

    # parameters
    parser = ArgumentParser()
    parser.add_argument("--rescale", default=None, choices=[None, "valset", "anchors", "groups"])
    parser.add_argument("--stat_name", default="mrpp", choices=["mrpp", "mmd", "auc"])
    parser.add_argument("--use", default="metrics", choices=["metrics", "embeddings", "logits", "inputs"])
    parser.add_argument("--base_path", default="tests_20240327")
    parser.add_argument("--perm_count", default=1500, type=int)
    parser.add_argument("--sample_size", default=50, type=int)
    parser.add_argument("--test", default="toy", type=str, choices=["toy", "dom"])
    parser.add_argument("--hypothesis", default=2, type=int, choices=[1,2])
    ctx = parser.parse_args()
    rescale, stat_name, basepath, perm_count, sample_size, use, test_type, hypothesis = (
        ctx.rescale,
        ctx.stat_name,
        ctx.base_path,
        ctx.perm_count,
        ctx.sample_size,
        ctx.use,
        ctx.test,
        ctx.hypothesis,
    )

    qpm_groups = ["AB", "BS", "EC", "KP", "SA"]
    rbc_groups = ["Healthy", "Sickle"]
    qpm_class_mapping = list(map(int, "012422234222333300000"))
    rbc_class_mapping = list(map(int, "1100111111111111110011100110011000111000011110011"))
    qpm_class_names = [f"{qpm_groups[j]}_{i}" for i, j in enumerate(qpm_class_mapping)]
    rbc_class_names = [f"{rbc_groups[j]}_{i}" for i, j in enumerate(rbc_class_mapping)]
    qpm_group_names = [qpm_groups[j] for j in qpm_class_mapping]
    rbc_group_names = [rbc_groups[j] for j in rbc_class_mapping]
    
    c10_class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    data = {}

    loadmat("assets/labels_summary.mat", data)
    loadmat("assets/results_summary.mat", data)

    idx_dfm = ["model_name", "dataset_name", "ood"]
    idx_dfl = ["dataset_name", "ood"]

    dfl = mat_to_df(data["label"])
    dfm = mat_to_df(data["result"])
    df_flat = dfm.melt(id_vars=idx_dfm, var_name="metric").copy().dropna()

    autoencoders = ["resnet_mse"]
    classifiers = ["resnet50_vicreg_ce"]
    classifiers_ae = ["resnet_ce_mse", "resnet_edl_mse"]
    flows = ["flow_ss_vcr_mse"]

    df_autoencoders = df_flat[df_flat.model_name.isin(autoencoders)]
    df_classifiers = df_flat[df_flat.model_name.isin(classifiers)]
    df_classifiers_ae = df_flat[df_flat.model_name.isin(classifiers_ae)]
    df_flows = df_flat[df_flat.model_name.isin(flows)]

    mu = lambda x: np.mean(x, axis=0, keepdims=True)
    std = lambda x: np.mean(x, axis=0, keepdims=True)
    tag = f"{stat_name}_{rescale}_p{perm_count}_s{sample_size}"
    fp_h = f"{basepath}/{tag}_h{hypothesis}"

    if hypothesis == 1:
        htest, htest_name = htest_pairwise, "pairwise"
    elif hypothesis == 2:
        htest, htest_name = htest_new_sample, "new_sample"

    if test_type == "toy":
        # class mappings
        mc = c10_class_names
        # toy tests
        htest_stats = {}
        runs = {
            "resnet_mse": [
                ("CIFAR10", "A", None, mc),
                ("CIFAR10", "B", None, mc),
                ("MNIST", "A", None),
                ("MNIST", "B", None)
            ],
            "resnet50_vicreg_ce": [
                ("CIFAR10", "A", None, mc),
                ("CIFAR10", "B", None, mc),
                ("MNIST", "A", None),
                ("MNIST", "B", None),
            ],
            "resnet_ce_mse": [
                ("CIFAR10", "A", None, mc),
                ("CIFAR10", "B", None, mc),
                ("MNIST", "A", None),
                ("MNIST", "B", None),
            ],
        }
        for model, evals in runs.items():
            # skip impossible cases
            if model == "resnet_mse" and use == "logits":
                continue
            # logging, setup
            print(f"\nToy: model={model}, stat={stat_name}_{rescale}, perms={perm_count}, sample={sample_size}")
            args = dict(model=model, stat=stat_name, sample_size=sample_size, perms=perm_count, rescale=rescale, suffix="toy", use=use)
            # hypothesis test
            htest_stats[model] = inference(basepath=fp_h, evals=evals, test=htest, test_name=htest_name, **args)
        with open(f"{fp_h}/stats_toy_{use}.json", "w") as f:
            json.dump(htest_stats, f)

    if test_type == "dom":
        # class mappings
        mq_1 = qpm_class_names
        mq_2 = qpm_group_names
        mr_1 = rbc_class_names
        mr_2 = rbc_group_names
        # domain tests
        htest_stats = {}
        runs = {
            "resnet_mse": [
                ("QPM_species", "A", "strain-level", mq_1),
                ("QPM_species", "B", "strain-level", mq_1),
                ("QPM_species", "A", "species-level", mq_2),
                ("QPM_species", "B", "species-level", mq_2),
                ("QPM2_species", "A", "strain-level", mq_1),
                ("QPM2_species", "B", "strain-level", mq_1),
                ("QPM2_species", "A", "species-level", mq_2),
                ("QPM2_species", "B", "species-level", mq_2),
            ],
            "resnet50_vicreg_ce": [
                ("QPM_species", "A", "strain-level", *mq_1),
                ("QPM_species", "B", "strain-level", *mq_1),
                ("QPM_species", "A", "species-level", *mq_2),
                ("QPM_species", "B", "species-level", *mq_2),
                ("QPM2_species", "A", "strain-level", *mq_1),
                ("QPM2_species", "B", "strain-level", *mq_1),
                ("QPM2_species", "A", "species-level", *mq_2),
                ("QPM2_species", "B", "species-level", *mq_2),
            ],
            "resnet_ce_mse": [
                ("QPM_species", "A", "strain-level", mq_1),
                ("QPM_species", "B", "strain-level", mq_1),
                ("QPM_species", "A", "species-level", mq_2),
                ("QPM_species", "B", "species-level", mq_2),
                ("QPM2_species", "A", "strain-level", mq_1),
                ("QPM2_species", "B", "strain-level", mq_1),
                ("QPM2_species", "A", "species-level", mq_2),
                ("QPM2_species", "B", "species-level", mq_2)
            ],
        }
        for model, evals in runs.items():
            # skip impossible cases
            if model == "resnet_mse" and use == "logits":
                continue
            # logging, setup
            print(f"\nDom: model={model}, stat={stat_name}_{rescale}, perms={perm_count}, sample={sample_size}")
            args = dict(model=model, stat=stat_name, sample_size=sample_size, perms=perm_count, rescale=rescale, suffix="dom", use=use)
            # hypothesis tests
            htest_stats[model] = inference(basepath=fp_h, evals=evals, test=htest, test_name=htest_name, **args)
        with open(f"{fp_h}/stats_dom_{use}.json", "w") as f:
            json.dump(htest_stats, f)
