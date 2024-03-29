import logging
import sys
from typing import List

import einops
import numpy as np
import umap
import umap.plot
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def gen_umap_3d(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: str,
    labels: List[str],
):
    """
    Create a 3D UMAP plot

    """
    logging.info("Fitting UMAP")
    reducer = umap.UMAP(random_state=42, n_components=3)
    reducer.fit(x)

    logging.info("Generating UMAP embeddings")
    embedding = np.array(reducer.transform(x))

    logging.info("Plotting UMAP Projection")
    mpl.rcParams['lines.markersize'] = 2.25
    fig = plt.figure(figsize=(10, 7), dpi=150)
    plt.title(title)
    ax = fig.add_subplot(111, projection="3d")
    cmap = cm.get_cmap("Spectral")
    L = len(labels) - 1
    for l, label in enumerate(labels):
        c = cmap(l / L, 0.5)
        cond = np.where(y == l)
        ax1 = embedding[cond, 0]
        ax2 = embedding[cond, 1]
        ax3 = embedding[cond, 2]
        ax.scatter(ax1, ax2, ax3, c=c, label=label)
    ax.legend()

    logging.info("Saving plot")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def gen_umap_2d(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    out_path: str,
    labels: List[str],
):
    """
    Create a 2D UMAP plot

    """

    logging.info("Fitting UMAP")
    reducer = umap.UMAP(random_state=42, n_components=2)
    reducer.fit(x)

    logging.info("Generating UMAP Plot")
    fig = plt.figure(figsize=(10, 7), dpi=150)
    plt.title(title)
    ax = fig.add_subplot(111)
    umap.plot.points(reducer, labels=np.take(labels, y), ax=ax)

    logging.info("Saving plot")
    plt.savefig(out_path)
    plt.close()


def gen_umap(
    x: np.ndarray,
    y: np.ndarray,
    out_path: str,
    title: str,
    labels: List[str],
    projection="2d",
):
    x = einops.rearrange(x, "b ... -> b (...)")
    if projection == "2d":
        gen_umap_2d(x, y, title, out_path, labels)
    elif projection == "3d":
        gen_umap_3d(x, y, title, out_path, labels)
    else:
        sys.exit(1)


def plot_samples(
    x: np.ndarray,
    y: np.ndarray,
    out_path: str,
    labels: List[str],
    cols=15,
):
    nX, nY = cols, len(labels)
    cX = [0] * nY
    S = 2
    fig, axs = plt.subplots(
        nY, nX, sharey="all", sharex="all", figsize=(nX * S, nY * S), dpi=100
    )
    for X, Y in zip(x, y):
        y_i = np.argmax(Y)
        if cX[y_i] < nX:
            ax = axs[y_i][cX[y_i]]
            ax.imshow(X.squeeze())
            axs[y_i][0].set_ylabel(labels[y_i])
            cX[y_i] += 1
        if cX == [nY] * nX:
            break
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
