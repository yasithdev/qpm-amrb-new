import logging
import sys
from typing import List

import einops
import numpy as np
import umap
import umap.plot
from matplotlib import cm
from matplotlib import pyplot as plt


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
    embedding = reducer.transform(x)

    logging.info("Plotting UMAP Projection")
    fig = plt.figure(figsize=(10, 7), dpi=150)
    plt.title(title)
    ax = fig.add_subplot(111, projection="3d")
    color = iter(cm.Spectral(np.linspace(0, 1, len(labels))))
    for p, label in enumerate(labels):
        idx = np.where(y == p)
        ax1 = embedding[idx, 0]
        ax2 = embedding[idx, 1]
        ax3 = embedding[idx, 2]
        c = next(color)
        ax.scatter(ax1, ax2, ax3, c=c, label=label, s=5, alpha=0.5)
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
    labels = np.array([labels[i] for i in y])

    logging.info("Generating UMAP Plot")
    fig = plt.figure(figsize=(10, 7), dpi=150)
    plt.title(title)
    ax = fig.add_subplot(111)
    umap.plot.points(reducer, labels=labels, ax=ax)

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
    x = einops.rearrange(x, "b c h w -> b (c h w)")
    y = np.argmax(y, axis=1)
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
            ax = axs[y_i, cX[y_i]]
            ax.imshow(X.squeeze())
            axs[y_i, 0].set_ylabel(labels[y_i])
            cX[y_i] += 1
        if cX == [nY] * nX:
            break
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
