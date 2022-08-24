from os import path

from ds_util import load_numpy, take_n
from util.vis import plot_samples, gen_umap

if __name__ == "__main__":
    # imports --------------------------
    from ds_info import DS_NAME, EXPERIMENT_DIR, SPLIT_NUM

    # ----------------------------------

    data = load_numpy()
    vis_path = path.join(EXPERIMENT_DIR, "vis", DS_NAME, f"S{SPLIT_NUM}")

    (x_trn, y_trn), (x_val, y_val), (x_tst, y_tst) = data

    # ----------------------------------
    # Preview - Samples
    # ----------------------------------
    print("Preview: Inputs")

    print("* TRN")
    plot_samples(x_trn, y_trn, path.join(vis_path, "preview.trn.pdf"))

    print("* VAL")
    plot_samples(x_val, y_val, path.join(vis_path, "preview.val.pdf"))

    print("* TST")
    plot_samples(x_tst, y_tst, path.join(vis_path, "preview.tst.pdf"))

    # ----------------------------------
    # UMAP - Samples
    # ----------------------------------
    print(f"Selecting 30k items from {DS_NAME}")
    (x_trn, y_trn, n_trn), (x_val, y_val, n_val), (x_tst, y_tst, n_tst) = take_n(
        data, limit=3e4
    )

    print("UMAP: Inputs")

    print("* TRN")
    gen_umap(
        x_trn,
        y_trn,
        f"UMAP - Training Samples ({n_trn})",
        path.join(vis_path, "umap.trn.png"),
    )

    print("* VAL")
    gen_umap(
        x_val,
        y_val,
        f"UMAP - Validation Samples ({n_trn})",
        path.join(vis_path, "umap.val.png"),
    )

    print("* TST")
    gen_umap(
        x_tst,
        y_tst,
        f"UMAP - Testing Samples ({n_trn})",
        path.join(vis_path, "umap.tst.png"),
    )
