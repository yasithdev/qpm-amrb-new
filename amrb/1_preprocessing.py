import json
import logging
import os
from typing import Iterable, Tuple

import cv2
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm

from amrb_util import load_config


def get_mat_data(_path: str) -> Iterable[np.ndarray]:
    """
    Read images from a .mat file into a list of images

    :param _path: path to the .mat file
    :return: list of all images in the .mat file
    """

    # Load MAT files
    mat = loadmat(_path)
    # as images could be in multiple sizes, return a list of images
    # === Dataset 1 ===
    if "A1" in mat:
        data = np.transpose(mat.get("A1"), (2, 0, 1))
    # === Dataset 2 ===
    elif "B1" in mat:
        data = mat.get("B1")[0]
    elif "B1_1" in mat:
        data = mat.get("B1_1")[0]
    elif "B1_2" in mat:
        data = mat.get("B1_2")[0]
    else:
        data = []
    return data


def find_contours(
    x: np.ndarray,
    thresh: float = 0.5,
) -> Tuple[int, Tuple[int, int, int, int], np.ndarray]:
    """
    Compute the number of contours, their bounding coordinates, and their bitmask

    First, it normalizes and binarizes the input image.
    Next, it finds contours from the binary image, and generates their convex hulls.
    Following this, it obtains the bounding rectangle of all convex hulls.
    Finally, it returns the number of contours, their (convex hull) bounding rectangle, and their (convex hull) bitmask.

    :param x: an input image
    :param thresh: the binarizing threshold
    :return: number of contours, their (convex hull) bounding rectangle, and their (convex hull) bitmask.

    """

    # take a copy of x for processing
    img = x.copy()

    # find the value range of x
    hi = np.max(img)
    lo = np.min(img)

    # edge case: incorrect phase unwrapping / no data
    if lo < 0 or hi == lo:
        return 0, None, None

    # normalize image to [0,1]
    normalized_img = img / hi

    # binarize image and convert to uint8
    binary_img: np.ndarray
    (_, binary_img) = cv2.threshold(
        src=normalized_img,
        thresh=thresh,
        maxval=1,
        type=cv2.THRESH_BINARY,
    )
    binary_img = binary_img.astype(np.uint8)

    # find contours of the binary image
    contours: np.ndarray
    (contours, _) = cv2.findContours(
        image=binary_img,
        mode=cv2.RETR_EXTERNAL,
        method=cv2.CHAIN_APPROX_NONE,
    )
    num_contours = len(contours)

    # edge case: no contours detected
    if num_contours == 0:
        return 0, None, None

    # get convex hulls from contours
    contour_convex_hulls = list(map(cv2.convexHull, contours))

    # get the bounding rectangle of all convex hulls
    bounding_rect = list(cv2.boundingRect(np.concatenate(contour_convex_hulls, axis=0)))

    # generate a bitmask for the convex hulls
    bitmask: np.ndarray = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(
        image=bitmask,
        contours=contour_convex_hulls,
        contourIdx=-1,
        color=1,
        thickness=cv2.FILLED,
    )

    return num_contours, bounding_rect, bitmask


def main(config: dict):

    dataset_path = os.path.join(config.get("data_dir"), config.get("dataset_name"))
    logging.info(f"(Dataset Path)={dataset_path}")

    experiment_name = "1_preprocessing"
    logging.info(f"(Experiment Name)={experiment_name}")

    experiment_path = os.path.join(
        config.get("experiment_dir"), experiment_name, config.get("dataset_name")
    )
    logging.info(f"(Experiment Path)={experiment_path}")

    info_json_path = os.path.join(dataset_path, "info.json")
    logging.info(f"Reading: {info_json_path}")
    with open(info_json_path, "r") as f:
        dataset_info = json.load(f)

    logging.info(f"Generating Dataset: {config.get('dataset_name')}")

    # LOGIC ============================================================================

    # categorized dataset
    ds_imag_accepted = {}
    ds_mask_accepted = {}
    ds_imag_rejected = {}
    ds_mask_rejected = {}

    ds_stats = {}

    labels = sorted(dataset_info["data"])

    # iterate each label
    for label in labels:

        # statistics
        num_invalid = 0
        num_oversize_acc = 0
        num_oversize_rej = 0
        cnum_vals = []
        bbox_vals = []

        # per-class image buffer
        imag_accepted = []
        imag_rejected = []

        # per-class bitmask buffer
        mask_accepted = []
        mask_rejected = []

        # iterate each file of class
        images = []
        for filepath in dataset_info["data"][label]["files"]:
            images.extend(get_mat_data(os.path.join(dataset_path, filepath)))

        # read mat data
        for imag in tqdm(images, desc=f"Label: {label}", **config.get("tqdm_args")):

            # get contour count, bbox, and mask
            c_num, bounds, mask = find_contours(imag)

            # case - contours detected
            if c_num > 0:

                # crop
                (x, y, w, h) = bounds
                crop_imag: np.ndarray = imag[y : y + h, x : x + w]
                crop_mask: np.ndarray = mask[y : y + h, x : x + w]

                # squarify
                if h > w:  # case - tall crop
                    a = (h - w) // 2
                    b = h - w - a
                    crop_imag = np.pad(crop_imag, ((0, 0), (a, b)))
                    crop_mask = np.pad(crop_mask, ((0, 0), (a, b)))
                elif w > h:  # case - wide crop
                    a = (w - h) // 2
                    b = w - h - a
                    crop_imag = np.pad(crop_imag, ((a, b), (0, 0)))
                    crop_mask = np.pad(crop_mask, ((a, b), (0, 0)))

                img_d = crop_imag.shape[0]

                # collect c_num and img_d statistics
                cnum_vals.append(c_num)
                bbox_vals.append(img_d)

                # adjust image to target size
                if (
                    config.get("img_l") >= img_d
                ):  # case - exact/undersized: zero-pad around the object
                    a = (config.get("img_l") - img_d) // 2
                    b = config.get("img_l") - img_d - a
                    final_mask = np.pad(crop_mask, ((a, b), (a, b)))
                    final_imag = np.pad(crop_imag, ((a, b), (a, b)))

                    # filtering
                    if c_num == 1:  # accept
                        mask_accepted.append(final_mask)
                        imag_accepted.append(
                            final_imag * final_mask
                        )  # masking noise (if any)
                    else:  # reject
                        imag_rejected.append(final_imag)
                        mask_rejected.append(final_mask)

                else:  # case - oversize: ignore
                    if c_num == 1:
                        num_oversize_acc += 1
                    else:
                        num_oversize_rej += 1

            # case - no contours detected
            else:
                num_invalid += 1

        # calculate statistics
        num_accepted = len(imag_accepted)
        num_rejected = len(imag_rejected)

        ds_stats[f"{label}_cnum_0"] = num_invalid
        ds_stats[f"{label}_cnum_Y_size_Y"] = num_accepted
        ds_stats[f"{label}_cnum_Y_size_N"] = num_oversize_acc
        ds_stats[f"{label}_cnum_N_size_Y"] = num_rejected
        ds_stats[f"{label}_cnum_N_size_N"] = num_oversize_rej
        ds_stats[f"{label}_cnum_vals"] = np.array(cnum_vals)
        ds_stats[f"{label}_bbox_vals"] = np.array(bbox_vals)

        # log stats
        logging.info(
            {
                "✔ cnum_y_size_y": num_accepted,
                "x cnum_y_size_n": num_oversize_acc,
                "x cnum_n_size_y": num_rejected,
                "x cnum_n_size_n": num_oversize_rej,
                "x cnum_0": num_invalid,
            }
        )

        # update categorized dataset
        ds_imag_accepted[label] = imag_accepted
        ds_imag_rejected[label] = imag_rejected

        ds_mask_accepted[label] = mask_accepted
        ds_mask_rejected[label] = mask_rejected

    os.makedirs(experiment_path, exist_ok=True)

    accepted_imag_path = os.path.join(experiment_path, "accepted.imag.npz")
    logging.info(f"writing: {accepted_imag_path}")
    np.savez_compressed(accepted_imag_path, **ds_imag_accepted)
    logging.info(f"written: {accepted_imag_path}")

    accepted_mask_path = os.path.join(experiment_path, "accepted.mask.npz")
    logging.info(f"writing: {accepted_mask_path}")
    np.savez_compressed(accepted_mask_path, **ds_mask_accepted)
    logging.info(f"written: {accepted_mask_path}")

    rejected_imag_path = os.path.join(experiment_path, "rejected.imag.npz")
    logging.info(f"writing: {rejected_imag_path}")
    np.savez_compressed(rejected_imag_path, **ds_imag_rejected)
    logging.info(f"written: {rejected_imag_path}")

    rejected_mask_path = os.path.join(experiment_path, "rejected.mask.npz")
    logging.info(f"writing: {rejected_mask_path}")
    np.savez_compressed(rejected_mask_path, **ds_mask_rejected)
    logging.info(f"written: {rejected_mask_path}")

    stats_path = os.path.join(experiment_path, "stats.npz")
    logging.info(f"writing: {stats_path}")
    np.savez_compressed(stats_path, **ds_stats)
    logging.info(f"written: {stats_path}")

    logging.info(f"DONE")


if __name__ == "__main__":

    # initialize the RNG deterministically
    np.random.seed(42)
    config = load_config()
    main(config)
