import random
from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import ImageFilter
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from tqdm.auto import tqdm


class AddGaussianNoise(object):
    def __init__(self, mean: float, std: float):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(self.mean, self.std)


class ZeroPad2D(object):
    def __init__(self, h1: int, h2: int, w1: int, w2: int):
        self.h1 = h1
        self.h2 = h2
        self.w1 = w1
        self.w2 = w2

    def __call__(self, tensor: torch.Tensor):
        pad = (self.w1, self.w2, self.h1, self.h2)
        return F.pad(tensor, pad)

    def __repr__(self):
        return self.__class__.__name__ + "(H={0},{1}, W={2}{3})".format(self.h1, self.h2, self.w1, self.w2)


class TileChannels2d(object):
    def __init__(self, repeats: int):
        self.repeats = repeats

    def __call__(self, tensor: torch.Tensor):
        return tensor.tile((self.repeats, 1, 1))

    def __repr__(self):
        return self.__class__.__name__ + "(repeats={0})".format(self.repeats)


class GaussianBlur(object):
    """
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Borrowed from MoCo implementation

    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class FixedRandomRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)


class Denormalize(object):
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.demean = [-m / s for m, s in zip(mean, std)]
        self.std = std
        self.destd = [1 / s for s in std]
        self.inplace = inplace

    def __call__(self, tensor):
        tensor = TF.normalize(tensor, self.demean, self.destd, self.inplace)
        # clamp to get rid of numerical errors
        return torch.clamp(tensor, 0.0, 1.0)


def concat_torchvision_transforms(eval: bool, aug: dict):
    # assuming transform is given tensor inputs
    trans = []

    trans.append(T.ConvertImageDtype(torch.float32))

    # not ok for qpi
    # if aug["resize"]:
    #     trans.append(T.Resize(aug["resize"]))

    # not ok for qpi
    # if aug["randcrop"] and aug["scale"] and not eval:
    #     trans.append(T.RandomResizedCrop(size=aug["randcrop"], scale=aug["scale"]))

    # not ok for qpi
    # if aug["randcrop"] and eval:
    #     trans.append(T.CenterCrop(aug["randcrop"]))

    # ok for qpi
    if aug["flip"] and not eval:
        trans.append(T.RandomHorizontalFlip(p=0.5))
        trans.append(T.RandomVerticalFlip(p=0.5))

    # not ok for qpi
    # if aug["jitter_d"] and not eval:
    #     trans.append(
    #         T.RandomApply(
    #             [
    #                 T.ColorJitter(
    #                     0.8 * aug["jitter_d"], 0.8 * aug["jitter_d"], 0.8 * aug["jitter_d"], 0.2 * aug["jitter_d"]
    #                 )
    #             ],
    #             p=aug["jitter_p"],
    #         )
    #     )

    # ok for qpi?
    if aug["gaussian_blur"] and not eval:
        trans.append(T.RandomApply([GaussianBlur([0.1, 2.0])], p=aug["gaussian_blur"]))

    # ok for qpi
    if aug["rotation"] and not eval:
        trans.append(FixedRandomRotation(angles=[0, 90, 180, 270]))

    # not ok for qpi
    # if aug["grayscale"]:
    #     trans.append(T.Grayscale())
    #     trans.append(T.Normalize(mean=aug["bw_mean"], std=aug["bw_std"]))
    # elif aug["mean"]:
    #     trans.append(T.Normalize(mean=aug["mean"], std=aug["std"]))

    return trans


def simclr_transform(opt, eval=False):
    aug = {
        "resize": None,
        "randcrop": opt.image_size,
        "scale": opt.scale,
        "flip": True,
        "jitter_d": opt.rgb_jitter_d,
        "jitter_p": opt.rgb_jitter_p,
        "grayscale": False,
        "gaussian_blur": opt.rgb_gaussian_blur_p,
        "rotation": True,
        "contrast": opt.rgb_contrast,
        "contrast_p": opt.rgb_contrast_p,
        "grid_distort": opt.rgb_grid_distort_p,
        "grid_shuffle": opt.rgb_grid_shuffle_p,
        "mean": None,
        "std": None,
        "bw_mean": None,
        "bw_std": None,
        # "mean": [0.4914, 0.4822, 0.4465],  # values for train+unsupervised combined
        # "std": [0.2023, 0.1994, 0.2010],
        # "bw_mean": [0.4120],  # values for train+unsupervised combined
        # "bw_std": [0.2570],
    }

    return T.Compose(concat_torchvision_transforms(eval, aug))


def reindex_for_ood(
    targets: List[str],
    ood: List[int],
) -> List[str]:
    ind_targets = [y for i, y in enumerate(targets) if i not in ood]
    ood_targets = [y for i, y in enumerate(targets) if i in ood]
    permuted_targets = ind_targets + ood_targets
    return permuted_targets


def take_splits(
    trn_set: Dataset,
    val_set: Dataset | None,
    tst_set: Dataset,
    ind_labels: List[str],
    ood_labels: List[str],
) -> Tuple[Dataset, Dataset, Dataset, Dataset]:
    # reference indices of targets
    permuted_idx = ind_labels + ood_labels
    # if no validation set give, create one
    if val_set is None:
        trn_set, val_set = random_split(trn_set, [0.8, 0.2], torch.manual_seed(42))
    print("Performing ind/ood split")
    # train set data
    trn_od_idx = []
    trn_id_idx = []
    for idx, (_, target, *_) in enumerate(tqdm(trn_set)):  # type: ignore
        dest = trn_od_idx if permuted_idx[target] in ood_labels else trn_id_idx
        dest.append(idx)
    print("Train - OK")
    # validation set data
    val_od_idx = []
    val_id_idx = []
    for idx, (_, target, *_) in enumerate(tqdm(val_set)):  # type: ignore
        dest = val_od_idx if permuted_idx[target] in ood_labels else val_id_idx
        dest.append(idx)
    print("Val - OK")
    # test set data
    tst_od_idx = []
    tst_id_idx = []
    for idx, (_, target, *_) in enumerate(tqdm(tst_set)):  # type: ignore
        dest = tst_od_idx if permuted_idx[target] in ood_labels else tst_id_idx
        dest.append(idx)
    print("Test - OK")
    print("Performed ind/ood split")

    trn = Subset(trn_set, trn_id_idx)
    val = Subset(val_set, val_id_idx)
    tst = Subset(tst_set, tst_id_idx)
    ood = ConcatDataset([Subset(trn_set, trn_od_idx), Subset(val_set, val_od_idx), Subset(tst_set, tst_od_idx)])

    print(len(trn), len(val), len(tst), len(ood))
    return trn, val, tst, ood
