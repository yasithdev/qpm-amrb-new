from typing import List, Tuple

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset, random_split
from torchvision.transforms import Compose


class AddGaussianNoise(object):
    def __init__(self, mean: float, std: float):
        self.std = std
        self.mean = mean

    def __call__(self, tensor: torch.Tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + "(mean={0}, std={1})".format(
            self.mean, self.std
        )


class ZeroPad2D(object):
    def __init__(self, h1: int, h2: int, w1: int, w2: int):
        self.h1 = h1
        self.h2 = h2
        self.w1 = w1
        self.w2 = w2

    def __call__(self, tensor: torch.Tensor):
        pad = (self.w1, self.w2, self.h1, self.h2)
        return torch.constant_pad_nd(tensor, pad)

    def __repr__(self):
        return self.__class__.__name__ + "(H={0},{1}, W={2}{3})".format(
            self.h1, self.h2, self.w1, self.w2
        )


class TileChannels2d(object):
    def __init__(self, repeats: int):
        self.repeats = repeats

    def __call__(self, tensor: torch.Tensor):
        return tensor.tile((self.repeats, 1, 1))

    def __repr__(self):
        return self.__class__.__name__ + "(repeats={0})".format(self.repeats)


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
    trn_od_idx, trn_id_idx = [], []
    for idx, (_, target) in enumerate(trn_set):  # type: ignore
        dest = trn_od_idx if permuted_idx[target] in ood_labels else trn_id_idx
        dest.append(idx)
    # validation set data
    val_od_idx, val_id_idx = [], []
    for idx, (_, target) in enumerate(trn_set):  # type: ignore
        dest = val_od_idx if permuted_idx[target] in ood_labels else val_id_idx
        dest.append(idx)
    # test set data
    tst_od_idx, tst_id_idx = [], []
    for idx, (_, target) in enumerate(tst_set):  # type: ignore
        dest = tst_od_idx if permuted_idx[target] in ood_labels else tst_id_idx
        dest.append(idx)
    print("Performed ind/ood split")

    trn = Subset(trn_set, trn_id_idx)
    val = Subset(trn_set, val_id_idx)
    tst = Subset(tst_set, tst_id_idx)
    ood = ConcatDataset([Subset(trn_set, trn_od_idx), Subset(val_set, val_od_idx), Subset(tst_set, tst_od_idx)])

    print(len(trn), len(val), len(tst), len(ood))
    return trn, val, tst, ood
