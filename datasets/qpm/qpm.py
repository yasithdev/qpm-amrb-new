import glob

import numpy as np
from torch.utils.data import Dataset


class bacteria_dataset(Dataset):
    """
    A standard dataset class to get the bacteria dataset

    Args:
        data_dir   : data directory which contains data hierarchy
        type_      : whether dataloader is train/val/test
        transform  : torchvision.transforms
        target_transform  : torchvision.transforms
        label_type : There are multiple types of classification in bacteria dataset
                     therefore, specify which label you need as follows:
                        | label_type              | Description
                        |------------------------ |---------------
                        | 'strain' (default)      | Strain (0-20)
                        | 'antibiotic_resistant'  | Non wild type (1) / Wild type (0)
                        | 'gram_strain'           | Gram Positive (1) / Gram Negative (0)
                        | 'species'               | Species (0-4)
        balance_data    : If true, dataset will be balanced by the minimum class count (default: False)
    """

    def __init__(
        self,
        data_dir: str,
        type_: str,
        transform=None,
        target_transform=None,
        label_type="strain",
        filter_labels=[],
        filter_mode: str = "exclude",
        balance_data=False,
        strainwise_split=False,
        strains: list[int] = list(range(21)),
    ):
        # validation
        assert type_ in ["train", "val", "test"]
        assert filter_mode in ["include", "exclude"]

        # params
        self.transform = transform
        self.target_transform = target_transform
        self.label_type = label_type
        self.filter_labels = filter_labels
        self.filter_mode = filter_mode
        self.type_ = type_
        self.strains = strains
        print(f"Dataset type {type_} label type: {label_type}")

        ### Extract directories of all files to a dictionary (key: class (strain), value: list of files)
        dirs = {}
        if strainwise_split:
            all_files = [f"{data_dir}/QPM_np_v2/{i}.npy" for i in self.strains]
        else:
            all_files = [f"{data_dir}/QPM_np/{self.type_}/{i}.npy" for i in self.strains]
        for x in all_files:
            # read strain class, encoded in folder name (x.split('/')[-2][:-4])
            strain = int(x.split("/")[-1][:-4])
            dirs[strain] = np.load(x)  # NOTE removed mmap_mode as its slow when shuffle=True

        ## Get the class with minimum count
        min_class_count = 1000000000

        # if dataset needs to be balanced in terms of count per each strain
        if balance_data:
            for i in self.strains:
                count = dirs[i].shape[0]
                if count < min_class_count:
                    min_class_count = count
            print(" - Min class count: ", min_class_count)

        self.images = []
        self.targets = []
        for i in self.strains:  # iterate through all classes
            if self.__must_filter(i):
                continue
            if balance_data:
                count = min_class_count
            else:
                count = dirs[i].shape[0]
            count = int(count)

            # NOTE! to test, only use a small portion of data (e.g. 10% => int(count*0.1) )
            if strainwise_split == False:
                self.images.extend(dirs[i][0:count, ..., None])
                self.targets.extend([i] * count)
            elif type_ == "train":
                frac = int(count * 0.8)
                self.images.extend(dirs[i][0:frac, ..., None])
                self.targets.extend([i] * frac)
            elif type_ == "val":
                frac = int(count * 0.8)
                self.images.extend(dirs[i][frac:count, ..., None])
                self.targets.extend([i] * (count - frac))
            elif type_ == "test":
                self.images.extend(dirs[i][0:count, ..., None])
                self.targets.extend([i] * count)
            else:
                raise ValueError(type_)

        print(f"Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __must_filter(self, cls) -> bool:
        cond1 = self.filter_mode == "exclude"
        cond2 = cls in self.filter_labels
        return cond1 == cond2

    def __getitem__(self, idx):
        image, orig = self.images[idx], self.targets[idx]

        if self.transform:
            image = self.transform(image)

        target = orig
        if self.target_transform:
            target = self.target_transform(orig)

        return image, target, orig
