import glob

import numpy as np
from torch.utils.data import Dataset

### Species level mapping
# 0 => Acinetobacter
# 1 => B subtilis
# 2 => E. coli
# 3 => K. pneumoniae
# 4 => S. aureus
# More info => https://ruhsoft-my.sharepoint.com/:p:/g/personal/im_ramith_fyi/EYMDb528EVlClCp2y8nIM8oB9LBZ-lbqEiCXwcAZHX7wew?e=lAROoR
species_mapping = [0, 1, 2, 4, 2, 2, 2, 3, 4, 2, 2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0]


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
        print(f"Dataset type {type_} label type: {label_type}")

        ### Extract directories of all files to a dictionary (key: class (strain), value: list of files)
        dirs = {}
        all_files = glob.glob(f"{data_dir}/QPM_np/{self.type_}/*.npy")
        for x in all_files:
            # read strain class, encoded in folder name (x.split('/')[-2][:-4])
            class_ = int(x.split("/")[-1][:-4])
            dirs[class_] = np.load(x, mmap_mode="c")

        ## Get the class with minimum count
        min_class_count = 1000000000

        # if dataset needs to be balanced in terms of count per each class (strain)
        if balance_data:
            for i in range(0, 21):
                count = dirs[i].shape[0]
                if count < min_class_count:
                    min_class_count = count
            print(" - Min class count: ", min_class_count)

        self.images = []
        self.targets = []
        for i in range(0, 21):  # iterate through all classes
            if self.__must_filter(i):
                continue
            if balance_data:
                count = min_class_count
            else:
                count = dirs[i].shape[0]

            # NOTE! to test, only use a small portion of data (e.g. 10% => int(count*0.1) )
            self.images.extend(dirs[i][: int(count), ..., None])
            self.targets.extend([i] * int(count))

        print(f"Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getclass_(self, strain):
        if self.label_type == "strain":
            return strain
        elif self.label_type == "species":
            return species_mapping[strain]  # map class to species

        else:
            raise Exception("Invalid label type")

    def __must_filter(self, i) -> bool:
        cls = self.__getclass_(i)
        cond1 = self.filter_mode == "exclude"
        cond2 = cls in self.filter_labels
        return cond1 == cond2

    def __getitem__(self, idx):
        image, strain = self.images[idx], self.targets[idx]
        target = self.__getclass_(strain)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target, strain
