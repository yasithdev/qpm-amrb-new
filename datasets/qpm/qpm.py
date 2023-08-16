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

species_mapping_dict = {
    0: 0,
    1: 1,
    2: 2,
    3: 4,
    4: 2,
    5: 2,
    6: 2,
    7: 3,
    8: 4,
    9: 2,
    10: 2,
    11: 2,
    12: 3,
    13: 3,
    14: 3,
    15: 3,
    16: 0,
    17: 0,
    18: 0,
    19: 0,
    20: 0,
}


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
                        | 'class' (default)       | Strain (0-20)
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
        label_type="class",
        balance_data=False,
    ):
        # validation
        assert type_ in ["train", "val", "test"]
        # params
        self.transform = transform
        self.target_transform = target_transform
        self.label_type = label_type
        self.type_ = type_
        all_dirs = sorted(
            glob.glob(f"{data_dir}/QPM/{self.type_}/*/*"),
            key=lambda x: int(x.split("/")[-1][:-4]),
        )

        print(f"Dataset type {type_} label type: {label_type}", end=" -> ")

        ### Extract directories of all files to a dictionary (key: class (strain), value: list of files)
        dirs = {}
        print(dirs)

        for i, x in enumerate(all_dirs):
            class_ = int(
                x.split("/")[-2]
            )  # read strain class, encoded in folder name (x.split('/')[-2])

            if class_ in dirs.keys():
                dirs[class_].append(x)
            else:
                dirs[class_] = [x]

        img_dirs_filtered = []

        ## Get the class with minimum count
        min_class_count = 1000000000

        if (
            balance_data
        ):  # if dataset needs to be balanced in terms of count per each class (strain)
            for i in range(0, 21):
                count = len(dirs[i])
                if count < min_class_count:
                    min_class_count = count
            print(" - Min class count: ", min_class_count)

        for i in range(0, 21):  # iterate through all classes
            if balance_data:
                count = min_class_count
            else:
                count = len(dirs[i])

            img_dirs_filtered.append(
                dirs[i][: int(count)]
            )  # NOTE! to test, only use a small portion of data (e.g. 10% => int(count*0.1) )

        self.img_dirs = [
            item for sublist in img_dirs_filtered for item in sublist
        ]  # flatten list

        print(f"Loaded {len(self.img_dirs)} images")

    def __len__(self):
        return len(self.img_dirs)

    def __getclass_(self, meta_data, label_type):
        if label_type == "class":
            return meta_data[0]

        elif label_type == "antibiotic_resistant":
            """
            Dataset wild_type equals to class 1
            antibiotic_resistance is when class is => not wild_type
            """
            antibiotic_resistance = int(not (meta_data[1]))
            return antibiotic_resistance

        elif label_type == "gram_strain":
            return meta_data[2]

        elif label_type == "species":
            return species_mapping_dict[meta_data[0]]  # map class to species

        else:
            raise Exception("Invalid label type")

    def __getitem__(self, idx):
        image, metadata = np.load(self.img_dirs[idx], allow_pickle=True)

        label = self.__getclass_(metadata, self.label_type)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
