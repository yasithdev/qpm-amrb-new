import glob

import numpy as np
from torch.utils.data import Dataset

patient_to_binary_mapping = [1,0,0,0,0,1,1,1,1,0]

class rbc_dataset(Dataset):
    """
    A standard dataset class to get the RBC dataset

    Args:
        data_dir   : data directory which contains data hierarchy
        type_      : whether dataloader is train/val/test
        transform  : torchvision.transforms
        target_transform  : torchvision.transforms
        label_type : only type of label we can return is healthy (0) or sick (1)
    """

    def __init__(
        self,
        data_dir: str,
        type_: str,
        transform=None,
        target_transform=None,
        label_type="class",
        filter_labels=[],
        filter_mode: str = "exclude",
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
        all_files = glob.glob(f"{data_dir}/{self.type_}/*.npy")
        
        for x in all_files:
            # read patient id, embedded in filename (e.g. "0.npy")
            patient = int(x.split("/")[-1].split(".")[0])
            data = np.load(x)
            
            dirs[patient] = data

        self.images = []
        self.targets = []
        for i in range(0, len(dirs.keys())):  # iterate through patients
            if self.__must_filter(i):
                continue
            
            count = dirs[i].shape[0]

            # NOTE! to test, only use a small portion of data (e.g. 10% => int(count*0.1) )
            self.images.extend(dirs[i][: int(count), ..., None])
            self.targets.extend([i] * int(count))

        print(f"Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __must_filter(self, i) -> bool:
        cond1 = self.filter_mode == "exclude"
        cond2 = i in self.filter_labels
        return cond1 == cond2

    def __getitem__(self, idx):
        image, patient = self.images[idx], self.targets[idx]
        # target should be the pateint id
        # class is whether the patient is healthy or not
        target = patient_to_binary_mapping[patient]
        
        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            target = self.target_transform(target)

        return image, target, patient
