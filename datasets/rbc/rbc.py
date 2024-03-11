import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class rbc_dataset(Dataset):
    """
    A standard dataset class to get the RBC dataset

    Args:
        data_dir   : data directory which contains data hierarchy
        type_      : whether dataloader is train/val/test
        transform  : torchvision.transforms
        target_transform  : torchvision.transforms
        image_type : only type of label we can return is healthy (0) or sick (1)
    """

    def __init__(
        self,
        data_dir: str,
        type_: str,
        transform=None,
        target_transform=None,
        image_type="phase",
        filter_labels=[],
        filter_mode: str = "exclude",
        patientwise_split=False,
        patients: list[int] = list(range(21)),
    ):
        # validation
        assert type_ in ["train", "val", "test"]
        assert filter_mode in ["include", "exclude"]

        # params
        self.transform = transform
        self.target_transform = target_transform
        self.image_type = image_type
        self.filter_labels = filter_labels
        self.filter_mode = filter_mode
        self.type_ = type_
        self.patients = patients
        print(f"RBC Dataset V.3 => Fractional Split OR Patient wise Split")
        print(f"Dataset split type {type_}, image type: {image_type}")

        ### Extract directories of all files to a dictionary (key: class (patient), value: list of files)
        data = {}
        if patientwise_split:
            phase_base = amp_base = f"{data_dir}/RBC_split_patients"
        else:
            phase_base = f"{data_dir}/RBC_all_patients/{self.type_}/phase"
            amp_base = f"{data_dir}/RBC_all_patients/{self.type_}/amp"
              
        for i in self.patients:
            phase_fp = f"{phase_base}/{i}_phase.npy"
            amp_fp = f"{amp_base}/{i}_amp.npy"
            dp = np.load(phase_fp)
            da = np.load(amp_fp)
            assert dp.shape == da.shape
            data[i] = [dp, da]

        self.images_phs = []
        self.images_amp = []
        self.targets = []
        
        for i in self.patients:  # iterate through patients
            if self.__must_filter(i):
                continue
            dp, da = data[i]
            count = int(dp.shape[0])
            
            # NOTE! to test, only use a small portion of data (e.g. 10% => int(count*0.1) )
            if patientwise_split == False:                
                self.images_phs.extend(dp[0:count, ..., None])
                self.images_amp.extend(da[0:count, ..., None])
                self.targets.extend([i] * count)
            elif type_ == "train":
                frac = int(count * 0.8)
                self.images_phs.extend(dp[0:frac, ..., None])
                self.images_amp.extend(da[0:frac, ..., None])
                self.targets.extend([i] * frac)
            elif type_ == "val":
                frac = int(count * 0.8)
                self.images_phs.extend(dp[frac:count, ..., None])
                self.images_amp.extend(da[frac:count, ..., None])
                self.targets.extend([i] * (count - frac))
            elif type_ == "test":
                self.images_phs.extend(dp[0:count, ..., None])
                self.images_amp.extend(da[0:count, ..., None])
                self.targets.extend([i] * count)
            else:
                raise ValueError(type_)

        print(f"Loaded {len(self.targets)} images")

    def __len__(self):
        return len(self.targets)

    def __must_filter(self, i) -> bool:
        cond1 = self.filter_mode == "exclude"
        cond2 = i in self.filter_labels
        return cond1 == cond2

    def __getitem__(self, idx):        
        
        # if user requests both channels, return phase/amp concatenated image
        if self.image_type == "phase":
            image = self.images_phs[idx]
        elif self.image_type == "amp":
            image = self.images_amp[idx]
        elif self.image_type == "both":
            image = np.concatenate([self.images_phs[idx], self.images_amp[idx]], dim=-1)
        else:
            raise ValueError(self.image_type)
        
        orig = self.targets[idx]

        if self.transform:
            image = self.transform(image)
        
        target = orig
        if self.target_transform:
            target = self.target_transform(target)

        return image, target, orig