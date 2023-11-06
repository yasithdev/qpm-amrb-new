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
        print(f"RBC Dataset V.2 => All pateints")
        print(f"Dataset split type {type_}, image type: {image_type}")

        ### Extract directories of all files to a dictionary (key: class (strain), value: list of files)
        dirs = {}
        dirs_amp = {}
        
        all_files = glob.glob(f"{data_dir}/RBC_all_patients/{self.type_}/phase/*.npy")
        
        for x in all_files:
            # read patient id, embedded in filename (e.g. "0.npy")
            patient = int(x.split("/")[-1].split(".")[0].split("_")[0])
            
            amp_dir  = f"{data_dir}/RBC_all_patients/{self.type_}/amp/{patient}_amp.npy"
            
            data = np.load(x)
            data_amp = np.load(amp_dir) #amplitude counterpart
            
            assert data.shape == data_amp.shape
            
            dirs[patient] = data
            dirs_amp[patient] = data_amp

        self.images = []
        self.images_amp = []
        
        self.targets = []
        for i in range(0, len(dirs.keys())):  # iterate through patients
            if self.__must_filter(i):
                continue
            
            count = dirs[i].shape[0]

            # NOTE! to test, only use a small portion of data (e.g. 10% => int(count*0.1) )
            
            # #stack phase and amplitude
            # temp_img = np.stack((dirs[i], dirs_amp[i]), axis=1)
            
            self.images.extend(dirs[i][: int(count), ..., None])
            self.images_amp.extend(dirs_amp[i][: int(count), ..., None])
            self.targets.extend([i] * int(count))

        print(f"Loaded {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __must_filter(self, i) -> bool:
        cond1 = self.filter_mode == "exclude"
        cond2 = i in self.filter_labels
        return cond1 == cond2

    def __getitem__(self, idx):
        image, image_amp, orig = self.images[idx], self.images_amp[idx], self.targets[idx]

        if self.transform:
            image     = self.transform(image)
            image_amp = self.transform(image_amp)
        
        # if user requests both channels
        # Concat two amp and phase images as a single image, else return relevant image
        if(self.image_type == "both"):
            image = torch.cat((image, image_amp), dim=0)
        elif(self.image_type == "amp"):
            image = image_amp
        
        target = orig
        if self.target_transform:
            target = self.target_transform(target)

        return image, target, orig