import glob
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F

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


class bacteria_dataset(torch.utils.data.Dataset):
    """
    A standard dataset class to get the bacteria dataset

    Args:
        data_dir   : data directory which contains data hierarchy
        type_      : whether dataloader if train/ val
        transform  : torchvision.transforms
        label_type : There are multiple types of classification in bacteria dataset
                     therefore, specify which label you need as follows:
                        | label_type              | Description
                        |------------------------ |---------------
                        | 'class' (default)       | Strain (0-20)
                        | 'antibiotic_resistant'  | Non wild type (1) / Wild type (0)
                        | 'gram_strain'           | Gram Positive (1) / Gram Negative (0)
                        | 'species'               | Species (0-4)
        balance_data    : If true, dataset will be balanced by the minimum class count (default: False)
        expand_channels : If true, bacteria image will be copied to 3 channels  (default: False)
                          (used for some predefined backbones which need RGB images)
    """

    def __init__(
        self,
        data_dir: str,
        type_="train",
        transform=None,
        label_type="class",
        balance_data=False,
        expand_channels=False,
        one_hot=False,
    ):
        self.transform = transform
        self.label_type = label_type
        self.type_ = type_
        self.expand_channels = expand_channels
        self.one_hot = one_hot

        all_dirs = sorted(
            glob.glob(f"{data_dir}/QPM_{type_}/*/*"),
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
        data = np.load(self.img_dirs[idx], allow_pickle=True)
        image = data[0]

        label = self.__getclass_(data[1], self.label_type)

        if self.transform:
            image = self.transform(image)

        if self.expand_channels:
            image = image.expand(3, image.shape[1], image.shape[1])

        if self.one_hot:
            target = torch.zeros(len(self.labels), dtype=torch.float32)
            target[label] = 1
        else:
            target = label

        return image, target


# class bacteria_dataset_selective(torch.utils.data.Dataset):
#     '''
#         A standard dataset class to get the bacteria dataset

#         Args:
#             data_dir  : data directory which contains data hierarchy
#             type_     : whether dataloader if train/ val
#             transform : torchvision.transforms
#     '''

#     def __init__(self, data_dir='datasets/bacteria_np', type_= 'train', transform= None, label_type = "class", expand_channels = False, isolate_class = False):
#         self.transform= transform
#         self.label_type = label_type
#         self.type_ = type_
#         self.expand_channels = expand_channels

#         #load dictionary which contains metadata of all bacteria images
#         with open('/n/home12/ramith/FYP/bacteria-classification/saved_dictionary.pkl', 'rb') as f:
#             global_dict = pickle.load(f)

#         #get all image paths in 'train', 'val' or 'test' folder
#         all_dirs = sorted(glob.glob(f'{data_dir}/{type_}/*/*'), key= lambda x: int(x.split('/')[-1][:-4]))

#         print(f"Dataset type {type_}; dataloader will have label type: {label_type}", end = " -> ")
#         print(f"All files = {len(all_dirs)}")


#         img_dirs_filtered = []

#         for i,x in enumerate(all_dirs):
#             #data  = np.load(x, allow_pickle=True)[1]
#             class_ = global_dict[x.split('/')[-1]]['class'] #always batching will be done along the same strain class
#             #class_ = self.__getclass_(data, self.label_type)

#             if(class_ == isolate_class): # only select the class needed
#                 img_dirs_filtered.append(x)

#         self.img_dirs = img_dirs_filtered

#         print(f"Loaded {len(self.img_dirs)} images only from class {isolate_class}")


#     def __len__(self):
#         return len(self.img_dirs)

#     def __getclass_(self, meta_data, label_type):
#         if(label_type == 'class'):
#             return meta_data[0]

#         elif(label_type == 'antibiotic_resistant'):
#             '''
#                 Dataset wild_type equals to class 1
#                 antibiotic_resistance is when class is => not wild_type
#             '''
#             antibiotic_resistance = int(not(meta_data[1]))
#             return antibiotic_resistance

#         elif(label_type == 'gram_strain'):
#             return meta_data[2]

#         elif(label_type == 'species'):
#             return species_mapping_dict[meta_data[0]] #map class to species

#         else:
#             raise Exception("Invalid label type")

#     def __getitem__(self, idx):
#         data  = np.load(self.img_dirs[idx], allow_pickle=True)
#         image = data[0]


#         label = self.__getclass_(data[1], self.label_type)

#         if self.transform:
#             image = self.transform(image)

#         if(self.expand_channels):
#             image = image.expand(3, image.shape[1], image.shape[1])

#         return image, label
