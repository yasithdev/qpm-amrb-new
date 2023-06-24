import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .bac_datasets import bacteria_dataset


def get_bacteria_dataloaders(
    img_size=40,
    train_batch_size=32,
    test_batch_size=32,
    torch_seed=10,
    label_type="species",
    balanced_mode=False,
    expand_channels=True,
    data_dir="/home/pjaya001/datasets",
    one_hot=False,
):
    """
    Function to return train, validation QPM dataloaders
    Args:
        img_size         : Image size to resize
        train_batch_size : batch size for training
        torch_seed       : seed
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

        data_dir         : data directory which has the data hierachy as `./test/0/0_15612.npy`
    Returns:
        train_loader : Data loader for training
        val_loader   : Data loader for validation
    """
    if balanced_mode:
        print("Using balanced mode")

    torch.manual_seed(torch_seed)
    # transforms.ToPILImage(),
    my_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Resize((img_size, img_size), antialias=True)]
    )

    train_data = bacteria_dataset(
        data_dir=data_dir,
        type_="train",
        transform=my_transform,
        label_type=label_type,
        balance_data=balanced_mode,
        expand_channels=expand_channels,
        one_hot=one_hot,
    )
    # val_data   = bacteria_dataset(data_dir=data_dir, type_= 'val',   transform = my_transform, label_type = label_type, balance_data = balanced_mode , expand_channels = expand_channels)
    test_data = bacteria_dataset(
        data_dir=data_dir,
        type_="test",
        transform=my_transform,
        label_type=label_type,
        balance_data=balanced_mode,
        expand_channels=expand_channels,
        one_hot=one_hot,
    )

    train_loader = DataLoader(
        train_data,
        batch_size=train_batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
    )
    # val_loader   = DataLoader(val_data, batch_size  = 32, shuffle=True, drop_last= True, num_workers=2)
    test_loader = DataLoader(
        test_data, batch_size=test_batch_size, shuffle=True, drop_last=True, num_workers=2
    )

    return train_loader, test_loader


# def get_bacteria_eval_dataloaders(img_size, test_batch_size ,torch_seed=10, label_type = "class", expand_channels = False, data_dir= '/n/holyscratch01/wadduwage_lab/ramith/bacteria_processed', isolate_class = False):
#     '''
#         Function to return train, validation QPM dataloaders
#         Args:
#             img_size         : Image size to resize
#             train_batch_size : batch size for training
#             torch_seed       : seed
#             data_dir         : data directory which has the data hierachy as `./train/amp/00001.png`
#         Returns:
#             train_loader : Data loader for training
#             val_loader   : Data loader for validation
#     '''

#     torch.manual_seed(torch_seed)
#     # transforms.ToPILImage(),
#     my_transform= transforms.Compose([transforms.ToTensor(), transforms.Resize((img_size, img_size))])

#     test_data  = bacteria_dataset_selective(data_dir=data_dir, type_= 'test',  transform = my_transform, label_type = label_type, expand_channels = expand_channels, isolate_class = isolate_class)
#     test_loader  = DataLoader(test_data, batch_size = test_batch_size, shuffle=True, drop_last= True, num_workers=2)

#     dataset_sizes = {'test': len(test_loader)}

#     return test_loader, dataset_sizes
