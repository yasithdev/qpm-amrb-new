import json

from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm

from config import load_config

def main(dataset_name: str, ood: str):
    
    import numpy as np
    import torch

    # initialize the RNG deterministically
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('medium')

    # initialize data attributes and loaders
    config = load_config(dataset_name=dataset_name, ood=ood)
    # use apply_target_transform=False to get unpermuted labels from ood splits
    config.load_data(shuffle_training_data=False, apply_target_transform=False)
    config.print_labels()
    dm = config.datamodule
    assert dm

    # initialize data modules
    dm.setup("fit")
    dm.setup("test")
    dm.setup("predict")

    # create data loaders
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()
    if ood:
        predict_loader = dm.predict_dataloader()

    val_labels = []
    ind_labels = []
    ood_labels = []
    
    for batch_idx, batch in enumerate(tqdm(val_loader)):
        x, y, *extra = batch
        if len(extra) > 0:
            l = extra[0].long().tolist()
        else:
            l = y.long().tolist()
        val_labels.extend(l)
    print("val - OK")

    for batch_idx, batch in enumerate(tqdm(test_loader)):
        x, y, *extra = batch
        if len(extra) > 0:
            l = extra[0].long().tolist()
        else:
            l = y.long().tolist()
        ind_labels.extend(l)
    print("test InD - OK")

    if ood:
        for batch_idx, batch in enumerate(tqdm(predict_loader)):
            x, y, *extra = batch
            if len(extra) > 0:
                l = extra[0].long().tolist()
            else:
                l = y.long().tolist()
            ood_labels.extend(l)
        print("test OoD - OK")

    index_labels = {
        "dataset_name": dataset_name,
        "ood": ood,
        "val_labels": val_labels,
        "ind_labels": ind_labels,
        "ood_labels": ood_labels,
    }
    
    return index_labels

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--ood", type=str)
    args = parser.parse_args()
    
    result = main(args.dataset_name, args.ood)
    with open(f"assets/stats/{datetime.now()}.json", "w") as f:
        json.dump(result, f)
