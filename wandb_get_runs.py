import pandas as pd

import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>
runs = api.runs("yasith/robustml")


def list_to_dict(input: list[dict]) -> dict:
    keys = sorted(set(k for dic in input for k in dic))
    return {k: [dic.get(k, "") or "" for dic in input] for k in keys}


summary_list, config_list, identi_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append({k: v for k, v in run.summary._json_dict.items() if not k.startswith("gradient") and not k.startswith("parameters")})

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_") and not k in ["permutations", "labels", "datamodule", "data_dir", "emb_dir", "src_labels", "opt"]})

    # .name is the human-readable name of the run.
    identi_list.append({"name": run.name, "id": run.id})

identi_dict = list_to_dict(identi_list)
config_dict = list_to_dict(config_list)
summary_dict = list_to_dict(summary_list)
runs_df = pd.DataFrame({**identi_dict, **config_dict, **summary_dict}).set_index("id")

print(runs_df)
runs_df.to_csv("runs.csv")
