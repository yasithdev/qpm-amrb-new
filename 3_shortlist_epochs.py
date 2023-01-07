import pandas as pd
import wandb
import wandb.apis
import itertools
from typing import List

if __name__ == "__main__":

    api = wandb.Api()

    datasets = ["AMRB_2"]
    label_types = ["strain", "species"]
    cv_modes = ["leave-out"]
    models = ["resnet", "drcaps"]

    agg_results = {}
    
    combinations = itertools.product(datasets, label_types, models, cv_modes)

    for (dataset, label_type, model, cv_mode) in combinations:
        runs: List[wandb.apis.public.Run] = api.runs(
            path="yasith/qpm-amrb",
            filters={
                "config.dataset": dataset,
                "config.label_type": label_type,
                "config.model": model,
                "config.cv_mode": cv_mode,
            },
        )

        for run in runs:
            run_name = run.name
            print(run_name)

            min_metrics = [
                "train_loss",
                "test_loss",
            ]
            max_metrics = [
                "train_accuracy",
                "test_accuracy",
                "train_top2accuracy",
                "test_top2accuracy",
                "train_top3accuracy",
                "test_top3accuracy",
            ]

            df = pd.DataFrame(run.history())
            locs = set()
            for metric in min_metrics:
                locs.add(df[metric].argmin())
            for metric in max_metrics:
                locs.add(df[metric].argmax())
            
            sub_df = df.loc[sorted(locs),[*min_metrics, *max_metrics]]
            print(sub_df)

            agg_results[run_name] = sub_df

