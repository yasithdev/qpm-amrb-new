import pandas as pd
import wandb
import wandb.apis
import itertools
from typing import List

if __name__ == "__main__":

    api = wandb.Api()

    datasets = ["AMRB2"]
    label_types = ["strain", "species"]
    cv_modes = ["leave-out", "k-fold"]
    models = ["resnet", "drcaps"]

    summary_df = pd.DataFrame()

    combinations = itertools.product(datasets, label_types, models, cv_modes)

    pd.set_option("expand_frame_repr", False)

    for (dataset, label_type, model, cv_mode) in combinations:
        runs: List[wandb.apis.public.Run] = api.runs(
            path="yasith/qpm-amrb-v2",
            filters={
                "config.dataset": dataset,
                "config.label_type": label_type,
                "config.model": model,
                "config.cv_mode": cv_mode,
            },
        )

        for run in runs:

            print(run.name)
            cv_k = int(str(run.name).split("-")[-1])

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

            instance_df = df.loc[sorted(locs), [*min_metrics, *max_metrics]]
            instance_df["dataset"] = dataset
            instance_df["label_type"] = label_type
            instance_df["model"] = model
            instance_df["cv_mode"] = cv_mode
            instance_df["cv_k"] = cv_k
            instance_df.index.name = "step"
            instance_df = instance_df.reset_index()
            instance_df = instance_df.set_index(
                ["dataset", "label_type", "model", "cv_mode", "cv_k", "step"]
            )
            instance_df = instance_df.dropna(axis=0)

            summary_df = pd.concat([summary_df, instance_df])

    print("writing CLS summary to CSV")
    summary_df = summary_df.sort_index()
    summary_df.to_csv("results/AMRB2_cls_raw_summary.csv")

    print("DONE")
