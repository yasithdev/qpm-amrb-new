import pandas as pd

from config import load_config, Config
from models import get_model_optimizer_and_loops
from models.common import load_saved_state
from copy import copy


def main(
    df: pd.DataFrame,
    config: Config,
) -> None:

    for row in df:
        # model, optim, train_model, test_model = get_model_optimizer_and_loops(config)
        print(row)

    # # load saved model and optimizer, if present
    # load_saved_state(
    #     model=model,
    #     optim=optim,
    #     experiment_path=experiment_path,
    #     config=config,
    # )
    # model = model.float().to(config.device)

    # train_labels = list(config.train_loader.dataset.labels)
    # test_labels = list(config.test_loader.dataset.labels)
    # labels = (
    #     [*train_labels, *test_labels] if config.cv_mode == "leave-out" else train_labels
    # )

    # # testing loop
    # stats = test_model(
    #     model=model,
    #     epoch=epoch,
    #     config=config,
    # )


if __name__ == "__main__":

    src_config = load_config()

    file_name = "results/"

    # K-Fold - Species
    df_kf_sp = pd.read_csv("results/AMRB2_cls_summary_k-fold_species.csv")
    config = copy(src_config)
    config.cv_mode = "k-fold"
    config.label_type = "species"
    main(df_kf_sp, config)

    # K-Fold - Strain
    df_kf_st = pd.read_csv("results/AMRB2_cls_summary_k-fold_strain.csv")
    config = copy(src_config)
    config.cv_mode = "k-fold"
    config.label_type = "strain"
    main(df_kf_st, config)

    # Leave-Out - Species
    df_lo_sp = pd.read_csv("results/AMRB2_cls_summary_leave-out_species.csv")
    config = copy(src_config)
    config.cv_mode = "leave-out"
    config.label_type = "species"
    main(df_lo_sp, config)

    # Leave-Out - Strain
    df_lo_st = pd.read_csv("results/AMRB2_cls_summary_leave-out_strain.csv")
    config = copy(src_config)
    config.cv_mode = "leave-out"
    config.label_type = "strain"
    main(df_lo_st, config)
