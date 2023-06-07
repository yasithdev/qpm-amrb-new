import pandas as pd

if __name__ == "__main__":

    cols = [
        "train_loss",
        "test_loss",
        "train_accuracy",
        "test_accuracy",
        "train_top2accuracy",
        "test_top2accuracy",
        "train_top3accuracy",
        "test_top3accuracy",
    ]

    df_kf_sp = pd.read_csv("results/AMRB2_cls_summary_k-fold_species.csv")
    df_kf_st = pd.read_csv("results/AMRB2_cls_summary_k-fold_strain.csv")
    df_lo_sp = pd.read_csv("results/AMRB2_cls_summary_leave-out_species.csv")
    df_lo_st = pd.read_csv("results/AMRB2_cls_summary_leave-out_strain.csv")

    # K-Fold - Species
    results_kf_sp = (
        df_kf_sp.groupby("model")[cols]
        .agg(lambda x: f"{x.mean():.5f} ± {x.std():.5f}")
        .T
    )
    print("\nAMRB2 K-Fold (Species)")
    print(results_kf_sp)

    # K-Fold - Strain
    results_kf_st = (
        df_kf_st.groupby("model")[cols]
        .agg(lambda x: f"{x.mean():.5f} ± {x.std():.5f}")
        .T
    )
    print("\nAMRB2 K-Fold (Strain)")
    print(results_kf_st)

    # Leave-Out - Strain
    results_lo_sp = (
        df_lo_sp.groupby("model")[cols]
        .agg(lambda x: f"{x.mean():.5f} ± {x.std():.5f}")
        .T
    )
    print("\nAMRB2 Leave-Out (Species)")
    print(results_lo_sp)

    # Leave-Out - Strain
    results_lo_st = (
        df_lo_st.groupby("model")[cols]
        .agg(lambda x: f"{x.mean():.5f} ± {x.std():.5f}")
        .T
    )
    print("\nAMRB2 Leave-Out (Strain)")
    print(results_lo_st)
