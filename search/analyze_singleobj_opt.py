import os
import optuna
import argparse

import pandas as pd
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", type=str, help="path to study")
    parser.add_argument("--eps", type=float, default=0.01, help="threshold buffer")
    parser.add_argument("--topk", type=int, default=10, help="topk record")

    arg = parser.parse_args()

    study_name: str = os.path.basename(arg.study)
    study_name = study_name.split(".db")[0]
    print(f"loading study {study_name} ...")
    storage_name = f"sqlite:///{arg.study}"
    study = optuna.study.load_study(
        study_name=study_name,
        storage=storage_name,
    )

    thresh = 0.90 + arg.eps
    print(f"Threshold of recall = {thresh}")

    df: pd.DataFrame = study.trials_dataframe()

    # filter by threshold
    df = df[df["user_attrs_recall"] >= thresh]
    df = df.sort_values(by=["user_attrs_qps"], ascending=False)
    df = df.head(arg.topk)
    param_keys = []
    for key in df.columns:
        if "param" in key:
            param_keys.append(key)
    # print(df[param_keys])

    for i in range(arg.topk):
        row = df.iloc[i]
        print(f"============ Best {i+1}th ===============")
        print(f"qps = {row['user_attrs_qps']} [1/s]")
        print(f"recall = {row['user_attrs_recall']}")
        for key in param_keys:
            print(f"{key} = {row[key]}")
