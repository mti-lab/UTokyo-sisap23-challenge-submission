import os
import optuna
import optuna.visualization
import argparse

import pandas as pd
import numpy as np
import plotly

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

    pareto_front = optuna.visualization.plot_pareto_front(
        study, target_names=["recall", "qps"]
    )
    pareto_front.show()

    imp_qps = optuna.visualization.plot_param_importances(
        study, target=lambda t: t.values[1], target_name="qps"
    )
    imp_qps.show()

    imp_recall = optuna.visualization.plot_param_importances(
        study, target=lambda t: t.values[0], target_name="recall"
    )
    imp_recall.show()
