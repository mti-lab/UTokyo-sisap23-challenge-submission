import argparse
from datetime import datetime
import os
from copy import deepcopy
import time
import timeit
from pathlib import Path
from urllib.request import urlretrieve
from typing import Any, Literal, Optional, Dict

import faiss
import h5py
import numpy as np
import optuna

from antihub_remover import AntihubRemover
from ep_searcher import (
    EPSearcher,
    EPSearcherOriginal,
    EPSearcherRandom,
    EPSearcherKmeans,
)


Algorithm = Literal["NSG32,Flat",]
Size = Literal["100K", "300K", "10M", "30M", "100M"]
EntryPointSearch = Literal[
    "original",
    "random",
    "kmeans",
]


def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print("downloading %s -> %s..." % (src, dst))
        urlretrieve(src, dst)


def prepare(kind: str, size: Size):
    url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    # url = "http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge"
    task = {
        "query": f"{url}/public-queries-10k-{kind}.h5",
        "dataset": f"{url}/laion2B-en-{kind}-n={size}.h5",
    }

    for version, url in task.items():
        download(url, os.path.join("data", kind, size, f"{version}.h5"))


def get_groundtruth(size: Size = "100K") -> np.ndarray:
    url = f"http://ingeotec.mx/~sadit/metric-datasets/LAION/SISAP23-Challenge/laion2B-en-public-gold-standard-v2-{size}.h5"

    out_fn = os.path.join("data", f"groundtruth-{size}.h5")
    download(url, out_fn)
    gt_f = h5py.File(out_fn, "r")
    true_I = np.array(gt_f["knns"])
    gt_f.close()
    return true_I


def get_ep_searcher(
    data: np.ndarray,
    cur_ep: int,
    ep_search: EntryPointSearch,
    n_ep: int,
) -> EPSearcher:
    if ep_search == "original":
        return EPSearcherOriginal(data, cur_ep)
    if ep_search == "random":
        return EPSearcherRandom(data, cur_ep, n_ep)
    if ep_search == "kmeans":
        return EPSearcherKmeans(data, cur_ep, n_ep)
    raise ValueError(f"Unknown ep_search: {ep_search}")


def calc_recall(
    I: np.ndarray,  # [N_QUERY, k]
    gt: np.ndarray,  # [N_QUERY, k]
    k: int,
) -> float:
    assert k <= I.shape[1]
    assert I.shape[0] == gt.shape[0]

    n = I.shape[0]
    recall = 0.0
    for i in range(n):
        recall += len(set(I[i, :k]) & set(gt[i, :k]))
    return recall / (n * k)


def run_search(
    index: faiss.Index,
    data: np.ndarray,  # [N_DATA, d]
    query: np.ndarray,  # [N_QUERY, d]
    gt: np.ndarray,  # [N_QUERY, k]
    pca_mat: faiss.PCAMatrix,
    ep_searcher: EPSearcher,
    param: Dict[str, Any],
) -> Dict[str, float]:
    start = time.time()
    if pca_mat is not None:
        # PCA for query
        query = pca_mat.apply_py(query)
        assert query.shape[1] == data.shape[1]

    index.nsg.search_L = param["search_L"]

    D = np.empty((query.shape[0], param["k"]), dtype=np.float32)
    I = np.empty((query.shape[0], param["k"]), dtype=np.int64)
    epts = ep_searcher.search(query)
    for ep in np.unique(epts):
        selector = epts == ep
        query_vec = query[selector, :]
        index.nsg.enterpoint = int(ep)
        D[selector, :], I[selector, :] = index.search(query_vec, param["k"])

    search_duration = time.time() - start

    I = I + 1  # 1-indexed

    recall = calc_recall(I, gt, param["k"])
    qps = query.shape[0] / search_duration

    return {
        "recall": recall,
        "qps": qps,
    }


def optimize_params(args):
    print(f"Running {args.kind} {args.size}")

    # prepare data
    prepare(args.kind, args.size)
    gt = get_groundtruth(args.size)

    # cache index
    # cache[size] = {pca_mat, data, index, ep_searcher}
    cache: Dict[str, Dict[str, Any]] = dict()

    # optuna objective function
    def objective(trial: optuna.Trial):
        nonlocal gt, args, cache
        # load data
        data_h5 = h5py.File(
            os.path.join("data", args.kind, args.size, "dataset.h5"), "r"
        )
        query_h5 = h5py.File(
            os.path.join("data", args.kind, args.size, "query.h5"), "r"
        )
        data = np.array(data_h5[args.key]).astype("float32")
        data /= np.linalg.norm(data, axis=1, keepdims=True)
        query = np.array(query_h5[args.key]).astype("float32")
        query /= np.linalg.norm(query, axis=1, keepdims=True)
        # setup parameters to be optimized
        ## index parameter
        # R = int(trial.suggest_categorical("R", ["32", "64"]))
        # metric_type_key = trial.suggest_categorical("metric_type", ["l2", "ip"])
        # metric_type = {
        #     "l2": faiss.METRIC_L2,
        #     "ip": faiss.METRIC_INNER_PRODUCT,
        # }[metric_type_key]
        ## runtime parameter
        ### for search
        search_L = trial.suggest_int("search_L", 14, 20)
        ### for entry point search
        # ep_search = trial.suggest_categorical("ep_search", ["original", "kmeans"])
        ep_search = "kmeans"
        n_ep = trial.suggest_int("n_ep", 4, 24)
        ### for PCA
        pca_dim = trial.suggest_int("pca_dim", 648, 768, step=8)
        ### for antihub removal
        # TODO: support antihub removal
        # alpha = trial.suggest_float("alpha", 0.9, 1.0)

        # pca mat
        build_start = time.time()

        pca_mat = None
        orig_d = data.shape[1]
        if args.size in cache:
            # use cache
            pca_mat = cache[args.size]["pca_mat"]
            assert pca_mat.is_trained
            data = cache[args.size]["data"]
        else:
            if pca_dim < orig_d:
                pca_mat = faiss.PCAMatrix(orig_d, pca_dim)
                pca_mat.train(data)
                assert pca_mat.is_trained
                # cache pca_mat
                cache[args.size]["pca_mat"] = deepcopy(pca_mat)
                data = pca_mat.apply_py(data)
                # cache data
                cache[args.size]["data"] = deepcopy(data)
        # build index
        dim = data.shape[1]
        if args.size in cache:
            # use cache
            index = cache[args.size]["index"]
            assert index.d == dim
            assert index.is_trained
        else:
            index = faiss.IndexNSGFlat(dim, 32)
            index.train(data)
            index.add(data)
            cache[args.size]["index"] = deepcopy(index)

        build_duration = time.time() - build_start
        trial.set_user_attr("buildtime", build_duration)

        # setup ep searcher
        cur_ep = index.nsg.enterpoint
        if args.size in cache:
            # use cache
            ep_searcher = cache[args.size]["ep_searcher"]
        else:
            ep_searcher = get_ep_searcher(data, cur_ep, ep_search, n_ep)
            cache[args.size]["ep_searcher"] = deepcopy(ep_searcher)

        param = {
            "search_L": search_L,
            "k": 10,
        }

        # run search
        # eval recall
        I = run_search(
            index=index,
            data=data,
            query=query,
            pca_mat=pca_mat,
            ep_searcher=ep_searcher,
            param=param,
        )
        recall = calc_recall(I, gt, param["k"])
        trial.set_user_attr("recall", recall)

        # eval qps
        single_loop = lambda: run_search(
            index=index,
            data=data,
            query=query,
            pca_mat=pca_mat,
            ep_searcher=ep_searcher,
            param=param,
        )
        n_timeit_loop = 10
        total_runtime = timeit.timeit(single_loop, number=n_timeit_loop)
        avg_runtime = total_runtime / n_timeit_loop
        avg_qps = query.shape[0] / avg_runtime
        trial.set_user_attr("qps", avg_qps)
        # get recall
        print(f"Trial {trial.number}: recall@10={recall:.4f}, QPS={avg_qps:.3f}")

        eps = 0.2
        thresh = 0.9 + eps
        constraint = thresh - recall  # recall@10 should be greater than 0.9
        trial.set_user_attr("constraints", (constraint,))
        return avg_qps

    def constraints(trial: optuna.Trial):
        return trial.user_attrs["constraints"]

    # create study
    os.makedirs(args.outdir, exist_ok=True)
    study_name = (
        f"study-{args.kind}-{args.size}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    storage_name = f"sqlite:///{os.path.join(args.outdir, study_name)}.db"
    sampler = optuna.samplers.TPESampler(
        constraints_func=constraints,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="maximize",
        sampler=sampler,
    )
    # set user attributes
    study.set_user_attr("kind", args.kind)
    study.set_user_attr("size", args.size)

    # run optimization
    study.optimize(objective, n_trials=args.n_trials)

    # show results
    trial = study.best_trial
    print(f"best trial: {trial.number}")
    print(f"recall@10: {trial.user_attrs['recall']}")
    print(f"QPS [1/s]: {trial.user_attrs['qps']}")
    print(f"build time [s]: {trial.user_attrs['buildtime']}")
    print("best params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        default="100K",
        type=str,
        help="The number of samples in dataset",
        choices=["100K", "300K", "10M", "30M", "100M"],
    )
    parser.add_argument("-k", default=10, type=int)
    parser.add_argument(
        "--threads",
        default=16,
        type=int,
        help="number of threads",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="random seed",
    )
    parser.add_argument(
        "--outdir",
        default="study",
        type=str,
        help="output directory",
    )
    parser.add_argument(
        "--n-trials",
        default=1000,
        type=int,
        help="number of trials",
    )

    args = parser.parse_args()
    args.kind = "clip768v2"
    args.key = "emb"

    # validate args
    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

    # set seed
    np.random.seed(args.seed)

    # set number of threads
    faiss.omp_set_num_threads(args.threads)
    print(f"set number of threads to {args.threads}")

    optimize_params(args)
