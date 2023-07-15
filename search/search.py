import argparse
import os
import time
from pathlib import Path
from urllib.request import urlretrieve
from typing import Literal, Dict, Any

import faiss
import h5py
import numpy as np

from antihub_remover import AntihubRemover
from ep_searcher import (
    EPSearcher,
    EPSearcherOriginal,
    EPSearcherRandom,
    EPSearcherKmeans,
)


Size = Literal["100K", "300K", "10M", "30M", "100M"]
EntryPointSearch = Literal[
    "original",
    "random",
    "kmeans",
]
size2params: Dict[str, Dict[str, Any]] = {
    "300K": {
        "ep_search": "kmeans",
        "n_ep": 12,
        "pca_dim": 636,
        "search_L": 21,
        "alpha": 0.9628679294098882,
    },
    "10M": {
        "ep_search": "kmeans",
        "n_ep": 29,
        "pca_dim": 600,
        "search_L": 29,
        "alpha": 0.9772473945888052,
    },
    "30M": {
        "ep_search": "kmeans",
        "n_ep": 21,
        "pca_dim": 636,
        "search_L": 34,
        "alpha": 0.9990569475035961,
    },
    "100M": {
        "ep_search": "kmeans",
        "n_ep": 21,
        "pca_dim": 636,
        "search_L": 34,
        "alpha": 0.9990569475035961,
    },
}


def download(src, dst):
    if not os.path.exists(dst):
        os.makedirs(Path(dst).parent, exist_ok=True)
        print("downloading %s -> %s..." % (src, dst))
        urlretrieve(src, dst)


def prepare(kind: str, size: Size):
    url = "https://sisap-23-challenge.s3.amazonaws.com/SISAP23-Challenge"
    task = {
        "query": f"{url}/public-queries-10k-{kind}.h5",
        "dataset": f"{url}/laion2B-en-{kind}-n={size}.h5",
    }

    for version, url in task.items():
        download(url, os.path.join("data", kind, size, f"{version}.h5"))


def store_results(dst, algo, kind: str, D, I, buildtime, querytime, params, size):
    os.makedirs(Path(dst).parent, exist_ok=True)
    f = h5py.File(dst, "w")
    f.attrs["algo"] = algo
    f.attrs["data"] = kind
    f.attrs["buildtime"] = buildtime
    f.attrs["querytime"] = querytime
    f.attrs["size"] = size
    f.attrs["params"] = params
    f.create_dataset("knns", I.shape, dtype=I.dtype)[:] = I
    f.create_dataset("dists", D.shape, dtype=D.dtype)[:] = D
    f.close()


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


def run(
    kind: str,
    key: str,
    algo: str, # e.g. "NSG32,Flat
    /,
    size: Size = "100K",
    k: int = 10,
    alpha: float = 0.5,
    n_ep: int = 8,
    ep_search: EntryPointSearch = "original",
    outdir: str = "result",
):
    print("Running", kind, algo)

    prepare(kind, size)

    # load & normalize data and queries
    data_h5 = h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")
    queries_h5 = h5py.File(os.path.join("data", kind, size, "query.h5"), "r")
    data = np.array(data_h5[key]).astype("float32")
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    queries = np.array(queries_h5[key]).astype("float32")
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    _, d = data.shape

    # removing antihub
    do_removing_antihub = alpha < 1.0
    if do_removing_antihub:
        ahr = AntihubRemover(k=16, d=d)
        print(f"removing antihub... (alpha = {alpha:.3f})")
        print(f"Original DB Size: {data.shape}")
        sz = data.shape[0] // 10
        data, hub2id = ahr.remove_approximated_antihub(
            data[:sz], data, alpha=alpha, n_cluster=512, return_vecs=True
        )
        print(f"Reduced DB Size: {data.shape}")

    index_identifier = algo

    # dimension reduction with PCA
    do_reduction = algo.startswith("PCA")
    print("do_reduction", do_reduction)
    start = time.time()
    if do_reduction:
        pca_id = algo.split(",")[0]
        index_identifier = ",".join(algo.split(",")[1:])  # exclude PCA
        reduced_d = int(pca_id[3:])

        # reduce data dim
        print(f"before reduction: data.shape = {data.shape}")
        mat = faiss.PCAMatrix(d, reduced_d)
        mat.train(data)
        assert mat.is_trained
        data = mat.apply_py(data)
        print(f"after reduction: data.shape = {data.shape}")

        # reduce query dim
        print(f"before reduction: queries.shape = {queries.shape}")
        queries = mat.apply_py(queries)
        print(f"after reduction: queries.shape = {queries.shape}")

    # init index instance
    index_dim = data.shape[1]
    index = faiss.index_factory(index_dim, index_identifier)

    # training
    print(f"Training index on {data.shape}")
    index.train(data)
    index.add(data)
    assert index.is_trained

    # setup entrypoint searcher
    cur_ep = index.nsg.enterpoint
    ep_searcher = get_ep_searcher(data, cur_ep, ep_search, n_ep)

    elapsed_build = time.time() - start
    print(f"Done training in {elapsed_build}s.")

    param_name = "search_L"
    param = args.search_l

    print(f"Starting search on {queries.shape} with {param_name}={param}")
    start = time.time()
    index.nsg.search_L = param

    if ep_search != "original":
        # search with optimized entrypoint
        D = np.zeros((queries.shape[0], k), dtype=np.float32)
        I = np.zeros((queries.shape[0], k), dtype=np.int64)

        epts = ep_searcher.search(queries)
        for ep in np.unique(epts):
            selector = epts == ep
            query_vec = queries[selector, :]
            index.nsg.enterpoint = int(ep)
            D[selector, :], I[selector, :] = index.search(query_vec, k)
    else:
        D, I = index.search(queries, k)

    if do_removing_antihub:
        # restore antihub
        I = hub2id[I]

    elapsed_search = time.time() - start
    print(f"Done searching in {elapsed_search}s.")

    I = I + 1  # FAISS is 0-indexed, groundtruth is 1-indexed

    identifier = f"index=({algo}),query=({param_name}={param}),alpha={alpha},threads={faiss.omp_get_max_threads()},ep={ep_search}"

    store_results(
        os.path.join(outdir, kind, size, f"{identifier}.h5"),
        identifier,  # algo
        kind,
        D,
        I,
        elapsed_build,
        elapsed_search,
        identifier,
        size,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algo",
        default="NSG32,Flat",
        type=str,
        help="A list of algorithm names to run",
    )
    parser.add_argument(
        "--size",
        default="100K",
        type=str,
        help="The number of samples in dataset",
        choices=["100K", "300K", "10M", "30M", "100M"],
    )
    parser.add_argument("-k", default=10, type=int)
    parser.add_argument(
        "--alpha",
        default=1.0,
        type=float,
        help="alpha for antihub removal",
    )
    parser.add_argument(
        "--ep",
        default=None,
        type=str,
        help="initialization method for graph index entrypoint",
    )
    parser.add_argument(
        "--n-ep",
        default=8,
        type=int,
        help="number of entrypoint candidates",
    )
    parser.add_argument(
        "--threads",
        default=4,
        type=int,
        help="number of threads",
    )
    parser.add_argument(
        "--search-l",
        default=24,
        type=int,
        help="search_L for NSG",
    )
    parser.add_argument(
        "--outdir",
        default="result",
        type=str,
        help="output directory",
    )

    args = parser.parse_args()

    # validate args
    assert 0.0 <= args.alpha <= 1.0
    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

    # set number of threads
    faiss.omp_set_num_threads(args.threads)
    print(f"set number of threads to {args.threads}")

    # run search
    run(
        "clip768v2",
        "emb",
        args.algo,
        args.size,
        args.k,
        args.alpha,
        args.n_ep,
        args.ep,
        args.outdir,
    )  # NOTE: the naming convention of key is different from other datasets
