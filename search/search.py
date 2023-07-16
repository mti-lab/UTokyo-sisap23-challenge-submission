import argparse
import os
import time
from pathlib import Path
from urllib.request import urlretrieve
from typing import Literal, Dict, Any, List, Optional, Tuple

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
# Build parameter (fixed)
size2build_param: Dict[str, Dict[str, Any]] = {
    "300K": {
        "n_ep": 12,
        "pca_dim": 636,
        "alpha": 0.9628679294098882,
        "threads": 64,
    },
    "10M": {
        "n_ep": 12,
        "pca_dim": 732,
        "alpha": 0.9344174472408312,
        "threads": 64,
    },
    "30M": {
        "n_ep": 12,
        "pca_dim": 732,
        "alpha": 0.9344174472408312,
        "threads": 64,
    },
    "100M": {
        "n_ep": 12,
        "pca_dim": 768,
        "alpha": 0.9344174472408312,
        "threads": 64,
    },
}
# Runtime parameters to be sweeped.
# Note that sweeping does not require re-build index.
size2runtime_params: Dict[str, List[Dict[str, Any]]] = {
    "300K": [
        {"search_L": x, "ep_search_mode": e, "threads": t}
        for x in range(20, 22)
        for e in ["original", "kmeans"]
        for t in [64, 48, 32, 16]
    ],
    "10M": [
        {"search_L": x, "ep_search_mode": e, "threads": t}
        for x in range(40, 57)
        for e in ["original", "kmeans"]
        for t in [64, 48, 32, 16]
    ],
    "30M": [
        {"search_L": x, "ep_search_mode": e, "threads": t}
        for x in range(56, 77)
        for e in ["original", "kmeans"]
        for t in [64, 48, 32, 16]
    ],
    "100M": [
        {"search_L": x, "ep_search_mode": e, "threads": t}
        for x in range(70, 100)
        for e in ["original", "kmeans"]
        for t in [64, 48, 32, 16]
    ],
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


def run_search(
    index: faiss.Index,
    data: np.ndarray,  # [N_DATA, d]
    query: np.ndarray,  # [N_QUERY, d]
    k: int,
    pca_mat: Optional[faiss.PCAMatrix],
    ep_searcher: EPSearcher,
    search_L: int,
    hub2id: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    if pca_mat is not None:
        # PCA for query
        query = pca_mat.apply_py(query)

    index.nsg.search_L = search_L

    D = np.empty((query.shape[0], k), dtype=np.float32)
    I = np.empty((query.shape[0], k), dtype=np.int64)
    epts = ep_searcher.search(query)
    for ep in np.unique(epts):
        selector = epts == ep
        query_vec = query[selector, :]
        index.nsg.enterpoint = int(ep)
        D[selector, :], I[selector, :] = index.search(query_vec, k)

    if hub2id is not None:
        # restore hub ids
        I = hub2id[I]

    return D, I


def build_index(
    data: np.ndarray,
    build_param: Dict[str, Any],
) -> Dict[str, Any]:
    # parse param:
    alpha: float = build_param["alpha"]
    # removing antihub
    do_removing_antihub = alpha < 1.0
    hub2id = None
    if do_removing_antihub:
        d = data.shape[1]
        ahr = AntihubRemover(k=16, d=d)
        print(f"removing antihub... (alpha = {alpha:.3f})")
        print(f"Original DB Size: {data.shape}")
        # sz = data.shape[0] / 10
        sz = data.shape[0] // 10
        data, hub2id = ahr.remove_approximated_antihub(
            data[:sz, :], data, alpha=alpha, n_cluster=512, return_vecs=True
        )
        print(f"Reduced DB Size: {data.shape}")

    # pca
    pca_dim: int = build_param["pca_dim"]
    orig_dim = data.shape[1]
    do_pca = pca_dim < orig_dim
    pca_mat = None
    if do_pca:
        # do dim reduction
        pca_mat = faiss.PCAMatrix(orig_dim, pca_dim)
        pca_mat.train(data)
        assert pca_mat.is_trained
        print(f"before reduction: data.shape = {data.shape}")
        data = pca_mat.apply_py(data)
        print(f"after reduction: data.shape = {data.shape}")

    # build index
    index_dim = data.shape[1]
    index = faiss.IndexNSGFlat(index_dim, 32)
    index.train(data)
    assert index.is_trained
    index.add(data)

    # set up ep searcher (original, kmeans)
    original_ep_searcher = EPSearcherOriginal(data, index.nsg.enterpoint)
    kmeans_ep_searcher = EPSearcherKmeans(
        data, index.nsg.enterpoint, build_param["n_ep"]
    )

    return {
        "data": data,
        "index": index,
        "pca_mat": pca_mat,
        "original_ep_seracher": original_ep_searcher,
        "kmeans_ep_searcher": kmeans_ep_searcher,
        "hub2id": hub2id,
    }


def run(
    kind: str,  # e.g. "clip768v2"
    key: str,  # e.g. "emb"
    size: Size,  # 100K, 300K, 10M, 30M, 100M
    k: int = 10,
    outdir: str = "result",
):
    algo = "NSG32,Flat"
    print(f"Running {kind} (size={size}) (algo={algo})")
    build_param = size2build_param[size]
    n_threads = build_param["threads"] if "threads" in build_param else 64
    faiss.omp_set_num_threads(n_threads)
    print(f"set number of threads to {n_threads} for build")

    # prepare data
    prepare(kind, size)

    # load & normalize data and queries
    data_h5 = h5py.File(os.path.join("data", kind, size, "dataset.h5"), "r")
    query_h5 = h5py.File(os.path.join("data", kind, size, "query.h5"), "r")
    data = np.array(data_h5[key]).astype("float32")
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    query = np.array(query_h5[key]).astype("float32")
    query /= np.linalg.norm(query, axis=1, keepdims=True)

    # build index
    ### begin build section
    build_start = time.time()
    build_output = build_index(data, build_param)
    build_duration = time.time() - build_start
    ### end build section

    # parse build output
    data = build_output["data"]
    index = build_output["index"]
    pca_mat = build_output["pca_mat"]
    original_ep_searcher = build_output["original_ep_seracher"]
    kmeans_ep_searcher = build_output["kmeans_ep_searcher"]
    hub2id = build_output["hub2id"]
    ep_searchers = {
        "original": original_ep_searcher,
        "kmeans": kmeans_ep_searcher,
    }

    # run search (sweep runtime param candidates)
    runtime_params = size2runtime_params[size]
    for param in runtime_params:
        print(f"Running search with {param}")
        ep_searcher = ep_searchers[param["ep_search_mode"]]
        runtime_n_threads = param["threads"] if "threads" in param else 64
        faiss.omp_set_num_threads(runtime_n_threads)
        print(f"set number of threads to {runtime_n_threads} for runtime")

        ### begin search section
        search_start = time.time()
        D, I = run_search(
            index=index,
            data=data,
            query=query,
            k=k,
            pca_mat=pca_mat,
            ep_searcher=ep_searcher,
            search_L=param["search_L"],
            hub2id=hub2id,
        )
        search_duration = time.time() - search_start
        ### end search section

        # faiss is 0-indexed, groundtruth is 1-indexed
        I = I + 1

        # store results
        preprocess_algo = f"PCA{build_param['pca_dim']}" if pca_mat is not None else ""
        index_algo = f"{preprocess_algo},{algo}"
        identifier = f"index=({index_algo}),query=(search_L={param['search_L']}),build=(alpha={build_param['alpha']},pca_dim={build_param['pca_dim']}),threads={faiss.omp_get_max_threads()},ep={param['ep_search_mode']}"
        params_label = f"query=(search_L={param['search_L']},ep={param['ep_search_mode']},threads={faiss.omp_get_max_threads()})"
        store_results(
            dst=os.path.join(outdir, kind, size, f"{identifier}.h5"),
            algo=identifier,
            kind=kind,
            D=D,
            I=I,
            buildtime=build_duration,
            querytime=search_duration,
            params=params_label,
            size=size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--size",
        default="100K",
        type=str,
        help="The number of samples in dataset",
        choices=["100K", "300K", "10M", "30M", "100M"],
    )
    parser.add_argument(
        "-k",
        default=10,
        type=int,
        help="number of nearest neighbors",
    )

    args = parser.parse_args()

    # validate args
    assert args.size in ["100K", "300K", "10M", "30M", "100M"]

    # run session
    run(
        kind="clip768v2",
        key="emb",
        size=args.size,
        k=args.k,
        outdir="result",
    )
